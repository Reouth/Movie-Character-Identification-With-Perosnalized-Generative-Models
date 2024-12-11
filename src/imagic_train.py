import logging
import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training script for Imagic embeddings.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--input_image", required=True, type=str, help="Path to input image to edit.")
    parser.add_argument("--target_text", type=str, help="The target text describing the output image.")
    parser.add_argument("--output_dir", type=str, default="text-inversion-model",
                        help="Directory for model predictions and checkpoints.")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility.")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Center crop images before resizing.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--emb_train_steps", type=int, default=500, help="Training steps for embeddings.")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Max training steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Steps to accumulate before backward pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory.")
    parser.add_argument("--emb_learning_rate", type=float, default=1e-3, help="Learning rate for embeddings.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for fine-tuning.")
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale learning rate by GPUs, batch size, and gradient accumulation.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam optimizer.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, help="Token for Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, help="Repository name for Hugging Face Hub.")
    parser.add_argument("--project_dir", type=str, default="logs",
                        help="Log directory for TensorBoard.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval steps.")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Use mixed precision training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")

    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    return args

class AverageMeter:
    """Tracks and computes the average of values."""
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_optimizer(args, params):
    """Initialize optimizer with specified parameters."""
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.Adam8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam.")
    else:
        optimizer_class = torch.optim.Adam

    return optimizer_class(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )

def prepare_image(image_path, resolution, center_crop, device):
    """Prepare input image for training."""
    image = Image.open(image_path).convert("RGB")
    transforms_list = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    image_tensor = transforms.Compose(transforms_list)(image)
    return image_tensor[None].to(device=device, dtype=torch.float32)


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=Path(args.output_dir, args.project_dir),
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name or args.pretrained_model_name_or_path)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Process input image
    init_image = prepare_image(args.input_image, args.resolution, args.center_crop, accelerator.device)
    with torch.inference_mode():
        init_latents = vae.encode(init_image).latent_dist.sample() * 0.18215

    # Encode target text
    text_ids = tokenizer(args.target_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    text_embeddings = text_encoder(text_ids.to(accelerator.device))[0].float()

    # Optimize embeddings
    optimized_embeddings = text_embeddings.clone().requires_grad_(True)
    optimizer = initialize_optimizer(args, [optimized_embeddings])
    unet, optimizer = accelerator.prepare(unet, optimizer)

    def train_embeddings():
        progress_bar = tqdm(range(args.emb_train_steps), desc="Embedding Optimization")
        for step in progress_bar:
            optimizer.zero_grad()
            noise = torch.randn_like(init_latents)
            timesteps = torch.randint(0, 1000, (init_latents.shape[0],), device=init_latents.device)
            noisy_latents = DDPMScheduler.add_noise(init_latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, optimized_embeddings).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

    train_embeddings()

    # Save embeddings
    torch.save(optimized_embeddings.cpu(), Path(args.output_dir, "optimized_embeddings.pt"))
    torch.save(text_embeddings.cpu(), Path(args.output_dir, "target_embeddings.pt"))

    if args.push_to_hub:
        repo = Repository(args.output_dir, clone_from=args.hub_model_id)
        repo.push_to_hub("End of training")

if __name__ == "__main__":
    main()
