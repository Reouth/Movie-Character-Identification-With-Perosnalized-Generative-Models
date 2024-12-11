import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from typing import List, Dict

class StableDiffusionPipeline:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        precision: str = "fp16",
    ):
        """
        Initialize the Stable Diffusion pipeline.

        Args:
            model_name_or_path (str): Path or name of the pretrained model.
            device (str): Device to run the model on (e.g., 'cuda' or 'cpu').
            precision (str): Mixed precision setting ('fp16' or 'fp32').
        """
        self.device = device
        self.precision = precision
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16 if precision == "fp16" else torch.float32
        )
        self.pipeline.to(device)

    def generate_images(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = None,
    ) -> List[torch.Tensor]:
        """
        Generate images based on text prompts.

        Args:
            prompts (List[str]): List of text prompts to guide the image generation.
            num_inference_steps (int): Number of inference steps for the pipeline.
            guidance_scale (float): Scale for classifier-free guidance.
            height (int): Height of the output images.
            width (int): Width of the output images.
            seed (int, optional): Seed for reproducibility.

        Returns:
            List[torch.Tensor]: List of generated images as tensors.
        """
        generator = torch.manual_seed(seed) if seed else None

        results = []
        for prompt in prompts:
            image = self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            results.append(image)

        return results

    def save_images(self, images: List[torch.Tensor], output_dir: str, file_names: List[str]) -> None:
        """
        Save generated images to the specified directory.

        Args:
            images (List[torch.Tensor]): List of image tensors to save.
            output_dir (str): Directory to save images.
            file_names (List[str]): Corresponding filenames for the images.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        for img, name in zip(images, file_names):
            img.save(os.path.join(output_dir, name))

    def apply_conditioning(self, prompts: List[str], conditioning_params: Dict[str, float]) -> List[torch.Tensor]:
        """
        Apply conditioning to generated images.

        Args:
            prompts (List[str]): List of text prompts.
            conditioning_params (Dict[str, float]): Parameters for conditioning (e.g., scale or steps).

        Returns:
            List[torch.Tensor]: List of conditioned images as tensors.
        """
        return self.generate_images(
            prompts,
            num_inference_steps=conditioning_params.get("num_inference_steps", 50),
            guidance_scale=conditioning_params.get("guidance_scale", 7.5),
        )

    def generate_images_with_embeddings(
        self, prompts: List[str], embeddings: List[torch.Tensor], guidance_scale: float = 7.5
    ) -> List[torch.Tensor]:
        """
        Generate images conditioned on both text prompts and additional embeddings.

        Args:
            prompts (List[str]): List of text prompts.
            embeddings (List[torch.Tensor]): List of embeddings for additional conditioning.
            guidance_scale (float): Guidance scale for the pipeline.

        Returns:
            List[torch.Tensor]: List of generated images as tensors.
        """
        results = []
        for prompt, embedding in zip(prompts, embeddings):
            image = self.pipeline(
                prompt,
                guidance_scale=guidance_scale,
                conditioning=embedding,
            ).images[0]
            results.append(image)
        return results

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess the image for use in the pipeline.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        return self.pipeline.preprocess(image)

    def load_embeddings(self, embedding_path: str) -> torch.Tensor:
        """
        Load embeddings from a specified path.

        Args:
            embedding_path (str): Path to the embedding file.

        Returns:
            torch.Tensor: Loaded embedding tensor.
        """
        return torch.load(embedding_path)

# Example usage:
# pipeline = StableDiffusionPipeline(model_name_or_path="CompVis/stable-diffusion-v1-4")
# images = pipeline.generate_images(["a sunny beach", "a mountain landscape"])
# pipeline.save_images(images, output_dir="outputs", file_names=["beach.png", "mountain.png"])
