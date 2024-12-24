
"""
This script provides functions for running diffusion-based generation and identification models.
"""

import torch
from typing import Optional
import os
import gc
from src import DataUpload
from src import HelperFunctions




def multi_image_generator(input_files, output_folder, imagic_pretrained_path, imagic_pipe, sd_model_name, clip_model_name, device,
                          seed_range, alpha_range, guidance_scale_range, height: Optional[int] = 512,
                          width: Optional[int] = 512, num_inference_steps: Optional[int] = 50):

    for embeds in input_files:
        parameters, pipe_name = DataUpload.get_pretrained_params(imagic_pipe, imagic_pretrained_path, sd_model_name, clip_model_name, embeds, device)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for seed in seed_range:
            for alpha in alpha_range:
                for guidance_scale in guidance_scale_range:

                    output_dir = os.path.join(output_folder, pipe_name, f"seed_{seed}")
                    os.makedirs(output_dir, exist_ok=True)

                    image_path,_  = HelperFunctions.generate_image_path(output_dir, embeds, alpha, guidance_scale)
                    image_exists = HelperFunctions.image_check(image_path)
                    if image_exists:
                        continue

                    image_generator(
                        os.path.dirname(image_path),
                        parameters,
                        os.path.basename(image_path),
                        alpha,
                        seed=seed,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    )


def image_generator(output_folder, parameters, image_name, alpha=0,
                     seed: int = 0, height: Optional[int] = 512, width: Optional[int] = 512,
                     num_inference_steps: Optional[int] = 50, guidance_scale: float = 7.5):
    """
    Generate a single image using the specified parameters.

    Args:
        output_folder (str): Directory to save the image.
        parameters (tuple): Parameters from pipeline (pipline weights, target embedding, optimized embedding).
        image_name (str): Name of the output image.
        alpha (float): Alpha blending parameter for embeddings.
        seed (int): Random seed for reproducibility.
        height (int): Height of the generated image.
        width (int): Width of the generated image.
        num_inference_steps (int): Number of diffusion steps.
        guidance_scale (float): Guidance scale for classifier-free guidance.
    """

    pipeline, target_embeddings, optimized_embeddings = parameters
    embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings

    with torch.autocast("cuda"), torch.inference_mode():
        images = pipeline.generateImage(
            cond_embeddings=embeddings,
            seed=seed,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    images[0].save(os.path.join(output_folder, image_name))



if __name__ == "__main__":
    print("This module contains Stable Diffusion and Imagic utilities for image generation and classification.")
