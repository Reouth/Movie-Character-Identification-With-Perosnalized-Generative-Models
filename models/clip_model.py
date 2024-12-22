import clip
import torch
from src import data_upload

class CLIPPipeline:
    """
    A class for handling CLIP-based image embeddings and classification.

    Attributes:
        model_CLIP (torch.nn.Module): Pretrained CLIP model.
        preprocess_CLIP (callable): Preprocessing pipeline for input images.
        device (str): Device to run the computations on (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, device, model_name):
        """
        Initialize the CLIP pipeline with the specified model and device.

        Args:
            device (str): Device to run the CLIP model on.
            model_name (str): Name of the pretrained CLIP model to load.
        """
        self.model_CLIP, self.preprocess_CLIP = clip.load(model_name, device)
        self.device = device

    def image_to_embedding(self, image):
        """
        Convert a single image into its corresponding embedding using CLIP.

        Args:
            image (PIL.Image): Input image to convert into embedding.

        Returns:
            torch.Tensor: Normalized image embedding tensor.
        """
        CLIP_image = self.preprocess_CLIP(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model_CLIP.encode_image(CLIP_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize embeddings
        return image_features.to(self.device)

    def images_to_embeddings(self, path):
        """
        Generate embeddings for all images in a specified directory.

        Args:
            path (str): Path to the directory containing images.

        Returns:
            dict: A dictionary mapping image names to their embeddings.
        """
        clip_embeddings = {}
        images = data_upload.upload_images(path)  # Load images using helper function

        for name, img, _ in images:
            clip_embeddings[name] = self.image_to_embedding(img)

        return clip_embeddings

    def conditioned_classifier(self, test_image, clip_embeddings):
        """
        Perform classification on a test image by comparing it with stored embeddings.

        Args:
            test_image (PIL.Image): Input test image for classification.
            clip_embeddings (dict): Dictionary of embeddings for comparison.

        Returns:
            list: Sorted list of similarity scores and associated labels.
        """
        # Extract embedding names and stack them into a single tensor
        embed_names = list(clip_embeddings.keys())
        CLIP_ID_embeds = torch.cat([clip_embeddings[name] for name in clip_embeddings]).to(self.device)
        CLIP_ID_embeds /= CLIP_ID_embeds.norm(dim=-1, keepdim=True)  # Normalize embeddings

        # Generate embedding for the test image
        test_embeddings = self.image_to_embedding(test_image)

        # Calculate similarity scores
        similarity = test_embeddings @ CLIP_ID_embeds.T
        classification = (100.0 * similarity).softmax(dim=-1)

        unsorted_sim = {}  # Unsorted similarity scores

        # Map scores to their corresponding labels
        for cls, sim, name in zip(classification[0], similarity[0], embed_names):
            unsorted_sim[name] = sim.item()

        # Sort the similarity scores in descending order
        sorted_sim = sorted(unsorted_sim.items(), key=lambda kv: kv[1], reverse=True)

        return sorted_sim
