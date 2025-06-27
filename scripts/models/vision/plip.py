import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from scripts.models.vision.base import BaseEncoder


class PLIPTransform:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)


class PLIPEncoder(BaseEncoder, nn.Module):
    """
    Encoder using the PLIP model (CLIP fine-tuned for pathology) from Hugging Face.

    Args:
        model_name (str): Hugging Face model ID (default: 'vinid/plip')
        device (str or torch.device): Device for inference
    """

    def __init__(self, model_name: str = "vinid/plip", device=None):
        super().__init__()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.transform = PLIPTransform(self.processor)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass is not used for raw tensors in CLIP models."""
        raise NotImplementedError("Use extract_embeddings() or extract_image_features() instead.")

    def extract_embeddings(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """
        Extract image features from a batch of preprocessed tensors.

        Args:
            batch_tensors (torch.Tensor): Preprocessed image tensors

        Returns:
            torch.Tensor: Image embeddings
        """
        with torch.no_grad():
            return self.model.get_image_features(pixel_values=batch_tensors.to(self.device))

    def preprocess(self, image):
        """
        Preprocess a single image using CLIP processor.

        Args:
            image (PIL.Image): Input image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

    def get_summary(self):
        """
        Return model summary including number of parameters.

        Returns:
            dict: Contains model name and total parameter count.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "input_shape": (1, 3, 224, 224),
            "output_shape": (1, 512),
            "total_parameters": total_params
        }
