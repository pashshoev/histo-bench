import torch
import torch.nn as nn

from scripts.models.vision.base import BaseEncoder
from conch.open_clip_custom import create_model_from_pretrained


class CONCHEncoder(BaseEncoder, nn.Module):
    """
    Wrapper for CONCH ViT-B-16 encoder (from MahmoodLab via Hugging Face).

    Args:
        hf_token (str): Hugging Face authentication token (must be provided if required by repo)
        device (str or torch.device): Device to use for inference
        model_id (str): Model ID string (default: 'conch_ViT-B-16')
        repo_path (str): Hugging Face model reference (default: 'hf_hub:MahmoodLab/conch')
    """

    def __init__(self,
                 hf_token: str,
                 device,
                 model_id: str = "conch_ViT-B-16",
                 repo_path: str = "hf_hub:MahmoodLab/conch"):
        super().__init__()
        self.device = device

        model, preprocess = create_model_from_pretrained(
            model_id,
            repo_path,
            hf_auth_token=hf_token
        )

        self.model = model.to(self.device)
        self.model.eval()
        self.transform = preprocess  # typically a torchvision transform pipeline


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model.encode_image(x, proj_contrast=False, normalize=False)


    def extract_embeddings(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from a batch of input image tensors.

        Args:
            batch_tensors (torch.Tensor): Input batch (B, 3, H, W), preprocessed

        Returns:
            torch.Tensor: Embedding batch (B, D)
        """
        batch_tensors = batch_tensors.to(self.device)
        with torch.no_grad():
            return self.model.encode_image(batch_tensors, proj_contrast=False, normalize=False)


    def preprocess(self, pil_image):
        """
        Apply preprocessing to a PIL image.

        Args:
            pil_image (PIL.Image): Input image

        Returns:
            torch.Tensor: Preprocessed tensor
        """
        return self.transform(pil_image)


    def get_summary(self):
        """
        Return model summary including number of parameters and input/output shapes.

        Returns:
            dict: Contains input shape, output shape, and total parameter count.
        """
        input_size = (3, *self.model.visual.image_size)
        dummy_input = torch.randn(1, *input_size).to(self.device)
        with torch.no_grad():
            output = self.model.encode_image(dummy_input, proj_contrast=False, normalize=False)

        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "input_shape": (1, *input_size),
            "output_shape": tuple(output.shape),
            "total_parameters": total_params
        }
