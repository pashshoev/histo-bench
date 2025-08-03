import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm

from huggingface_hub import hf_hub_download
from scripts.models.vision.base import BaseEncoder, ModelName


class UNIEncoder(BaseEncoder, nn.Module):
    """
    Vision Transformer-based encoder using the UNI model from MahmoodLab (via Hugging Face Hub).
    Downloads the checkpoint automatically using the provided Hugging Face token.

    Args:
        hf_token (str): Hugging Face authentication token.
        device (str or torch.device): Device to use for inference.
    """

    def __init__(
        self,
        hf_token: str,
        device=None,
        hf_repo: str = "MahmoodLab/UNI",
        hf_filename: str = "pytorch_model.bin",
    ):
        super().__init__()

        checkpoint_path = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            token=hf_token,
            local_dir="./hf_uni_ckpt",
            force_download=False,
        )

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = ModelName.UNI.value

        # Important: UNI checkpoint requires these specific model args
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"), strict=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def extract_embeddings(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for a batch of input tensors.

        Args:
            batch_tensors (torch.Tensor): Input batch, shape (B, C, H, W)

        Returns:
            torch.Tensor: Output embeddings, shape (B, D)
        """
        batch_tensors = batch_tensors.to(self.device)
        with torch.no_grad():
            return self.model(batch_tensors)

    def preprocess(self, pil_image):
        """
        Preprocess a PIL image to a normalized tensor.

        Args:
            pil_image (PIL.Image): Input image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.transform(pil_image)

    def get_summary(self):
        """
        Return model summary including number of parameters and input/output shapes.

        Returns:
            dict: Contains input shape, output shape, and total parameter count.
        """
        input_size = (3, 224, 224)
        dummy_input = torch.randn(1, *input_size).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)

        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "input_shape": (1, *input_size),
            "output_shape": tuple(output.shape),
            "total_parameters": total_params,
        }
