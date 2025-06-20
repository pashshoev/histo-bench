import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm


class UNIEncoder(nn.Module):
    """
    Vision Transformer-based encoder using the UNI model from MahmoodLab.
    Requires a local path to the checkpoint (.bin file).

    Args:
        checkpoint_path (str): Path to the .bin checkpoint file.
        device (str or torch.device): Device to use for inference.
    """

    def __init__(self, checkpoint_path: str, device=None):
        super().__init__()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Important: UNI checkpoint requires these specific model args
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )

        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

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
