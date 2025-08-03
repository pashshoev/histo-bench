import torch
import torch.nn as nn
from torchvision import models, transforms
from scripts.models.vision.base import BaseEncoder, ModelName


class ResNetEncoder(BaseEncoder, nn.Module):
    """
    ResNet-based encoder using torchvision pretrained ResNet50 model.
    Strips classification head and provides built-in preprocessing.

    Args:
        device (str or torch.device): Device to load the model on.
    """

    def __init__(self, device=None):
        super().__init__()

        self.model = models.resnet50(weights="IMAGENET1K_V1")
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))  # Remove FC layer
        self.model_name = ModelName.RESNET50.value
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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
            torch.Tensor: Output embeddings, shape (B, 2048)
        """
        batch_tensors = batch_tensors.to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch_tensors)  # Shape: (B, 2048, 1, 1)
        return embeddings.view(embeddings.size(0), -1)

    def preprocess(self, pil_image):
        """
        Apply preprocessing to a PIL image.

        Args:
            pil_image (PIL.Image): Input image

        Returns:
            torch.Tensor: Transformed tensor ready for model input
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
            "total_parameters": total_params
        }
