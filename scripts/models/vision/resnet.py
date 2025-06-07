import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


ResNetTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])


class ResNetEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet50', device = None):
        super().__init__()
        if model_name not in models.__dict__:
            raise ValueError(f"Model '{model_name}' not found in torchvision.models.")

        self.model = models.__dict__[model_name](weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_embeddings(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """
        Extracts embeddings from a batch of pre-transformed image tensors.

        Args:
            batch_tensors (torch.Tensor): A batch of image tensors (C, H, W).
                                        Expected shape: (B, C, H, W) after transformations.

        Returns:
            torch.Tensor: Embeddings for the input batch. Shape: (B, embedding_dim).
        """
        # Ensure the input is on the correct device
        batch_tensors = batch_tensors.to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch_tensors) # Shape: (B, 2048, 1, 1) for ResNet

        return embeddings.view(embeddings.size(0), -1) # Shape: (B, 2048)