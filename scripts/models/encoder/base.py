from abc import ABC, abstractmethod
import torch
from enum import Enum

class BaseEncoder(ABC):
    """
    Abstract base class for all image encoders.
    Defines a consistent interface for preprocessing, forward pass, and summary.
    """

    @abstractmethod
    def preprocess(self, image):
        """Preprocess input (e.g., PIL image or tensor) into model-ready tensor."""
        pass

    @abstractmethod
    def extract_embeddings(self, batch_tensors: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a batch of input tensors."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_summary(self) -> dict:
        """Return summary: input/output shape, number of parameters, etc."""
        pass


class ModelName(Enum):
    RESNET50 = "ResNet50"
    UNI = "UNI"
    CONCH = "CONCH"
    PLIP = "PLIP"