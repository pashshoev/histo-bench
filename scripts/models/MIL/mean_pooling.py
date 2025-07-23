import torch
import torch.nn as nn
from loguru import logger

class MeanPooling(nn.Module):
    def __init__(self, n_classes: int, in_dim: int, *args, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x, attention_mask=None):
        """
        Forward pass for mean pooling MIL model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_instances, in_dim)
            attention_mask (torch.Tensor, optional): Boolean mask indicating which instances are real (True) vs padded (False).
                                                   Shape: (batch_size, num_instances). Default is None.
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, n_classes)
        """
        if attention_mask is not None:
            # Use masked mean: sum only real patches and divide by count of real patches
            masked_x = x * attention_mask.unsqueeze(-1)  # (batch_size, num_instances, in_dim)
            sum_x = torch.sum(masked_x, dim=1)  # (batch_size, in_dim)
            count = torch.sum(attention_mask, dim=1, keepdim=True)  # (batch_size, 1)
            # Avoid division by zero
            count = torch.clamp(count, min=1)
            x = sum_x / count  # (batch_size, in_dim)
        else:
            # Original behavior: simple mean across all instances
            x = torch.mean(x, dim=1)
        
        return self.fc(x)