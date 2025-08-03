import torch
import torch.nn as nn
from torchinfo import summary


class ABMIL(nn.Module):
    def __init__(self, n_classes=2, in_dim=512, hidden_dim=256, dropout=0.3, is_norm=True, *args, **kwargs):
        """
        Initializes the Attention-based Multiple Instance Learning (ABMIL) model
        with a Gated Attention mechanism, identical to the original paper.

        Args:
            n_classes (int): Number of output classes for the classifier. Default is 2.
            in_dim (int): Dimension of the input instance embeddings (M in the paper's notation).
                          This is the dimension of h_k. Default is 512.
            hidden_dim (int): Dimension for the intermediate attention layers (L in the paper's notation).
                              This is the output dimension of Vh_k^T and Uh_k^T. Default is 256.
            dropout (float): Dropout rate for the final classifier. Default is 0.3.
            is_norm (bool): Whether to apply softmax normalization to attention weights. Default is True.
        """
        super().__init__()

        # These directly implement V and U from the paper.
        self.tanh_Vh = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), # V * h_k^T
            nn.Tanh()
        )
        self.sigmoid_Uh = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), # U * h_k^T
            nn.Sigmoid()
        )
        
        self.attention_weight = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(in_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.is_norm = is_norm

    def forward(self, x, attention_mask=None):
        """
        Forward pass for the ABMIL model.

        Args:
            x (torch.Tensor): Input tensor representing a bag of instance embeddings (h_k).
                              Shape: (batch_size, num_instances, in_dim).
                              (N, K, M) where N=batch_size, K=num_instances, M=in_dim.
            attention_mask (torch.Tensor, optional): Boolean mask indicating which instances are real (True) vs padded (False).
                                                   Shape: (batch_size, num_instances). Default is None.

        Returns:
            torch.Tensor: Logits for the n_classes.
        """
        # x shape: (N, K, M) e.g., (batch_size, max_patches, 512)

        # Compute attention scores
        tanh_Vh = self.tanh_Vh(x) # ->  # (N, K, L)
        sigmoid_Uh = self.sigmoid_Uh(x) # -> (N, K, L)
        gated_attention_features = tanh_Vh * sigmoid_Uh # -> (N, K, L)
        attention_scores = self.attention_weight(gated_attention_features) # -> (N, K, 1)
        A = attention_scores.transpose(1, 2) # -> (N, 1, K)

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(1)  # (N, K) -> (N, 1, K)
            # Set attention scores for padded positions to large negative values
            A = A.masked_fill(~attention_mask_expanded, -1e9)

        # Apply softmax across the instances 
        if self.is_norm:
            A = torch.softmax(A, dim=-1) # -> (N, 1, K)

        # Perform weighted sum of instance embeddings.
        x = torch.bmm(A, x).squeeze(dim=1) # -> (N, M)
        # Classification layer
        x = self.dropout(x) # -> (N, M)
        logits = self.classifier(x) # -> (N, n_classes)

        return logits

# Example Usage
if __name__ == '__main__':
    input_embedding_dim = 512 
    attention_hidden_dim = 256
    model = ABMIL(n_classes=3, in_dim=input_embedding_dim, hidden_dim=attention_hidden_dim)

    # Test with batch size > 1 and variable lengths
    batch_size = 4
    max_patches = 1000
    dummy_data = torch.rand((batch_size, max_patches, input_embedding_dim)) 
    attention_mask = torch.ones(batch_size, max_patches, dtype=torch.bool)
    # Simulate variable lengths
    attention_mask[0, 800:] = False  # First sample has 800 patches
    attention_mask[1, 600:] = False  # Second sample has 600 patches
    attention_mask[2, 900:] = False  # Third sample has 900 patches
    attention_mask[3, 700:] = False  # Fourth sample has 700 patches
    
    input_size = dummy_data.shape
    out = model(dummy_data, attention_mask)
    
    print(f"\nInput data shape: {dummy_data.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Output logits shape: {out.shape}\n")

    print(summary(model, input_size=input_size))