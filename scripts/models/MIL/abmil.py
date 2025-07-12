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

    def forward(self, x):
        """
        Forward pass for the ABMIL model.

        Args:
            x (torch.Tensor): Input tensor representing a bag of instance embeddings (h_k).
                              Shape: (batch_size, num_instances, in_dim).
                              (N, K, M) where N=batch_size, K=num_instances, M=in_dim.

        Returns:
            torch.Tensor: Logits for the n_classes.
        """
        # x shape: (N, K, M) e.g., (1, 1000, 512)

        # Compute attention scores
        tanh_Vh = self.tanh_Vh(x) # ->  # (N, K, L)
        sigmoid_Uh = self.sigmoid_Uh(x) # -> (N, K, L)
        gated_attention_features = tanh_Vh * sigmoid_Uh # -> (N, K, L)
        attention_scores = self.attention_weight(gated_attention_features) # -> (N, K, 1)
        A = attention_scores.transpose(1, 2) # -> (N, 1, K)

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

    dummy_data = torch.rand((1, 1000, input_embedding_dim)) 
    input_size = dummy_data.shape
    out = model(dummy_data)
    
    print(f"\nInput data shape: {dummy_data.shape}\n")

    print(f"Output logits shape: {out.shape}\n")

    print(summary(model, input_size=input_size))