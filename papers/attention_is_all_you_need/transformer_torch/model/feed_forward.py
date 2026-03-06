import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Args:
            d_model (int): Input and output dimension.
            d_ff (int): Inner dimension of the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # First linear layer (d_model -> d_ff)
        self.linear_1 = nn.Linear(d_model, d_ff)
        # Activation function (ReLU as per original paper)
        self.activation = nn.ReLU()
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Second linear layer (d_ff -> d_model)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the FFN.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, d_model)
        """
        # Apply layers sequentially: Linear1 -> Activation -> Dropout -> Linear2
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)  # Dropout usually applied after activation
        x = self.linear_2(x)
        return x
