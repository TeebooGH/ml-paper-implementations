# Implementation of the Positional Encoding sub layer of the Transformer architecture

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds **non-learnable** positional encoding to token embeddings, as described in
    'Attention is All You Need', section 3.5, page 6.

    Args:
        d_model (int): Size of each embedding vector.
        max_seq_length (int): Maximum sequence length for which to precompute positional encodings.

    Returns:
        torch.Tensor: Embedded token representations with positional encoding added.
                      Shape: [batch_size, seq_length, d_model]

    Note:
        - PE encodes the **position** of each token in the sequence.
        - Positional encodings are precomputed and stored as a buffer (not a parameter).
        - During forward, positional encodings for the input sequence length are added to embeddings.
        - Broadcasting handles the batch dimension automatically.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()

        # Create a [max_seq_length, d_model] sized matrix to represent PE
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Using the exp(ln) trick to help with formula writing
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # Broadcasting: [max_seq_length, 1] * [d_model//2] -> [max_seq_length, d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but moves with model to GPU)
        # Shape: [max_seq_length, d_model] - NO unsqueeze!
        self.register_buffer("pe", pe)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, seq_length, d_model]
        Returns:
            [batch_size, seq_length, d_model]
        """
        # Get sequence length from input
        seq_length = embeddings.size(1)

        # Slice PE to match sequence length
        # self.pe[:seq_length] has shape [seq_length, d_model]
        # Broadcasting handles batch dimension automatically
        return embeddings + self.pe[:seq_length, :]
