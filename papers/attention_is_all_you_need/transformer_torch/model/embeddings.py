import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    """
    Converts token indices to embedding vectors of dimension d_model.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        d_model (int): Size of each embedding vector (Words live in a d_model = 512-dimensional vector space!).

    Returns:
        torch.Tensor: Embedded token representations of shape [batch_size, seq_length, d_model].

    Notes:
        - Uses learned embeddings as described in the Transformer paper.
        - Embeddings are scaled by sqrt(d_model) for stable training, matching the paper's recommendation.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_indices (torch.Tensor): Tensor of token indices, shape [batch_size, seq_length].

        Returns:
            torch.Tensor: Embedded token representations, shape [batch_size, seq_length, d_model].
        """
        return self.embedding(token_indices) * self.scale


# Example usage for testing
if __name__ == "__main__":
    vocab_size = 37000
    d_model = 512
    batch_size = 2
    seq_length = 5

    embedding_layer = Embeddings(vocab_size, d_model)
    dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    embedded = embedding_layer(dummy_tokens)

    print(f"dummy_tokens shape: {dummy_tokens.shape}")  # (2, 5)
    print(
        f"embedded shape: {embedded.shape}"
    )  # (2, 5, 512), which means PyTorch's broadcasting looked every token index up in the Embedding table!
