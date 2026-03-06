import torch
import torch.nn as nn

from transformer_torch.model.embeddings import Embeddings
from transformer_torch.model.attention import (
    SelfAttention,
    MaskedSelfAttention,
    MultiHeadAttention,
)
from transformer_torch.model.transformer import (
    Encoder,
    Decoder,
    Transformer,
    make_pad_mask,
)

print("\n--- Testing Full Transformer Model ---")

# Model Parameters (Matching Base Paper, Section 3 Table 3)
src_vocab_size = 37000
tgt_vocab_size = 37000
d_model = 512
n_layers = 6  # N=6
n_heads = 8
dropout = 0.1
d_ff = 2048
max_seq_len = 512
pad_id = 0

# --- Build the model ---

src_emb = Embeddings(src_vocab_size, d_model)
tgt_emb = Embeddings(tgt_vocab_size, d_model)

# Use factories so each layer gets its own independent copy of attention modules
self_attn_factory = lambda: SelfAttention(d_model, n_heads, dropout)
masked_self_attn_factory = lambda: MaskedSelfAttention(d_model, n_heads, dropout)
cross_attn_factory = lambda: MultiHeadAttention(d_model, n_heads, dropout)

encoder = Encoder(
    src_emb, d_model, max_seq_len, n_layers, self_attn_factory, d_ff, dropout
)
decoder = Decoder(
    tgt_emb,
    d_model,
    max_seq_len,
    n_layers,
    masked_self_attn_factory,
    cross_attn_factory,
    d_ff,
    dropout,
)

model = Transformer(encoder, decoder, d_model, tgt_vocab_size)

# Xavier uniform initialization (common practice for Transformers)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

print(
    f"Transformer created with {sum(p.numel() for p in model.parameters()):,} parameters."
)

# --- Dummy Data ---
B = 2
src_seq_len = 10
tgt_seq_len = 12

# Token IDs: start from 1 to avoid colliding with pad_id=0
src = torch.randint(1, src_vocab_size, (B, src_seq_len))
tgt = torch.randint(1, tgt_vocab_size, (B, tgt_seq_len))

# Padding masks: True = keep, False = masked out
# make_pad_mask returns [B, 1, 1, T], broadcastable over heads and query positions
src_mask = make_pad_mask(src, pad_id)  # [B, 1, 1, src_seq_len]
tgt_pad_mask = make_pad_mask(tgt, pad_id)  # [B, 1, 1, tgt_seq_len]

print(f"\nSource shape:    {src.shape}")
print(f"Target shape:    {tgt.shape}")
print(f"src_mask shape:  {src_mask.shape}")
print(f"tgt_mask shape:  {tgt_pad_mask.shape}")

# --- Forward pass ---
model.eval()
with torch.no_grad():
    logits = model(src, tgt, src_mask=src_mask, tgt_pad_mask=tgt_pad_mask)

# Expected: [B, tgt_seq_len, tgt_vocab_size]
print(f"\nLogits shape: {logits.shape}")
assert logits.shape == (B, tgt_seq_len, tgt_vocab_size), (
    f"Expected {(B, tgt_seq_len, tgt_vocab_size)}, got {logits.shape}"
)

print("\nFull Transformer test passed (shape check).")
