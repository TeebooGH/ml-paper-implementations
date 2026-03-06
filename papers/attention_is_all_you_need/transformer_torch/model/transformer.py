import torch
import torch.nn as nn

from .embeddings import Embeddings
from .normalization import ResidualConnection
from .positioning import PositionalEncoding
from .feed_forward import PositionwiseFeedForward
from .attention import SelfAttention, MaskedSelfAttention, MultiHeadAttention


def make_pad_mask(tokens: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    tokens: [B, T]
    returns: [B, 1, 1, T] avec True = keep, False = mask
    """
    return (tokens != pad_id).unsqueeze(1).unsqueeze(2)


# -------------------------
# Wrappers pour ResidualConnection
# -------------------------


class _SelfAttnSublayer(nn.Module):
    def __init__(self, self_attn: nn.Module):
        super().__init__()
        self.self_attn = self_attn
        self.mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.self_attn(x, mask=self.mask)  # SelfAttention renvoie (out, attn)
        return out


class _MaskedSelfAttnSublayer(nn.Module):
    def __init__(self, masked_self_attn: nn.Module):
        super().__init__()
        self.masked_self_attn = masked_self_attn
        self.pad_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.masked_self_attn(x, pad_mask=self.pad_mask)  # (out, attn)
        return out


class _CrossAttnSublayer(nn.Module):
    def __init__(self, cross_attn: nn.Module):
        super().__init__()
        self.cross_attn = cross_attn
        self.memory: torch.Tensor | None = None
        self.src_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Q = x (decoder), K,V = memory (encoder)
        out, _ = self.cross_attn(q=x, k=self.memory, v=self.memory, mask=self.src_mask)
        return out


# -------------------------
# Encoder Layer
# -------------------------


class EncoderLayer(nn.Module):
    """
    x -> SelfAttn -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(self, d_model: int, self_attn: nn.Module, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn_sub = _SelfAttnSublayer(self_attn)
        self.res1 = ResidualConnection(d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.res2 = ResidualConnection(d_model, dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.self_attn_sub.mask = src_mask
        x = self.res1(x, self.self_attn_sub)
        x = self.res2(x, self.ffn)
        return x


# -------------------------
# Decoder Layer
# -------------------------


class DecoderLayer(nn.Module):
    """
    y -> MaskedSelfAttn -> Add&Norm -> CrossAttn -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(
        self,
        d_model: int,
        masked_self_attn: nn.Module,
        cross_attn: nn.Module,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.masked_self_sub = _MaskedSelfAttnSublayer(masked_self_attn)
        self.res1 = ResidualConnection(d_model, dropout)

        self.cross_sub = _CrossAttnSublayer(cross_attn)
        self.res2 = ResidualConnection(d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(
        self,
        y: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1) masked self-attention (causal est géré dans MaskedSelfAttention)
        self.masked_self_sub.pad_mask = tgt_pad_mask
        y = self.res1(y, self.masked_self_sub)

        # 2) cross-attention vers l'encodeur
        self.cross_sub.memory = memory
        self.cross_sub.src_mask = src_mask
        y = self.res2(y, self.cross_sub)

        # 3) feed-forward
        y = self.res3(y, self.ffn)
        return y


# -------------------------
# Encoder (stack N layers)
# -------------------------


class Encoder(nn.Module):
    def __init__(
        self,
        embeddings: nn.Module,
        d_model: int,
        max_len: int,
        n_layers: int,
        self_attn_factory,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.emb = embeddings
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, self_attn_factory(), d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, src_tokens: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.dropout(self.pos(self.emb(src_tokens)))  # Emb + PE + dropout
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        return x  # memory


# -------------------------
# Decoder (stack N layers)
# -------------------------


class Decoder(nn.Module):
    def __init__(
        self,
        embeddings: nn.Module,
        d_model: int,
        max_len: int,
        n_layers: int,
        masked_self_attn_factory,
        cross_attn_factory,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.emb = embeddings
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    masked_self_attn_factory(),
                    cross_attn_factory(),
                    d_ff,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.dropout(self.pos(self.emb(tgt_tokens)))  # Emb + PE + dropout
        for layer in self.layers:
            y = layer(y, memory, tgt_pad_mask=tgt_pad_mask, src_mask=src_mask)
        return y


# -------------------------
# (Optionnel) Transformer complet
# -------------------------


class Transformer(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, d_model: int, tgt_vocab_size: int
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.proj = nn.Linear(d_model, tgt_vocab_size)  # logits

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_pad_mask=None):
        memory = self.encoder(src_tokens, src_mask=src_mask)
        dec_out = self.decoder(
            tgt_tokens, memory, tgt_pad_mask=tgt_pad_mask, src_mask=src_mask
        )
        return self.proj(dec_out)  # [B, Tt, vocab]


if __name__ == "__main__":
    """ exemple d'utilisation """

    # params
    src_vocab = 37000
    tgt_vocab = 37000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    dropout = 0.1
    max_len = 512
    pad_id = 0

    src_emb = Embeddings(src_vocab, d_model)
    tgt_emb = Embeddings(tgt_vocab, d_model)

    def self_attn_factory():
        return SelfAttention(d_model, n_heads, dropout)

    def masked_self_attn_factory():
        return MaskedSelfAttention(d_model, n_heads, dropout)

    def cross_attn_factory():
        return MultiHeadAttention(d_model, n_heads, dropout)

    encoder = Encoder(
        src_emb, d_model, max_len, n_layers, self_attn_factory, d_ff, dropout
    )
    decoder = Decoder(
        tgt_emb,
        d_model,
        max_len,
        n_layers,
        masked_self_attn_factory,
        cross_attn_factory,
        d_ff,
        dropout,
    )

    model = Transformer(encoder, decoder, d_model, tgt_vocab)

    # dummy data + masks
    B, Ts, Tt = 2, 10, 12
    src = torch.randint(0, src_vocab, (B, Ts))
    tgt = torch.randint(0, tgt_vocab, (B, Tt))

    src_mask = make_pad_mask(src, pad_id)  # [B,1,1,Ts]
    tgt_pad_mask = make_pad_mask(tgt, pad_id)  # [B,1,1,Tt]

    logits = model(src, tgt, src_mask=src_mask, tgt_pad_mask=tgt_pad_mask)
    print(logits.shape)  # [B, Tt, tgt_vocab]
