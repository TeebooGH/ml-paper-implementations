import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    """
    Causal mask (decoder): autorise l'attention uniquement vers le passé.
    Shape: [1, size, size] (broadcastable to batch and heads)
    True = allowed, False = masked
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0)  # [1, T, T]


def make_padding_mask(token_ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Padding mask from token indices.
    token_ids: [B, T]
    returns: [B, 1, 1, T] where True = keep, False = mask
    """
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
    (paper section 3.2.1) 
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,  # [B, H, Tq, Dk]
        K: torch.Tensor,  # [B, H, Tk, Dk]
        V: torch.Tensor,  # [B, H, Tk, Dv]
        mask: torch.Tensor | None = None,  # broadcastable to [B, H, Tq, Tk]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,Tq,Tk]

        if mask is not None:
            # mask: True = keep, False = mask out
            if mask.dtype != torch.bool:
                mask = mask != 0
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # [B,H,Tq,Tk]
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # [B,H,Tq,Dv]
        return out, attn


class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_i) W_O
    head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
    (paper section 3.2.2) 
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # d_k = d_v = d_model / h in the paper (base config)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] -> [B, H, T, d_head]
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, d_head] -> [B, T, d_model]
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return x

    def forward(
        self,
        q: torch.Tensor,  # [B, Tq, d_model]
        k: torch.Tensor,  # [B, Tk, d_model]
        v: torch.Tensor,  # [B, Tk, d_model]
        mask: torch.Tensor | None = None,  # broadcastable to [B, 1 or H, Tq, Tk]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q = self._split_heads(self.W_q(q))  # [B,H,Tq,Dh]
        K = self._split_heads(self.W_k(k))  # [B,H,Tk,Dh]
        V = self._split_heads(self.W_v(v))  # [B,H,Tk,Dh]

        # Make mask broadcastable to [B,H,Tq,Tk]
        if mask is not None:
            if mask.dim() == 2:  # [B, Tk] (rare)
                mask = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,Tk]
            elif mask.dim() == 3:  # [B, Tq, Tk]
                mask = mask.unsqueeze(1)  # [B,1,Tq,Tk]
            # if dim==4 already OK

        heads_out, attn_weights = self.attn(Q, K, V, mask=mask)  # [B,H,Tq,Dh], [B,H,Tq,Tk]
        out = self._combine_heads(heads_out)  # [B,Tq,d_model]
        out = self.W_o(out)
        out = self.out_dropout(out)
        return out, attn_weights


class SelfAttention(nn.Module):
    """
    Self-attention (encoder): Q=K=V=x
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.mha(x, x, x, mask=mask)


class MaskedSelfAttention(nn.Module):
    """
    Masked self-attention (decoder): causal mask + optional padding mask
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        B, T, _ = x.shape
        causal = subsequent_mask(T, device=x.device)  # [1,T,T]
        causal = causal.unsqueeze(1)  # [1,1,T,T]

        mask = causal
        if pad_mask is not None:
            # pad_mask expected [B,1,1,T]
            mask = mask & pad_mask  # broadcast to [B,1,T,T]

        return self.mha(x, x, x, mask=mask)


# Quick test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d_model, h = 2, 5, 512, 8
    x = torch.randn(B, T, d_model)

    sa = SelfAttention(d_model=d_model, n_heads=h, dropout=0.1)
    y, attn = sa(x)  # y: [B,T,d_model], attn: [B,H,T,T]
    print("SelfAttention out:", y.shape, "attn:", attn.shape)

    msa = MaskedSelfAttention(d_model=d_model, n_heads=h, dropout=0.1)
    y2, attn2 = msa(x)
    print("MaskedSelfAttention out:", y2.shape, "attn:", attn2.shape)
