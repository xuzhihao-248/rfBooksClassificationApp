import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn, V)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 128,
                 max_len: int = 512, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.self_attention = SelfAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = self._padding_mask(x, lengths)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.self_attention(x, mask)
        x = self.dropout(x)
        pooled = self._masked_mean_pool(x, mask)
        return self.classifier(pooled)

    def _padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def _masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float().unsqueeze(-1)
        x = x * valid
        summed = x.sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1)
        return summed / counts
