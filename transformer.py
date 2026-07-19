from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import BaseConfig, run


@dataclass
class Config(BaseConfig):
    name: str = "transformer"
    context_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    max_iters: int = 5000
    patience: int = 5


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.context_size, cfg.context_size)))
        self.head_size = head_size
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(att, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)   # before attention
        self.ln2 = nn.LayerNorm(cfg.n_embd)   # before feed-forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.context_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok + pos))
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    run(GPT, Config())
