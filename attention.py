from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import BaseConfig, run


@dataclass
class Config(BaseConfig):
    name: str = "attention"
    context_size: int = 32
    n_embd: int = 64
    n_head: int = 4
    lr: float = 1e-3
    max_iters: int = 5000


class Head(nn.Module):
    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.context_size, cfg.context_size)))
        self.head_size = head_size

    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(att, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)


class AttentionLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.context_size, cfg.n_embd)
        self.sa = MultiHeadAttention(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.sa(tok + pos)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    run(AttentionLM, Config())
