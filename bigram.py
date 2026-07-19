from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from common import BaseConfig, run


@dataclass
class Config(BaseConfig):
    name: str = "bigram"
    context_size: int = 8
    batch_size: int = 32
    lr: float = 1e-2
    max_iters: int = 10000
    train_frac: float = 0.8


class BigramLM(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embed(idx)
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


if __name__ == "__main__":
    run(BigramLM, Config())
