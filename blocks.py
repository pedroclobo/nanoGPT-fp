import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
CONTEXT_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LR = 3e-4
TRAIN_FRAC = 0.9
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)

# stoi/itos act as the tokenizer, mapping chars to ids.
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAIN_FRAC * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + CONTEXT_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + CONTEXT_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        self.head_size = head_size
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        att = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(att, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: nn.Sequential of
        #   Linear(N_EMBD, 4 * N_EMBD) -> ReLU -> Linear(4 * N_EMBD, N_EMBD) -> Dropout(DROPOUT)
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        # TODO: run x through the network
        return self.ff(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)   # before attention
        self.ln2 = nn.LayerNorm(N_EMBD)   # before feed-forward

    def forward(self, x):
        # TODO: pre-norm + residual, twice:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(CONTEXT_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, num_tokens):
        for _ in range(num_tokens):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_char], dim=1)
        return idx


def main():
    model = GPT().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"iter {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        xs, ys = get_batch("train")
        _, loss = model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(start, 500)[0].tolist()))


if __name__ == "__main__":
    main()
