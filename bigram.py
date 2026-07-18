import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
CONTEXT_SIZE = 8
MAX_ITERS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LR = 1e-2
TRAIN_FRAC = 0.8

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

# Samples `BATCH_SIZE` random sequences of length `CONTEXT_SIZE`
# from either the training or validation data.
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + CONTEXT_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + CONTEXT_SIZE + 1] for i in ix])
    return x, y


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


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        if targets is None:
            return self.embed(idx), None

        logits = self.embed(idx)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, num_tokens):
        for _ in range(num_tokens):
            logits, _ = self.forward(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(probs, 1) # sample one char from that distribution
            idx = torch.cat([idx, next_char], dim=1)
        return idx


def main():
    model = BigramLM(vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"iter {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        xs, ys = get_batch("train")
        _, loss = model.forward(xs, ys)
        if loss is None: continue
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(start, 500)[0].tolist()))


if __name__ == "__main__":
    main()
