"""Shared harness for the char-level LMs.

Each model file defines a Config(BaseConfig) and an nn.Module, then calls
run(Model, Config()).
"""
import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields, replace

import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class BaseConfig:
    # Training fields shared by every model. On resume these come from the current
    # invocation, while arch fields (per model) come from the checkpoint.
    name: str = "model"
    batch_size: int = 32
    lr: float = 3e-4
    max_iters: int = 5000
    train_frac: float = 0.9
    eval_interval: int = 500
    eval_iters: int = 200
    patience: int = 0  # early stop after this many evals without val improvement (0 disables)


def _base_names():
    return {f.name for f in fields(BaseConfig)}


def _arch_fields(config):
    # Model-specific, shape-affecting fields: everything not in BaseConfig.
    return [f.name for f in fields(config) if f.name not in _base_names()]


def load_data(config):
    text = open("input.txt", encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=DEVICE)
    n = int(config.train_frac * len(data))
    return data[:n], data[n:], stoi, itos


def get_batch(split, train_data, val_data, config):
    d = train_data if split == "train" else val_data
    T = config.context_size
    # Draw offsets on the CPU (keeps RNG resumable), gather on-device in one shot.
    ix = torch.randint(len(d) - T, (config.batch_size, 1))
    pos = (ix + torch.arange(T + 1)).to(DEVICE)
    chunk = d[pos]
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            xb, yb = get_batch(split, train_data, val_data, config)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def generate(model, config, num_tokens):
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    for _ in range(num_tokens):
        logits, _ = model(idx[:, -config.context_size:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        yield nxt.item()  # .item() forces the MPS sync we time against


def _wrap(text, width):
    # Split into display lines, honoring newlines and soft-wrapping at width.
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
        for i in range(0, len(para), width):
            lines.append(para[i:i + width])
    return lines


def stream_render(stream, itos):
    if not sys.stdout.isatty():
        n, t0 = 0, time.perf_counter()
        for tid in stream:
            sys.stdout.write(itos[tid]); n += 1
        sys.stdout.flush()
        dt = time.perf_counter() - t0
        if dt > 0:
            print(f"\n{n / dt:.1f} tok/s", file=sys.stderr)
        return

    out = sys.stdout
    cols, rows = shutil.get_terminal_size()
    body = rows - 1
    chars, n, t0 = [], 0, time.perf_counter()
    try:
        out.write("\x1b[2J\x1b[?25l")  # clear once, hide cursor
        for tid in stream:
            chars.append(itos[tid])
            n += 1
            dt = time.perf_counter() - t0
            rate = f" {n / dt:.1f} tok/s " if dt > 0 else ""
            lines = _wrap("".join(chars), cols)[-body:]
            buf = ["\x1b[H"]  # cursor home, then redraw the whole window
            for i in range(body):
                buf.append((lines[i] if i < len(lines) else "") + "\x1b[K\r\n")
            buf.append(f"\x1b[7m{rate}\x1b[0m\x1b[K")  # bottom status line
            out.write("".join(buf))
            out.flush()
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[?25h\n")  # show cursor, drop below
        out.flush()


def _abbr(name):
    return "".join(w[0] for w in name.split("_")).upper()


def ckpt_path(config):
    arch = "_".join(f"{_abbr(k)}{getattr(config, k)}" for k in _arch_fields(config))
    return f"checkpoints/{config.name}_{arch}.pt"


def save_ckpt(model, opt, it, val, config, stoi, itos):
    os.makedirs("checkpoints", exist_ok=True)
    path = ckpt_path(config)
    torch.save({
        "config": asdict(config),
        "stoi": stoi,
        "itos": itos,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "iter": it,
        "val": val,
        "rng_state": torch.get_rng_state(),
    }, path)
    return path


def _load_states(ckpt, model, opt):
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    opt.load_state_dict(ckpt["optimizer"])
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)
    torch.set_rng_state(ckpt["rng_state"])
    return ckpt["iter"]


def train(model, opt, config, train_data, val_data, stoi, itos, start_iter):
    best_val, stale, stopped = float("inf"), 0, False
    for it in range(start_iter, config.max_iters):
        if it % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"iter {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
            save_ckpt(model, opt, it, losses["val"], config, stoi, itos)
            if losses["val"] < best_val:
                best_val, stale = losses["val"], 0
            else:
                stale += 1
                if config.patience and stale >= config.patience:
                    print(f"early stop at iter {it}: no val improvement for {config.patience} evals")
                    stopped = True
                    break
        xs, ys = get_batch("train", train_data, val_data, config)
        _, loss = model(xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()

    if not stopped:
        losses = estimate_loss(model, train_data, val_data, config)
        print(f"iter {config.max_iters:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        print(f"  saved -> {save_ckpt(model, opt, config.max_iters, losses['val'], config, stoi, itos)}")

    print("".join(itos[i] for i in generate(model, config, 500)))


def _parse_args(default_config):
    p = argparse.ArgumentParser()
    p.add_argument("--resume", help="checkpoint to resume training from")
    p.add_argument("--generate", help="checkpoint to sample from, then exit")
    p.add_argument("--num_tokens", type=int, default=500)
    for f in fields(default_config):
        if f.name == "name":
            continue
        p.add_argument(f"--{f.name}", type=type(getattr(default_config, f.name)), default=None)
    return p.parse_args()


def run(model_cls, default_config):
    args = _parse_args(default_config)
    passed = {f.name: getattr(args, f.name) for f in fields(default_config)
              if f.name != "name" and getattr(args, f.name) is not None}

    if args.generate:
        ckpt = torch.load(args.generate, map_location="cpu")
        config = type(default_config)(**ckpt["config"])
        model = model_cls(config, len(ckpt["stoi"])).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        stream_render(generate(model, config, args.num_tokens), ckpt["itos"])
        return

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        overridable = _base_names() - {"name"}
        ignored = [k for k in passed if k not in overridable and passed[k] != ckpt["config"][k]]
        if ignored:
            print(f"warning: --resume adopts the checkpoint's arch, ignoring {ignored}")
        merged = {**ckpt["config"], **{k: v for k, v in passed.items() if k in overridable}}
        config = type(default_config)(**merged)
        train_data, val_data, stoi, itos = load_data(config)
        model = model_cls(config, len(stoi)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
        start_iter = _load_states(ckpt, model, opt)
        print(f"resumed from {args.resume} at iter {start_iter}")
    else:
        config = replace(default_config, **passed)
        train_data, val_data, stoi, itos = load_data(config)
        model = model_cls(config, len(stoi)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
        start_iter = 0

    train(model, opt, config, train_data, val_data, stoi, itos, start_iter)
