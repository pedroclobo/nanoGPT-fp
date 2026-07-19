# nanoGPT-fp

A from-scratch, character-level GPT trained on the poetry and prose of **Fernando Pessoa**.
It is built up in stages, from a [bigram lookup table](https://en.wikipedia.org/wiki/Bigram) to a full [transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning)).

It takes inspiration from Andrej Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) video and his [nanoGPT](https://github.com/karpathy/nanoGPT) repo.

![poetry sample](assets/poetry.gif)

## Models

| Model | Number of Parameters |
|---|---|
| [`bigram.py`](bigram.py) | 10.6K |
| [`attention.py`](attention.py) | 31.8K |
| [`blocks.py`](blocks.py) | 4.8M |

### Bigram

A single embedding table maps each character straight to logits over the next character.
There is no context beyond the current character.
It sets a baseline and puts the tokenizer, training loop, and sampling in place.

Most of the output is unrecognizable.
Even so, a few short Portuguese words already surface.

![bigram sample](assets/bigram.gif)

### Attention

Adds token and positional embeddings, then a single multi-head self-attention block.
Each position can now attend to every earlier character in the context window, not just the previous one.
This is the first model to pick up word-like structure.

Based on the infamous ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper.

Clearer patterns emerge, with more recognizable Portuguese words taking shape.

![attention sample](assets/attention.gif)

### Blocks

Stacks $N$ transformer blocks, each pairing multi-head attention with a feed-forward MLP.
Residual connections and pre-norm LayerNorm keep the deeper network trainable, and dropout regularizes it.
This is the full transformer and the one behind the released checkpoints.

![blocks sample](assets/blocks.gif)

Nearly every word is now recognizable Portuguese, laid out as verse.
It reads like a poem, though it carries no real semantic meaning.

## Dataset

Poems and prose scraped from [arquivopessoa.net](http://arquivopessoa.net), via the [turing-usp](https://github.com/turing-usp/fernando-pessoa) CSV.
Filtered to Fernando Pessoa and his main heteronyms, Portuguese only, original spelling.
Then reduced to a curated character whitelist.

```bash
# Poetry -> input.txt  (~1.1 MB)
python prepare_data.py --type poetry
# Prose -> input.txt   (~3.4 MB)
python prepare_data.py --type prose
```

## Results

`blocks.py` (E256 / H8 / L6), best validation loss per corpus:

| Corpus | Number of Parameters | Context Size | Training Iterations | Validation Loss |
|---|---|---|---|---|
| poetry (~1.1 MB) | 4.82M | 128 | 10.5k | 1.397 |
| prose (~3.4 MB) | 4.85M | 256 | 24k | 1.157 |

The larger prose corpus overfits later, reaching a lower loss.

## Samples

| Poetry | Prose |
|---|---|
| ![poetry](assets/poetry.gif) | ![prose](assets/prose.gif) |

## Usage

```bash
# Extract the data
python prepare_data.py --type poetry

# Train, flags override the model's Config defaults
python blocks.py --n_embd 256 --n_head 8 --n_layer 6 --context_size 128 --dropout 0.3

# Resume, adopting the arch from the checkpoint and training flags from the CLI
python blocks.py --resume checkpoints/blocks_CS128_NE256_NH8_NL6_D0.3.pt --max_iters 20000

# Generate, streaming token by token with a live tok/s readout
python blocks.py --generate checkpoints/poetry.pt --num_tokens 1000
```

Training auto-saves a checkpoint to `checkpoints/<model>_<arch>.pt` each eval interval.
It early-stops when validation stops improving (check `--patience`).

## Checkpoints

Pre-trained `blocks` weights are attached to the [v1.0 release](https://github.com/pedroclobo/nanoGPT-fp/releases/tag/v1.0): `poetry.pt` and `prose.pt`.
Download one and point `--generate` at it.
