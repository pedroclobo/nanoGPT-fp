"""
Build the Fernando Pessoa poetry corpus.

Source: poems scraped from arquivopessoa.net, distributed as a single CSV by the
USP Turing group (https://github.com/turing-usp/fernando-pessoa).
"""

import csv
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
CSV_URL = "https://raw.githubusercontent.com/turing-usp/fernando-pessoa/master/fernando_pessoa.csv"
CSV_PATH = ROOT / "data" / "fernando_pessoa.csv"
OUT_PATH = ROOT / "input.txt"

def load_rows(path):
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(CSV_URL, path)
    csv.field_size_limit(10_000_000)  # texto fields exceed the default cap
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)

# Distinctive function words per language
PT_WORDS = {
    "não", "uma", "com", "para", "mas", "como", "é", "são", "está", "você",
    "também", "muito", "nada", "tudo", "ser", "tão", "mesmo", "sem", "seu",
    "meu", "minha", "quando", "porque", "já", "só", "então", "coisas",
}
EN_WORDS = {
    "the", "and", "of", "to", "is", "in", "that", "with", "was", "for",
    "my", "your", "there", "which", "would", "but", "they", "this",
}
FR_WORDS = {
    "le", "les", "des", "une", "est", "dans", "je", "il", "au", "du",
    "pas", "qui", "vous", "ce", "cette", "aux", "être", "sont", "mais",
}
WORD_RE = re.compile(r"[a-zà-ÿ]+")


def is_portuguese(text):
    words = WORD_RE.findall(text.lower())
    pt = sum(w in PT_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    fr = sum(w in FR_WORDS for w in words)
    return pt >= en and pt >= fr

AUTHORS = {"Fernando Pessoa", "Alberto Caeiro", "Ricardo Reis", "Álvaro de Campos"}

def keep_row(row):
    return row["tipo"] == "poesia" and row["autor"] in AUTHORS and is_portuguese(row["texto"])

# editorial spans: [?], [a ver?], que[m]
BRACKET_RE = re.compile(r"\[[^\]]*\]")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTINL_RE = re.compile(r"\n{3,}")

# One translate pass: delete markup noise, unify dashes/quotes, strip foreign
# accents. Legit PT accents, guillemets « », and ordinals ª º are kept.
TRANS = {
    **{ord(c): None for c in "\xad_*[]<>\\"},
    **{ord(c): "-" for c in "‐‑‒–—"},
    **{ord(k): v for k, v in {
        "Ë": "E", "ë": "e", "È": "E", "è": "e", "Ì": "I", "ì": "i",
        "Î": "I", "î": "i", "Ï": "I", "ï": "i", "Ò": "O", "ò": "o",
        "Ù": "U", "ù": "u", "Ñ": "N", "ñ": "n",
        "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
    }.items()},
}


def clean(text):
    """Strip editorial cruft, keep original spelling."""
    text = BRACKET_RE.sub("", text)
    text = text.translate(TRANS)
    lines = [ln.rstrip() for ln in text.split("\n") if set(ln.strip()) != {"."}]
    text = "\n".join(lines)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINL_RE.sub("\n\n", text)
    return text.strip()


def dedup(texts):
    """Drop duplicates by a whitespace/case-insensitive key."""
    seen, out = set(), []
    for t in texts:
        key = " ".join(t.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def main():
    rows = [r for r in load_rows(CSV_PATH) if keep_row(r)]
    texts = dedup([clean(r["texto"]) for r in rows])
    corpus = "\n\n".join(texts) + "\n"
    chars = set(corpus)
    OUT_PATH.write_text(corpus, encoding="utf-8")
    print(
        f"wrote {OUT_PATH.name}: {len(texts)} texts, {len(corpus):,} chars, "
        f"{len(chars)} unique chars"
    )
    print(sorted(list(chars)))


if __name__ == "__main__":
    main()
