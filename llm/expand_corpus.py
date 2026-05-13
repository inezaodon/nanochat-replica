"""
Build a larger plain-text training corpus from public sources.

Sources supported here:
  * **Hugging Face Datasets** — streamed text columns (no full RAM download for big corpora).
  * **Project Gutenberg** — direct cache URLs for given ebook IDs (plain UTF-8).
  * **Local files** — append any UTF-8 paths you already have (e.g. `data/shakespeare.txt`).

**Kaggle** datasets are not auto-unpacked here: the official API needs `~/.kaggle/kaggle.json`
credentials. If you use Kaggle, download CSV/JSON locally, then pass those paths via
`--include-local`.

Examples:
  python -m pip install datasets  # if not already installed

  python -m llm.expand_corpus \\
    --out data/corpus_expanded.txt \\
    --include-local data/training_corpus.txt \\
    --hf-preset wikitext-103 ag_news imdb \\
    --gutenberg 11 1342 \\
    --max-chars-per-preset 150000 \\
    --max-chars-gutenberg 250000

  # Custom HF dataset (dataset, config, split, text column — use - for empty config)
  python -m llm.expand_corpus --out data/mix.txt --hf-custom "wikitext,wikitext-2-raw-v1,train,text"
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# Preset: (dataset_id, config_name or None, split, text_field)
HF_PRESETS: Dict[str, Tuple[str, Optional[str], str, str]] = {
    # English Wikipedia snippets (raw); good general-domain LM text.
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "train", "text"),
    # News headlines + article blurbs
    "ag_news": ("ag_news", None, "train", "text"),
    # Movie reviews (sentiment LM / style)
    "imdb": ("imdb", None, "train", "text"),
    # Longer Yelp reviews
    "yelp": ("yelp_review_full", None, "train", "text"),
    # Short tweets (informal language)
    "tweet_sentiment": ("tweet_eval", "sentiment", "train", "text"),
}


def _write_header(f, title: str) -> None:
    f.write(f"\n\n=== {title} ===\n\n")


def _chars_from_gutenberg(book_id: int, max_chars: int) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"[gutenberg] fetch {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "nanochat-replica-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310 - intentional URL
        raw = resp.read().decode("utf-8", errors="replace")
    # Strip boilerplate: keep from first "*** START OF" if present
    m = re.search(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", raw, re.I | re.S)
    if m:
        raw = raw[m.end() :]
    m2 = re.search(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK", raw, re.I | re.S)
    if m2:
        raw = raw[: m2.start()]
    raw = raw.strip()
    if len(raw) > max_chars:
        raw = raw[:max_chars]
    return raw


def _stream_hf(
    dataset_id: str,
    config: Optional[str],
    split: str,
    text_field: str,
) -> Iterator[str]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    kwargs: Dict[str, Any] = {"split": split, "streaming": True, "trust_remote_code": False}
    if config is not None:
        ds = load_dataset(dataset_id, config, **kwargs)
    else:
        ds = load_dataset(dataset_id, **kwargs)

    for row in ds:
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        yield text.strip()


def _parse_hf_custom(spec: str) -> Tuple[str, Optional[str], str, str]:
    """Format: dataset,config,split,column — use '-' for empty config."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("--hf-custom needs exactly 4 comma parts: dataset,config,split,column")
    ds, cfg, split, col = parts
    cfg_o = None if cfg in ("", "-", "none", "None") else cfg
    return ds, cfg_o, split, col


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand UTF-8 LM corpus from HF / Gutenberg / local files.")
    ap.add_argument("--out", type=str, default="data/corpus_expanded.txt", help="Output UTF-8 text path.")
    ap.add_argument(
        "--include-local",
        action="append",
        default=[],
        metavar="PATH",
        help="Append this UTF-8 file (repeat flag for multiple).",
    )
    ap.add_argument(
        "--hf-preset",
        action="append",
        default=[],
        choices=list(HF_PRESETS.keys()),
        help=f"HuggingFace preset name. Choices: {', '.join(sorted(HF_PRESETS))}.",
    )
    ap.add_argument(
        "--hf-custom",
        action="append",
        default=[],
        metavar="DS,CFG,SPLIT,COL",
        help="Custom HF stream: dataset,config (or -),split,column",
    )
    ap.add_argument(
        "--gutenberg",
        type=int,
        action="append",
        default=[],
        metavar="EBOOK_ID",
        help="Project Gutenberg ebook id (e.g. 11 = Alice, 1342 = Pride and Prejudice). Repeatable.",
    )
    ap.add_argument(
        "--max-chars-per-preset",
        type=int,
        default=200_000,
        help="Max characters to take from each HF preset / custom stream.",
    )
    ap.add_argument(
        "--max-chars-gutenberg",
        type=int,
        default=300_000,
        help="Max characters per Gutenberg download (truncates long books).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for local in args.include_local:
            p = Path(local)
            if not p.is_file():
                raise FileNotFoundError(local)
            _write_header(out, f"LOCAL:{p}")
            text = p.read_text(encoding="utf-8", errors="replace")
            out.write(text)
            total_written += len(text)
            print(f"[local] {p} ({len(text):,} chars)")

        for gid in args.gutenberg:
            _write_header(out, f"GUTENBERG:{gid}")
            chunk = _chars_from_gutenberg(gid, args.max_chars_gutenberg)
            out.write(chunk)
            total_written += len(chunk)
            print(f"[gutenberg] id={gid} wrote {len(chunk):,} chars")

        for preset in args.hf_preset or []:
            ds_id, cfg, split, col = HF_PRESETS[preset]
            _write_header(out, f"HF:{ds_id}/{cfg or 'default'}/{split}")
            n = 0
            for piece in _stream_hf(ds_id, cfg, split, col):
                if n >= args.max_chars_per_preset:
                    break
                room = args.max_chars_per_preset - n
                chunk = piece if len(piece) <= room else piece[:room]
                out.write(chunk)
                out.write("\n\n")
                n += len(chunk)
            total_written += n
            print(f"[hf] preset={preset} wrote {n:,} chars")

        for custom in args.hf_custom or []:
            ds_id, cfg, split, col = _parse_hf_custom(custom)
            label = f"HF:{ds_id}/{cfg or 'default'}/{split}/{col}"
            _write_header(out, label)
            n = 0
            for piece in _stream_hf(ds_id, cfg, split, col):
                if n >= args.max_chars_per_preset:
                    break
                room = args.max_chars_per_preset - n
                chunk = piece if len(piece) <= room else piece[:room]
                out.write(chunk)
                out.write("\n\n")
                n += len(chunk)
            total_written += n
            print(f"[hf] custom={custom} wrote {n:,} chars")

    print(f"[done] wrote {out_path} (≈{total_written:,} chars from counted blocks; file size {out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
