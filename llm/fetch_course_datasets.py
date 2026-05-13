"""
Download datasets from the ND CSE 10124 lecture repository (same files as on GitHub).

  https://github.com/wtheisen/nd-cse-10124-lectures/tree/master/Datasets

Default output: data/course/  (gitignored large files listed in .gitignore)

Examples:
  python -m llm.fetch_course_datasets
  python -m llm.fetch_course_datasets --fetch-hf-alpaca
  python -m llm.fetch_course_datasets --include-large
  python -m llm.fetch_course_datasets --only alpaca_instruction_data.json,tool_call_sft_small.jsonl
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Iterable, List

RAW_BASE = "https://raw.githubusercontent.com/wtheisen/nd-cse-10124-lectures/master/Datasets"

# (relative path under Datasets/, description, include_by_default)
MANIFEST: list[tuple[str, str, bool]] = [
    ("alpaca_instruction_data.json", "Course Alpaca-style instruction SFT JSON (~200 KB)", True),
    ("irishchat_eval_v1.jsonl", "IrishChat eval prompts (jsonl)", True),
    ("tool_call_sft_small.jsonl", "Small tool-calling SFT jsonl", True),
    ("tool_call_sft_small_alpaca.json", "Tool-calling SFT in Alpaca-style JSON", True),
    ("jabberwocky.txt", "Short text sample", True),
    ("zoomer.txt", "Short styled text sample", True),
    ("practice_packet_01_solutions.json", "Practice packet solutions JSON", True),
    ("shakespeare.txt", "Shakespeare full text (duplicate of common course data)", True),
    ("shakespeare_vocab.json", "Shakespeare vocab JSON", True),
    ("openweb10k_tokenizer.json", "Tokenizer export for openweb corpus", True),
    ("irishchat_rag.sqlite", "SQLite DB for RAG lab (~500 KB)", True),
    ("openweb10k.txt", "Large web-scrape text corpus (~50 MB)", False),
]

TOKENIZER_ASSETS = [
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def default_out_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "course"


def _url(rel: str) -> str:
    return f"{RAW_BASE}/{rel.replace(' ', '%20')}"


def download(rel: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} ({dest.stat().st_size} bytes)")
        return
    url = _url(rel)
    print(f"[get] {url}\n      -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Datasets/ from nd-cse-10124-lectures into data/course/.")
    ap.add_argument("--out_dir", type=str, default="", help="Override output directory (default: data/course)")
    ap.add_argument("--include-large", action="store_true", help=f"Also fetch {MANIFEST[-1][0]} (~50 MB).")
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of top-level filenames to fetch (still fetches tokenizer assets unless --no-tokenizer-assets).",
    )
    ap.add_argument("--no-tokenizer-assets", action="store_true", help="Skip Datasets/gpt2_tokenizer_assets/*.")
    ap.add_argument(
        "--fetch-hf-alpaca",
        action="store_true",
        help="Also download full Stanford Alpaca JSON (52k) into data/course/stanford_alpaca_data.json.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)

    selected: Iterable[tuple[str, str, bool]] = MANIFEST
    if args.only:
        names = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [m for m in MANIFEST if m[0] in names]

    for rel, desc, default_on in selected:
        if not default_on and not args.include_large:
            print(f"[skip large] {rel} — {desc} (pass --include-large)")
            continue
        download(rel, out / rel, force=args.force)

    if not args.no_tokenizer_assets:
        tok_dir = out / "gpt2_tokenizer_assets"
        for name in TOKENIZER_ASSETS:
            download(f"gpt2_tokenizer_assets/{name}", tok_dir / name, force=args.force)

    if args.fetch_hf_alpaca:
        fetch_hf_alpaca_json(out / "stanford_alpaca_data.json")

    sources = out / "SOURCES.txt"
    lines = [
        "Files mirror:",
        "  https://github.com/wtheisen/nd-cse-10124-lectures/tree/master/Datasets",
        "",
        "Full Stanford Alpaca (52k): data/course/stanford_alpaca_data.json (from --fetch-hf-alpaca)",
        "",
    ]
    sources.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {sources}")


def fetch_hf_alpaca_json(out_path: Path | None = None) -> Path:
    """Download tatsu-lab/stanford_alpaca alpaca_data.json from HuggingFace git mirror (no `datasets` install)."""
    out = out_path or (default_out_dir() / "stanford_alpaca_data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Raw JSON from the official Stanford Alpaca repo (same 52k entries the field uses).
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    print(f"[get] HF-style full Alpaca\n      {url}\n      -> {out}")
    urllib.request.urlretrieve(url, out)
    return out


if __name__ == "__main__":
    main()
