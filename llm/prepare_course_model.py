"""
One-shot: download ND release GPT-2 weights (if needed), convert to browser bundle.

Example:
  python -m llm.prepare_course_model \\
    --weight_file gpt2_small_converted.pt \\
    --out_dir public/models/gpt2-small

Then in the web app, load the gpt2-small preset (or point export_web at weights/...).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.course_weights import COURSE_WEIGHT_FILES, ensure_course_checkpoint
from llm.export_web import export_browser_bundle
from llm.gpt2_course_export import load_native_checkpoint, native_gpt2_dict_to_export


def write_gpt2_tokenizer_json(path: Path) -> None:
    path.write_text(
        json.dumps({"tokenizer_type": "gpt2_tiktoken", "encoding": "gpt2"}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + export ND weights-v1 GPT-2 for the browser.")
    ap.add_argument(
        "--weight_file",
        type=str,
        default="gpt2_small_converted.pt",
        choices=list(COURSE_WEIGHT_FILES),
        help="Filename from the weights-v1 GitHub Release.",
    )
    ap.add_argument(
        "--weights_dir",
        type=str,
        default="",
        help="Directory to store downloaded .pt (default: <repo>/weights).",
    )
    ap.add_argument("--out_dir", type=str, default="public/models/gpt2-small")
    ap.add_argument("--force_download", action="store_true")
    args = ap.parse_args()

    dest_root = Path(args.weights_dir) if args.weights_dir else None
    local_pt = ensure_course_checkpoint(args.weight_file, dest_dir=dest_root, force=args.force_download)

    ckpt = load_native_checkpoint(local_pt)
    cfg, sd = native_gpt2_dict_to_export(ckpt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    write_gpt2_tokenizer_json(tok_path)

    export_browser_bundle(
        cfg,
        sd,
        tok_path,
        out_dir,
        tokenizer_type="gpt2_tiktoken",
        manifest_extras={"eos_token_id": 50256},
    )

    print(f"[prepare_course_model] done. Browser bundle at {out_dir}")


if __name__ == "__main__":
    main()
