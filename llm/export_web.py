import argparse
import json
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def write_f32_bin(path: Path, arrays):
    # arrays: list of (name, tensor_cpu_contig_float32)
    offsets = {}
    off = 0
    with open(path, "wb") as f:
        for name, t in arrays:
            b = t.numpy().tobytes(order="C")
            offsets[name] = {"offset": off, "nbytes": len(b)}
            f.write(b)
            off += len(b)
    return offsets


def export_browser_bundle(
    cfg: Dict[str, Any],
    sd: Dict[str, torch.Tensor],
    tokenizer_src: Path,
    out_dir: Path,
    *,
    tokenizer_type: str | None = None,
    manifest_extras: Dict[str, Any] | None = None,
) -> None:
    """Write manifest.json, weights.f32.bin, and tokenizer.json for the web app."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = []
    shapes = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        t = v.detach().cpu().contiguous().float()
        tensors.append((k, t))
        shapes[k] = list(t.shape)

    bin_path = out_dir / "weights.f32.bin"
    offsets = write_f32_bin(bin_path, tensors)

    manifest: Dict[str, Any] = {
        "format": "f32",
        "weights": "weights.f32.bin",
        "tensors": {name: {"shape": shapes[name], **offsets[name]} for name, _ in tensors},
        "config": cfg,
    }
    if tokenizer_type:
        manifest["tokenizer_type"] = tokenizer_type
    if manifest_extras:
        manifest.update(manifest_extras)

    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    tok_json = tokenizer_src.read_text(encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(tok_json, encoding="utf-8")

    print(f"wrote: {out_dir / 'manifest.json'}")


def _load_training_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(
            f"{path} is not a training checkpoint (expected dict with 'config' and 'state_dict'). "
            "Use --native_gpt2 with a course release .pt, or python -m llm.prepare_course_model."
        )
    return ckpt["config"], ckpt["state_dict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/tiny-gpt/model.pt")
    ap.add_argument("--tokenizer", type=str, default="checkpoints/tiny-gpt/tokenizer.json")
    ap.add_argument("--out_dir", type=str, default="public/models/tiny-gpt")
    ap.add_argument(
        "--native_gpt2",
        action="store_true",
        help="Load IrishChat-style native dict (.pt from weights-v1 release), not nanochat train format.",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    tok_path = Path(args.tokenizer)
    out_dir = Path(args.out_dir)

    if args.native_gpt2:
        from llm.gpt2_course_export import load_native_checkpoint, native_gpt2_dict_to_export

        ckpt = load_native_checkpoint(ckpt_path)
        cfg, sd = native_gpt2_dict_to_export(ckpt)
        export_browser_bundle(
            cfg,
            sd,
            tok_path,
            out_dir,
            tokenizer_type="gpt2_tiktoken",
            manifest_extras={"eos_token_id": 50256},
        )
    else:
        cfg, sd = _load_training_checkpoint(ckpt_path)
        export_browser_bundle(cfg, sd, tok_path, out_dir)


if __name__ == "__main__":
    main()
