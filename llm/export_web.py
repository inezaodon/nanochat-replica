import argparse
import json
import struct
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/tiny-gpt/model.pt")
    ap.add_argument("--tokenizer", type=str, default="checkpoints/tiny-gpt/tokenizer.json")
    ap.add_argument("--out_dir", type=str, default="public/models/tiny-gpt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    sd = ckpt["state_dict"]

    # Export a minimal set of tensors; all float32.
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

    manifest = {
        "format": "f32",
        "weights": "weights.f32.bin",
        "tensors": {name: {"shape": shapes[name], **offsets[name]} for name, _ in tensors},
        "config": cfg,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    tok_json = Path(args.tokenizer).read_text(encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(tok_json, encoding="utf-8")

    print(f"wrote: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

