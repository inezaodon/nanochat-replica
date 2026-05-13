"""
Convert ND IrishChat-style native GPT-2 checkpoints (dict layout) into the
{config, state_dict} format expected by llm.export_web and the browser runtime.

Native layout matches irishGPT.irishChat.IrishChat.load_converted_gpt2_checkpoint:
  wte, wpe, blocks[i] with ln_1 / attn / ln_2 / mlp keys, ln_f.*, optional lm_head.weight
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def _require_keys(d: Dict[str, Any], keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where}: missing keys {missing}")


def native_gpt2_dict_to_export(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Map a native converted GPT-2 dict to manifest config + flat state_dict
    (parameter names aligned with llm.model.GPT and src/core/inferTinyGPT.ts).
    """
    _require_keys(ckpt, ["wte", "wpe", "blocks", "ln_f.weight", "ln_f.bias"], "top-level")

    wte = ckpt["wte"]
    wpe = ckpt["wpe"]
    blocks = ckpt["blocks"]
    if not isinstance(blocks, list) or len(blocks) == 0:
        raise ValueError("ckpt['blocks'] must be a non-empty list")

    vocab_size, n_embd = wte.shape
    ctx_len, n_embd_wpe = wpe.shape
    if n_embd != n_embd_wpe:
        raise ValueError(f"wte / wpe embedding dim mismatch: {n_embd} vs {n_embd_wpe}")

    n_layer = len(blocks)
    # Infer head count from attention c_attn shape (C, 3C) with standard GPT-2 layout.
    c0 = blocks[0]
    _require_keys(
        c0,
        [
            "ln_1.weight",
            "ln_1.bias",
            "ln_2.weight",
            "ln_2.bias",
            "attn.c_attn.weight",
            "attn.c_attn.bias",
            "attn.c_proj.weight",
            "attn.c_proj.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ],
        "blocks[0]",
    )
    c_attn_w = c0["attn.c_attn.weight"]
    if c_attn_w.shape != (n_embd, 3 * n_embd):
        raise ValueError(f"Unexpected attn.c_attn.weight shape {tuple(c_attn_w.shape)}; expected ({n_embd}, {3 * n_embd})")

    # Heuristic: GPT-2 small uses 12 heads; keep strict only for known width.
    common_heads = {64: 1, 128: 2, 256: 4, 384: 6, 512: 8, 768: 12, 1024: 16}
    if n_embd not in common_heads:
        raise ValueError(
            f"Cannot infer n_head for n_embd={n_embd}; add mapping in gpt2_course_export.py if you use a custom width."
        )
    n_head = common_heads[n_embd]
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd {n_embd} not divisible by n_head {n_head}")

    sd: Dict[str, torch.Tensor] = {}
    sd["tok_emb.weight"] = wte.detach().cpu().contiguous().float()
    sd["pos_emb.weight"] = wpe.detach().cpu().contiguous().float()
    sd["ln_f.weight"] = ckpt["ln_f.weight"].detach().cpu().contiguous().float()
    sd["ln_f.bias"] = ckpt["ln_f.bias"].detach().cpu().contiguous().float()

    if "lm_head.weight" in ckpt:
        sd["lm_head.weight"] = ckpt["lm_head.weight"].detach().cpu().contiguous().float()
    else:
        sd["lm_head.weight"] = sd["tok_emb.weight"].clone()

    for i, b in enumerate(blocks):
        prefix = f"blocks.{i}"
        sd[f"{prefix}.ln1.weight"] = b["ln_1.weight"].detach().cpu().contiguous().float()
        sd[f"{prefix}.ln1.bias"] = b["ln_1.bias"].detach().cpu().contiguous().float()
        sd[f"{prefix}.ln2.weight"] = b["ln_2.weight"].detach().cpu().contiguous().float()
        sd[f"{prefix}.ln2.bias"] = b["ln_2.bias"].detach().cpu().contiguous().float()

        wq, wk, wv = torch.chunk(b["attn.c_attn.weight"], 3, dim=1)
        bq, bk, bv = torch.chunk(b["attn.c_attn.bias"], 3, dim=0)
        qkv_w = torch.cat([wq.T, wk.T, wv.T], dim=0).contiguous().float()
        qkv_b = torch.cat([bq, bk, bv], dim=0).contiguous().float()
        sd[f"{prefix}.attn.qkv.weight"] = qkv_w
        sd[f"{prefix}.attn.qkv.bias"] = qkv_b

        sd[f"{prefix}.attn.proj.weight"] = b["attn.c_proj.weight"].T.detach().cpu().contiguous().float()
        sd[f"{prefix}.attn.proj.bias"] = b["attn.c_proj.bias"].detach().cpu().contiguous().float()

        sd[f"{prefix}.mlp.fc.weight"] = b["mlp.c_fc.weight"].T.detach().cpu().contiguous().float()
        sd[f"{prefix}.mlp.fc.bias"] = b["mlp.c_fc.bias"].detach().cpu().contiguous().float()
        sd[f"{prefix}.mlp.proj.weight"] = b["mlp.c_proj.weight"].T.detach().cpu().contiguous().float()
        sd[f"{prefix}.mlp.proj.bias"] = b["mlp.c_proj.bias"].detach().cpu().contiguous().float()

    config = {
        "vocab_size": int(vocab_size),
        "block_size": int(ctx_len),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_embd": int(n_embd),
        "dropout": 0.0,
    }
    return config, sd


def load_native_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    torch.load wrapper for IrishChat native dict checkpoints.
    weights_only is not supported for nested dict-of-tensors layouts.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def _synthetic_native_ckpt(*, C: int = 128, V: int = 200, T: int = 32, n_layer: int = 2) -> Dict[str, Any]:
    """Tiny random checkpoint for shape tests (matches IrishChat tensor layout)."""
    torch.manual_seed(0)
    wte = torch.randn(V, C)
    wpe = torch.randn(T, C)
    blocks = []
    for _ in range(n_layer):
        blocks.append(
            {
                "ln_1.weight": torch.ones(C),
                "ln_1.bias": torch.zeros(C),
                "ln_2.weight": torch.ones(C),
                "ln_2.bias": torch.zeros(C),
                "attn.c_attn.weight": torch.randn(C, 3 * C),
                "attn.c_attn.bias": torch.randn(3 * C),
                "attn.c_proj.weight": torch.randn(C, C),
                "attn.c_proj.bias": torch.randn(C),
                "mlp.c_fc.weight": torch.randn(C, 4 * C),
                "mlp.c_fc.bias": torch.randn(4 * C),
                "mlp.c_proj.weight": torch.randn(4 * C, C),
                "mlp.c_proj.bias": torch.randn(C),
            }
        )
    return {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f.weight": torch.ones(C),
        "ln_f.bias": torch.zeros(C),
    }


if __name__ == "__main__":
    cfg, sd = native_gpt2_dict_to_export(_synthetic_native_ckpt())
    assert cfg["vocab_size"] == 200 and cfg["n_embd"] == 128 and cfg["n_layer"] == 2
    assert sd["blocks.0.attn.qkv.weight"].shape == (3 * 128, 128)
    print("gpt2_course_export smoke OK:", cfg)
