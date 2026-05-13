"""
Supervised fine-tuning (SFT) on Alpaca-style JSON with Hugging Face GPT-2.

Matches the course lab ideas:
  - Alpaca prompt template (see llm.sft_format)
  - Loss only on response tokens (labels = -100 on prompt)
  - Optional partial freeze: train only last K transformer blocks + ln_f + lm_head

Example (small model, CPU-friendly):
  python -m llm.sft_train --model distilgpt2 --data data/course/alpaca_instruction_data.json --max_samples 256 --steps 200

Full Alpaca (after fetch):
  python -m llm.fetch_course_datasets --fetch-hf-alpaca
  python -m llm.sft_train --model distilgpt2 --data data/course/stanford_alpaca_data.json --max_samples 2000 --steps 1500
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.sft_dataset import AlpacaSFTDataset, collate_sft_batch, load_alpaca_records


def set_trainable_layers(model: torch.nn.Module, n_trainable_blocks: int) -> None:
    """Freeze all GPT2Block modules except the last `n_trainable_blocks` (plus ln_f + lm_head stay trainable)."""
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Expected a GPT-2-style model with .transformer.h (e.g. gpt2, distilgpt2).")
    blocks = list(model.transformer.h)
    n = len(blocks)
    freeze_up_to = max(0, n - n_trainable_blocks)
    for p in model.parameters():
        p.requires_grad = False
    for i in range(freeze_up_to, n):
        for p in blocks[i].parameters():
            p.requires_grad = True
    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True
    if n_trainable_blocks >= n:
        for p in model.transformer.wte.parameters():
            p.requires_grad = True
        if hasattr(model.transformer, "wpe") and model.transformer.wpe is not None:
            for p in model.transformer.wpe.parameters():
                p.requires_grad = True


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT on Alpaca JSON with GPT-2 (HF).")
    ap.add_argument("--model", type=str, default="distilgpt2", help="HF model id (gpt2, distilgpt2, …).")
    ap.add_argument(
        "--data",
        type=str,
        default="data/course/alpaca_instruction_data.json",
        help="Path to JSON array [{instruction, input, output}, …].",
    )
    ap.add_argument("--out_dir", type=str, default="checkpoints/sft-gpt2")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = use all rows in file.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument(
        "--trainable_blocks",
        type=int,
        default=1,
        help="Number of last transformer blocks to train (rest frozen). Use 999 to train all blocks.",
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Micro-batches per optimizer step (effective batch = batch_size * grad_accum).",
    )
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 autocast on CUDA (when supported).",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers for tokenized JSON (0 = main thread).",
    )
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"{data_path} not found. Run: python -m llm.fetch_course_datasets\n"
            "Optional full Alpaca: python -m llm.fetch_course_datasets --fetch-hf-alpaca"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_alpaca_records(data_path)
    max_samples = None if args.max_samples == 0 else args.max_samples
    ds = AlpacaSFTDataset(records, tokenizer, max_length=args.max_length, max_samples=max_samples)
    collate = partial(collate_sft_batch, pad_id=tokenizer.pad_token_id)
    pin = device.type == "cuda"
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        pin_memory=pin,
        num_workers=max(0, args.num_workers),
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.train()

    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()

    n_blocks = len(model.transformer.h)
    n_train = min(args.trainable_blocks, n_blocks) if args.trainable_blocks < 900 else n_blocks
    set_trainable_layers(model, n_train)
    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(train_params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    it = iter(dl)
    pbar = tqdm(range(args.steps), desc=f"sft ({device})")
    accum = max(1, args.grad_accum)
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            labels = batch["labels"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)

            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / accum
            total_loss += float(loss.detach().cpu())
            loss.backward()

        torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        opt.step()
        pbar.set_postfix(loss=total_loss)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved model + tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()
