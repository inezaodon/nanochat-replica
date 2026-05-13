import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm.model import GPT, GPTConfig
from llm.tokenizer_bpe import RegexBPETokenizer


class TokenDataset(Dataset):
    def __init__(self, ids, block_size: int):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, self.ids.numel() - self.block_size - 1)

    def __getitem__(self, i):
        x = self.ids[i : i + self.block_size]
        y = self.ids[i + 1 : i + 1 + self.block_size]
        return x, y


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_training_text(data_arg: str) -> tuple[str, list[str]]:
    """
    Load UTF-8 training text from one or more files.
    `--data` may be a single path or comma-separated paths, concatenated in order
    with a newline between files.
    """
    parts = [p.strip() for p in data_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("--data must name at least one text file.")
    chunks: list[str] = []
    labels: list[str] = []
    for p in parts:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Training data file not found: {path}")
        chunks.append(path.read_text(encoding="utf-8"))
        labels.append(str(path))
    text = "\n\n".join(chunks)
    return text, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="data/training_corpus.txt",
        help="UTF-8 text file(s), comma-separated; concatenated in order (default: multi-source corpus in repo).",
    )
    ap.add_argument("--out_dir", type=str, default="checkpoints/tiny-gpt")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument(
        "--tokenizer_chars",
        type=int,
        default=200_000,
        help="Train the tokenizer on only the first N characters for speed (0 = full text).",
    )
    ap.add_argument(
        "--train_chars",
        type=int,
        default=300_000,
        help="Train the model on only the first N characters for speed (0 = full text).",
    )
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=12)
    ap.add_argument("--n_embd", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process only). Try 4 on Linux/macOS for faster batching.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Wrap model in torch.compile (CUDA only; PyTorch 2+).",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_full, sources = load_training_text(args.data)
    print(f"[data] sources: {sources}")
    text_tok = text_full if args.tokenizer_chars == 0 else text_full[: args.tokenizer_chars]
    text_train = text_full if args.train_chars == 0 else text_full[: args.train_chars]

    print(f"[data] tokenizer_chars={len(text_tok):,} train_chars={len(text_train):,} file_chars={len(text_full):,}")
    tok = RegexBPETokenizer()
    print("[tokenizer] building character vocabulary…")
    tok.train(text_tok, args.vocab_size, verbose=True)
    print(f"[tokenizer] done. vocab={len(tok.vocab):,}")
    print("[tokenizer] encoding training text…")
    ids = tok.encode(text_train)
    print(f"[tokenizer] encoded tokens={len(ids):,}")

    # Save tokenizer export next to checkpoint for later browser export
    (out_dir / "tokenizer.json").write_text(json.dumps(tok.export_json()), encoding="utf-8")

    cfg = GPTConfig(
        vocab_size=len(tok.vocab),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    model = GPT(cfg).to(device)
    if args.compile:
        if device.type != "cuda":
            raise ValueError("--compile is only supported with CUDA in this repo.")
        model = torch.compile(model)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = TokenDataset(ids, cfg.block_size)
    pin = device.type == "cuda"
    nw = max(0, args.num_workers)
    dl_kw = dict(batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin, num_workers=nw)
    if nw > 0 and platform.system() != "Windows":
        dl_kw["persistent_workers"] = True
    dl = DataLoader(ds, **dl_kw)
    it = iter(dl)

    model.train()
    pbar = tqdm(range(args.steps), desc=f"train ({device})")
    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        nb = pin and device.type == "cuda"
        x = x.to(device, non_blocking=nb)
        y = y.to(device, non_blocking=nb)

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        pbar.set_postfix(loss=float(loss.detach().cpu()))

        if (step + 1) % args.eval_every == 0:
            ckpt = {
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, out_dir / "model.pt")

    ckpt = {
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, out_dir / "model.pt")
    print(f"saved: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

