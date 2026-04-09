import argparse
import json
import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/shakespeare.txt")
    ap.add_argument("--out_dir", type=str, default="checkpoints/tiny-gpt")
    ap.add_argument("--vocab_size", type=int, default=4096)
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.data).read_text(encoding="utf-8")
    tok = RegexBPETokenizer()
    tok.train(text, args.vocab_size, verbose=True)
    ids = tok.encode(text)

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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = TokenDataset(ids, cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    it = iter(dl)

    model.train()
    pbar = tqdm(range(args.steps), desc=f"train ({device})")
    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)

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

