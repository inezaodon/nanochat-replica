from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from torch.utils.data import Dataset

from llm.sft_format import format_alpaca_entry


def load_alpaca_records(path: Path) -> List[dict[str, Any]]:
    """Load a JSON array of {instruction, input, output} (course or Stanford Alpaca)."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


@dataclass
class AlpacaSFTExample:
    input_ids: torch.Tensor  # (T,) long
    labels: torch.Tensor  # (T,) long, -100 on prompt + padding


class AlpacaSFTDataset(Dataset):
    """
    Token-level SFT: concatenate prompt + response; labels are -100 on prompt tokens
    (standard causal LM masking for instruction tuning).
    """

    def __init__(
        self,
        records: List[dict[str, Any]],
        tokenizer,
        *,
        max_length: int = 512,
        max_samples: int | None = None,
    ):
        self.examples: List[AlpacaSFTExample] = []
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id

        n = 0
        for row in records:
            if max_samples is not None and n >= max_samples:
                break
            inst = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")
            prompt, resp = format_alpaca_entry(inst, inp, out)
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            r_ids = tokenizer.encode(resp + tokenizer.eos_token, add_special_tokens=False)
            ids = p_ids + r_ids
            if len(ids) > max_length:
                # Drop very long sequences (simple strategy); could truncate prompt instead.
                continue
            labels = [-100] * len(p_ids) + r_ids
            self.examples.append(
                AlpacaSFTExample(
                    input_ids=torch.tensor(ids, dtype=torch.long),
                    labels=torch.tensor(labels, dtype=torch.long),
                )
            )
            n += 1

        if not self.examples:
            raise RuntimeError("No valid training examples after filtering; lower max_length or check JSON.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> AlpacaSFTExample:
        return self.examples[i]


def collate_sft_batch(examples: List[AlpacaSFTExample], *, pad_id: int) -> dict[str, torch.Tensor]:
    max_t = max(len(e.input_ids) for e in examples)
    B = len(examples)
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    labels = torch.full((B, max_t), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_t), dtype=torch.long)
    for i, e in enumerate(examples):
        t = len(e.input_ids)
        input_ids[i, :t] = e.input_ids
        labels[i, :t] = e.labels
        attention_mask[i, :t] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
