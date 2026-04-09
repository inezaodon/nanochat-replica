import json
import re
from collections import Counter
from typing import Dict, List


class RegexBPETokenizer:
    """
    Character-level tokenizer (no BPE merges).
    Kept under the same class name for compatibility with existing imports.
    """

    def __init__(self):
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+"
        )
        self.merges: Dict[str, int] = {}
        self.vocab: Dict[int, str] = {}
        self.special_tokens: Dict[str, int] = {
            "<|sos|>": 0,
            "<|eos|>": 1,
            "<|unk|>": 2,
        }
        self.char_to_id: Dict[str, int] = {}
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token

    def train(self, text: str, max_vocab_size: int, verbose: bool = False) -> None:
        self.merges = {}
        self.vocab = {idx: tok for tok, idx in self.special_tokens.items()}
        self.char_to_id = {}

        counts: Counter = Counter()
        for part in self._split_with_specials(text):
            if part in self.special_tokens:
                continue
            for ch in part:
                counts[ch] += 1

        max_chars = max(0, max_vocab_size - len(self.special_tokens))
        most_common = [ch for ch, _ in counts.most_common(max_chars)]

        next_id = len(self.special_tokens)
        for ch in most_common:
            self.char_to_id[ch] = next_id
            self.vocab[next_id] = ch
            next_id += 1

        if verbose:
            print(f"built character vocab with {len(self.char_to_id)} chars")

    def _split_with_specials(self, chunk: str) -> List[str]:
        specials = list(self.special_tokens.keys())
        if not specials:
            return [chunk]
        special_re = "(" + "|".join(re.escape(s) for s in specials) + ")"
        return [p for p in re.split(special_re, chunk) if p]

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        unk = self.special_tokens["<|unk|>"]
        for part in self._split_with_specials(text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue
            for c in part:
                out.append(self.char_to_id.get(c, unk))
        return out

    def decode(self, ids: List[int]) -> str:
        unk = self.special_tokens["<|unk|>"]
        return "".join(self.vocab.get(i, self.vocab[unk]) for i in ids)

    def export_json(self) -> dict:
        return {
            "tokenizer_type": "character",
            "merges": self.merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }

    @staticmethod
    def from_export(obj: dict) -> "RegexBPETokenizer":
        t = RegexBPETokenizer()
        t.merges = {str(k): int(v) for k, v in obj.get("merges", {}).items()}
        t.special_tokens = {str(k): int(v) for k, v in obj["special_tokens"].items()}
        t.vocab = {int(k): str(v) for k, v in obj["vocab"].items()}
        t.char_to_id = {}
        special_set = set(t.special_tokens.keys())
        for idx, token in t.vocab.items():
            if token not in special_set:
                t.char_to_id[token] = idx
        return t

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_json(), f)

    @staticmethod
    def load(path: str) -> "RegexBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return RegexBPETokenizer.from_export(obj)

