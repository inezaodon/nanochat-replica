import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _pair_key(a: int, b: int) -> str:
    return f"{a},{b}"


def _bytes_to_str(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def _str_to_bytes(s: str) -> bytes:
    return s.encode("utf-8")


@dataclass
class TokenizerExport:
    # JSON-serializable export consumed by the web app.
    merges: Dict[str, int]  # "a,b" -> new_id
    vocab: Dict[str, str]   # id -> base64? (we store latin-1 string for bytes)
    special_tokens: Dict[str, int]


class RegexBPETokenizer:
    """
    BPE on UTF-8 bytes with a regex pre-split (Lab 01 concept),
    exported in a format that can be loaded in JS/TS.
    """

    def __init__(self):
        self.merges: Dict[str, int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {
            "<|sos|>": 256,
            "<|eos|>": 257,
            "<|unk|>": 258,
        }
        for k, v in self.special_tokens.items():
            self.vocab[v] = _str_to_bytes(k)

        # Browser-friendly regex close to the TS version.
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+"
        )

    def _get_stats(self, ids: List[int]) -> Counter:
        c = Counter()
        for a, b in zip(ids, ids[1:]):
            c[(a, b)] += 1
        return c

    def _merge_pairs(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        a, b = pair
        out: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, max_vocab_size: int, verbose: bool = False) -> None:
        pre = len(self.vocab)
        num_merges = max(0, max_vocab_size - pre)
        ids = list(_str_to_bytes(text))

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            (a, b), count = stats.most_common(1)[0]
            new_id = len(self.vocab)
            self.merges[_pair_key(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            ids = self._merge_pairs(ids, (a, b), new_id)
            if verbose and (i + 1) % 500 == 0:
                print(f"merge {i+1}/{num_merges}: ({a},{b}) -> {new_id} (count={count})")

    def _encode_chunk_bpe(self, chunk: str) -> List[int]:
        # Supports literal special tokens inside a chunk.
        parts = []
        specials = list(self.special_tokens.keys())
        if specials:
            special_re = "(" + "|".join(re.escape(s) for s in specials) + ")"
            parts = [p for p in re.split(special_re, chunk) if p]
        else:
            parts = [chunk]

        out: List[int] = []
        for part in parts:
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue

            ids = list(_str_to_bytes(part))
            while len(ids) >= 2:
                stats = self._get_stats(ids)
                # apply earliest merge id (lowest new_id) that exists
                best_pair = None
                best_merge = math.inf
                for (a, b) in stats.keys():
                    k = _pair_key(a, b)
                    mid = self.merges.get(k)
                    if mid is not None and mid < best_merge:
                        best_merge = mid
                        best_pair = (a, b)
                if best_pair is None:
                    break
                ids = self._merge_pairs(ids, best_pair, int(best_merge))
            out.extend(ids)
        return out

    def encode(self, text: str) -> List[int]:
        chunks = self.pattern.findall(text) if text else []
        out: List[int] = []
        for ch in chunks:
            out.extend(self._encode_chunk_bpe(ch))
        return out

    def decode(self, ids: List[int]) -> str:
        b = b"".join(self.vocab.get(i, self.vocab[self.special_tokens["<|unk|>"]]) for i in ids)
        return _bytes_to_str(b)

    def export_json(self) -> dict:
        # Store bytes using latin-1 so roundtrips 0-255 values without base64.
        vocab_out = {str(k): v.decode("latin-1") for k, v in self.vocab.items()}
        return {
            "merges": self.merges,
            "vocab": vocab_out,
            "special_tokens": self.special_tokens,
            "pattern": self.pattern.pattern,
        }

    @staticmethod
    def from_export(obj: dict) -> "RegexBPETokenizer":
        t = RegexBPETokenizer()
        t.merges = {str(k): int(v) for k, v in obj["merges"].items()}
        t.vocab = {int(k): v.encode("latin-1") for k, v in obj["vocab"].items()}
        t.special_tokens = {str(k): int(v) for k, v in obj["special_tokens"].items()}
        t.pattern = re.compile(obj.get("pattern") or t.pattern.pattern)
        return t

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export_json(), f)

    @staticmethod
    def load(path: str) -> "RegexBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return RegexBPETokenizer.from_export(obj)

