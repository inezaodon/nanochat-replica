import re
from collections import Counter

class AllPurposeTokenizer:
    def __init__(self):
        # GPT-4 inspired regex pattern to handle whitespace, letters, numbers, and punctuation
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.merges = {}  # Stores pair:new_token mappings
        self.vocab = {}   # Stores token_id:byte_value mappings

    def get_stats(self, ids):
        """Counts the frequency of adjacent pairs in a list of IDs."""
        counts = Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge(self, ids, pair, idx):
        """Replaces all occurrences of a pair with a new merged token ID."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=False):
        """Trains the BPE tokenizer on a given corpus."""
        num_merges = vocab_size - 256
        # Initial tokenization: convert text to UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        ids = list(tokens)
        
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            # Find the most frequent pair
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"Merging {top_pair} into new token {idx}")
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx
            
    def encode(self, text):
        """Encodes raw text into token IDs."""
        # First, split the text using the regex pattern
        text_chunks = re.findall(self.pat, text)
        all_ids = []
        
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            # Iteratively apply learned merges
            while len(chunk_ids) >= 2:
                stats = self.get_stats(chunk_ids)
                # Find the merge that occurred earliest in training
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = self.merge(chunk_ids, pair, idx)
            all_ids.extend(chunk_ids)
        return all_ids

    def decode(self, ids):
        """Decodes token IDs back into a string."""
        # Convert IDs back to bytes based on initial 0-255 map + merges
        # Note: A full implementation would build a reverse vocab map during training
        tokens = b"".join([bytes([idx]) if idx < 256 else b"" for idx in ids])
        return tokens.decode("utf-8", errors="replace")