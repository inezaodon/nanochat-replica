// Ported from the Lab 01 tokenizer concepts (BPE + regex splitting).

export type MergePair = `${number},${number}`;

function pairKey(a: number, b: number): MergePair {
  return `${a},${b}`;
}

function bytesToString(bytes: Uint8Array): string {
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

function stringToBytes(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

export class BPETokenizer {
  // Simple BPE on bytes. Includes special tokens similar to the lab.
  merges: Map<MergePair, number>;
  vocab: Map<number, Uint8Array>;
  specialTokens: Map<string, number>;
  private inverseSpecialTokens: Map<number, string>;

  constructor() {
    this.merges = new Map();
    this.vocab = new Map();

    for (let i = 0; i < 256; i++) this.vocab.set(i, new Uint8Array([i]));

    // Match the Lab 01 convention: <|sos|> at 256, <|eos|> at 257
    this.specialTokens = new Map([
      ["<|sos|>", 256],
      ["<|eos|>", 257],
      ["<|unk|>", 258],
    ]);
    this.inverseSpecialTokens = new Map();
    for (const [k, v] of this.specialTokens.entries()) {
      this.vocab.set(v, stringToBytes(k));
      this.inverseSpecialTokens.set(v, k);
    }
  }

  private getStats(ids: number[]): Map<MergePair, number> {
    const counts = new Map<MergePair, number>();
    for (let i = 0; i < ids.length - 1; i++) {
      const k = pairKey(ids[i], ids[i + 1]);
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    return counts;
  }

  private mergePairs(ids: number[], pair: MergePair, newId: number): number[] {
    const [aStr, bStr] = pair.split(",");
    const a = Number(aStr);
    const b = Number(bStr);
    const out: number[] = [];
    let i = 0;
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
        out.push(newId);
        i += 2;
      } else {
        out.push(ids[i]);
        i += 1;
      }
    }
    return out;
  }

  train(text: string, maxVocabSize: number, verbose = false): void {
    const pretrainSize = this.vocab.size;
    const numMerges = Math.max(0, maxVocabSize - pretrainSize);
    let ids = Array.from(stringToBytes(text), (b) => b);

    for (let i = 0; i < numMerges; i++) {
      const stats = this.getStats(ids);
      if (stats.size === 0) break;

      // Find most frequent pair (Lab 01 simple tokenizer behavior)
      let bestPair: MergePair | null = null;
      let bestCount = -1;
      for (const [p, c] of stats.entries()) {
        if (c > bestCount) {
          bestPair = p;
          bestCount = c;
        }
      }
      if (!bestPair) break;

      const newId = this.vocab.size;
      this.merges.set(bestPair, newId);

      const [aStr, bStr] = bestPair.split(",");
      const a = Number(aStr);
      const b = Number(bStr);
      const aBytes = this.vocab.get(a);
      const bBytes = this.vocab.get(b);
      if (!aBytes || !bBytes) break;
      const merged = new Uint8Array(aBytes.length + bBytes.length);
      merged.set(aBytes, 0);
      merged.set(bBytes, aBytes.length);
      this.vocab.set(newId, merged);

      ids = this.mergePairs(ids, bestPair, newId);
      if (verbose && i % 250 === 0) {
        console.log(`merge ${i + 1}/${numMerges}: ${bestPair} -> ${newId} (count=${bestCount})`);
      }
    }
  }

  encode(text: string): number[] {
    // Supports special tokens literally present in the text.
    // Strategy: split on special tokens, encode others as bytes + BPE merges.
    if (text.length === 0) return [];

    const specials = Array.from(this.specialTokens.keys());
    const specialRegex = specials.length
      ? new RegExp(`(${specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g")
      : null;

    const parts = specialRegex ? text.split(specialRegex).filter((p) => p.length > 0) : [text];
    const out: number[] = [];

    for (const part of parts) {
      const sid = this.specialTokens.get(part);
      if (sid !== undefined) {
        out.push(sid);
        continue;
      }

      // Start from raw bytes
      let ids = Array.from(stringToBytes(part), (b) => b);

      // Iteratively apply the earliest merge available (Lab 01 behavior for encode)
      while (ids.length >= 2) {
        const stats = this.getStats(ids);
        let bestPair: MergePair | null = null;
        let bestMergeId = Infinity;

        for (const p of stats.keys()) {
          const mid = this.merges.get(p);
          if (mid !== undefined && mid < bestMergeId) {
            bestMergeId = mid;
            bestPair = p;
          }
        }

        if (!bestPair || bestMergeId === Infinity) break;
        ids = this.mergePairs(ids, bestPair, bestMergeId);
      }

      out.push(...ids);
    }

    return out;
  }

  decode(ids: number[]): string {
    // Reconstruct bytes from vocab. Unknown ids become <|unk|>.
    const parts: Uint8Array[] = [];
    for (const id of ids) {
      const b = this.vocab.get(id);
      if (b) {
        parts.push(b);
      } else {
        parts.push(stringToBytes(this.inverseSpecialTokens.get(this.specialTokens.get("<|unk|>")!) ?? "<|unk|>"));
      }
    }
    const total = parts.reduce((s, b) => s + b.length, 0);
    const joined = new Uint8Array(total);
    let off = 0;
    for (const b of parts) {
      joined.set(b, off);
      off += b.length;
    }
    return bytesToString(joined);
  }
}

export class RegexBPETokenizer {
  // A more “lab accurate” variant: split with a regex first, then BPE within each chunk.
  private bpe: BPETokenizer;
  private pattern: RegExp;

  constructor() {
    this.bpe = new BPETokenizer();
    // A pragmatic browser-safe regex that splits into (optional leading space + word/number) or punctuation.
    // (The notebook mentions a more GPT‑4-ish regex; JS doesn't support \p{L} on all environments without /u.)
    this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^ \t\r\nA-Za-z0-9]+|\s+(?!\S)|\s+/g;
  }

  get merges(): Map<MergePair, number> {
    return this.bpe.merges;
  }
  get vocab(): Map<number, Uint8Array> {
    return this.bpe.vocab;
  }
  get specialTokens(): Map<string, number> {
    return this.bpe.specialTokens;
  }

  train(text: string, maxVocabSize: number): void {
    // Train on raw bytes for simplicity (still consistent with Lab 01 idea).
    this.bpe.train(text, maxVocabSize, false);
  }

  encode(text: string): number[] {
    const chunks = text.match(this.pattern) ?? [];
    const out: number[] = [];
    for (const ch of chunks) out.push(...this.bpe.encode(ch));
    return out;
  }

  decode(ids: number[]): string {
    return this.bpe.decode(ids);
  }
}

