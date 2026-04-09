export type MergePair = `${number},${number}`;

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class RegexBPETokenizer {
  // API kept stable, but implementation is now pure character-level tokenization.
  merges: Map<MergePair, number>;
  vocab: Map<number, string>;
  specialTokens: Map<string, number>;
  private charToId: Map<string, number>;
  private inverseSpecialTokens: Map<number, string>;

  constructor() {
    this.merges = new Map();
    this.vocab = new Map();
    this.specialTokens = new Map([
      ["<|sos|>", 0],
      ["<|eos|>", 1],
      ["<|unk|>", 2],
    ]);
    this.inverseSpecialTokens = new Map();
    this.charToId = new Map();

    for (const [token, id] of this.specialTokens.entries()) {
      this.vocab.set(id, token);
      this.inverseSpecialTokens.set(id, token);
    }
  }

  private splitBySpecialTokens(text: string): string[] {
    const specials = Array.from(this.specialTokens.keys());
    if (specials.length === 0) return [text];
    const re = new RegExp(`(${specials.map((s) => escapeRegex(s)).join("|")})`, "g");
    return text.split(re).filter((p) => p.length > 0);
  }

  train(text: string, maxVocabSize: number): void {
    // Char-level: no merges; just build a char vocabulary.
    this.merges.clear();
    this.charToId.clear();
    this.vocab = new Map();
    for (const [token, id] of this.specialTokens.entries()) this.vocab.set(id, token);

    const counts = new Map<string, number>();
    for (const part of this.splitBySpecialTokens(text)) {
      if (this.specialTokens.has(part)) continue;
      for (const ch of Array.from(part)) {
        counts.set(ch, (counts.get(ch) ?? 0) + 1);
      }
    }

    const maxChars = Math.max(0, maxVocabSize - this.specialTokens.size);
    const chars = Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxChars)
      .map(([ch]) => ch);

    let nextId = this.specialTokens.size;
    for (const ch of chars) {
      this.charToId.set(ch, nextId);
      this.vocab.set(nextId, ch);
      nextId += 1;
    }
  }

  encode(text: string): number[] {
    const out: number[] = [];
    const unk = this.specialTokens.get("<|unk|>") ?? 2;
    for (const part of this.splitBySpecialTokens(text)) {
      const sid = this.specialTokens.get(part);
      if (sid !== undefined) {
        out.push(sid);
        continue;
      }
      for (const ch of Array.from(part)) out.push(this.charToId.get(ch) ?? unk);
    }
    return out;
  }

  decode(ids: number[]): string {
    const unk = this.specialTokens.get("<|unk|>") ?? 2;
    return ids.map((id) => this.vocab.get(id) ?? this.vocab.get(unk) ?? "<|unk|>").join("");
  }

  toJSON(): {
    tokenizer_type: "character";
    merges: Record<string, number>;
    vocab: Record<string, string>;
    special_tokens: Record<string, number>;
  } {
    const vocab: Record<string, string> = {};
    for (const [id, value] of this.vocab.entries()) vocab[String(id)] = value;
    const special_tokens: Record<string, number> = {};
    for (const [k, v] of this.specialTokens.entries()) special_tokens[k] = v;
    return { tokenizer_type: "character", merges: {}, vocab, special_tokens };
  }

  static fromJSON(obj: {
    merges?: Record<string, number>;
    vocab: Record<string, string>;
    special_tokens: Record<string, number>;
  }): RegexBPETokenizer {
    const t = new RegexBPETokenizer();
    t.merges = new Map(Object.entries(obj.merges ?? {}) as Array<[MergePair, number]>);
    t.specialTokens = new Map(Object.entries(obj.special_tokens));
    t.inverseSpecialTokens = new Map();
    for (const [k, v] of t.specialTokens.entries()) t.inverseSpecialTokens.set(v, k);

    t.vocab = new Map();
    t.charToId = new Map();
    for (const [idStr, s] of Object.entries(obj.vocab)) {
      const id = Number(idStr);
      t.vocab.set(id, s);
      if (!t.specialTokens.has(s)) t.charToId.set(s, id);
    }
    return t;
  }
}

