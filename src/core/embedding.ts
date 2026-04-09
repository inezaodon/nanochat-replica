// Ported from Lab 02 concepts: an Embedding layer is a lookup table W[tokens].

export class EmbeddingLayer {
  readonly vocabSize: number;
  readonly embedDim: number;
  readonly W: Float32Array;

  constructor(vocabSize: number, embedDim: number, seed = 0) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.W = new Float32Array(vocabSize * embedDim);

    // Deterministic init ~ N(0, 1/sqrt(vocabSize)) (matches lab-ish scaling)
    const rng = mulberry32(seed >>> 0);
    const std = 1 / Math.sqrt(Math.max(1, vocabSize));
    for (let i = 0; i < this.W.length; i++) this.W[i] = randn(rng) * std;
  }

  forward(tokens: number[]): Float32Array {
    const T = tokens.length;
    const out = new Float32Array(T * this.embedDim);
    for (let t = 0; t < T; t++) {
      const id = clampInt(tokens[t], 0, this.vocabSize - 1);
      const src = id * this.embedDim;
      const dst = t * this.embedDim;
      for (let i = 0; i < this.embedDim; i++) out[dst + i] = this.W[src + i];
    }
    return out;
  }
}

function clampInt(x: number, lo: number, hi: number): number {
  const n = Math.trunc(x);
  return Math.max(lo, Math.min(hi, n));
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  // Box-Muller
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

