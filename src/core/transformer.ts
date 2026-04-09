// Ported from Lab 03 concepts: LayerNorm + scaled dot-product self-attention + causal masking.

export function layerNorm(x: Float32Array, eps = 1e-5): Float32Array {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= Math.max(1, x.length);
  let varSum = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mean;
    varSum += d * d;
  }
  const variance = varSum / Math.max(1, x.length);
  const inv = 1 / Math.sqrt(variance + eps);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = (x[i] - mean) * inv;
  return y;
}

export function softmaxRow(logits: Float32Array): Float32Array {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
  let s = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - m);
    exps[i] = e;
    s += e;
  }
  const out = new Float32Array(logits.length);
  const inv = 1 / (s || 1);
  for (let i = 0; i < logits.length; i++) out[i] = exps[i] * inv;
  return out;
}

export function dot(a: Float32Array, aOff: number, b: Float32Array, bOff: number, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}

export function matVecMul(W: Float32Array, outDim: number, inDim: number, x: Float32Array): Float32Array {
  // W row-major: [outDim][inDim]
  const y = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let s = 0;
    const base = i * inDim;
    for (let j = 0; j < inDim; j++) s += W[base + j] * x[j];
    y[i] = s;
  }
  return y;
}

export type RNG = () => number;

export function mulberry32(seed: number): RNG {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng: RNG): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function makeWeight(rng: RNG, outDim: number, inDim: number, scale = 0.02): Float32Array {
  const W = new Float32Array(outDim * inDim);
  for (let i = 0; i < W.length; i++) W[i] = randn(rng) * scale;
  return W;
}

export class SingleHeadAttention {
  readonly dModel: number;
  readonly dHead: number;
  readonly Wq: Float32Array;
  readonly Wk: Float32Array;
  readonly Wv: Float32Array;

  constructor(rng: RNG, dModel: number, dHead: number) {
    this.dModel = dModel;
    this.dHead = dHead;
    this.Wq = makeWeight(rng, dHead, dModel);
    this.Wk = makeWeight(rng, dHead, dModel);
    this.Wv = makeWeight(rng, dHead, dModel);
  }

  forward(X: Float32Array, T: number, causal: boolean): { O: Float32Array; A: Float32Array } {
    // X: (T, dModel) flattened
    const Q = new Float32Array(T * this.dHead);
    const K = new Float32Array(T * this.dHead);
    const V = new Float32Array(T * this.dHead);

    for (let t = 0; t < T; t++) {
      const x = X.subarray(t * this.dModel, (t + 1) * this.dModel);
      const xn = layerNorm(x);
      const q = matVecMul(this.Wq, this.dHead, this.dModel, xn);
      const k = matVecMul(this.Wk, this.dHead, this.dModel, xn);
      const v = matVecMul(this.Wv, this.dHead, this.dModel, xn);
      Q.set(q, t * this.dHead);
      K.set(k, t * this.dHead);
      V.set(v, t * this.dHead);
    }

    const A = new Float32Array(T * T); // attention weights
    const O = new Float32Array(T * this.dHead); // output
    const scale = 1 / Math.sqrt(this.dHead);

    for (let i = 0; i < T; i++) {
      const logits = new Float32Array(T);
      for (let j = 0; j < T; j++) {
        const masked = causal && j > i;
        logits[j] = masked ? -1e9 : dot(Q, i * this.dHead, K, j * this.dHead, this.dHead) * scale;
      }
      const row = softmaxRow(logits);
      for (let j = 0; j < T; j++) A[i * T + j] = row[j];

      // weighted sum
      for (let j = 0; j < T; j++) {
        const aij = row[j];
        const vOff = j * this.dHead;
        const oOff = i * this.dHead;
        for (let k = 0; k < this.dHead; k++) O[oOff + k] += aij * V[vOff + k];
      }
    }

    return { O, A };
  }
}

export class MultiHeadAttention {
  readonly dModel: number;
  readonly nHead: number;
  readonly dHead: number;
  readonly heads: SingleHeadAttention[];

  constructor(rng: RNG, dModel: number, nHead: number) {
    if (dModel % nHead !== 0) throw new Error("dModel must be divisible by nHead");
    this.dModel = dModel;
    this.nHead = nHead;
    this.dHead = dModel / nHead;
    this.heads = new Array(nHead);
    for (let h = 0; h < nHead; h++) this.heads[h] = new SingleHeadAttention(rng, dModel, this.dHead);
  }

  forward(X: Float32Array, T: number, causal: boolean): { O: Float32Array; AHeads: Float32Array[] } {
    // Concatenate head outputs: (T, dModel)
    const O = new Float32Array(T * this.dModel);
    const AHeads: Float32Array[] = [];
    for (let h = 0; h < this.nHead; h++) {
      const { O: Oh, A } = this.heads[h].forward(X, T, causal);
      AHeads.push(A);
      for (let t = 0; t < T; t++) {
        const dst = t * this.dModel + h * this.dHead;
        const src = t * this.dHead;
        O.set(Oh.subarray(src, src + this.dHead), dst);
      }
    }
    return { O, AHeads };
  }
}

export class TransformerBlock {
  readonly dModel: number;
  readonly mha: MultiHeadAttention;

  constructor(rng: RNG, dModel: number, nHead: number) {
    this.dModel = dModel;
    this.mha = new MultiHeadAttention(rng, dModel, nHead);
  }

  forward(X: Float32Array, T: number, causal: boolean): { Y: Float32Array; AHeads: Float32Array[] } {
    const { O, AHeads } = this.mha.forward(X, T, causal);
    // residual add: Y = X + O (Lab 03 idea)
    const Y = new Float32Array(X.length);
    for (let i = 0; i < X.length; i++) Y[i] = X[i] + O[i];
    return { Y, AHeads };
  }
}

