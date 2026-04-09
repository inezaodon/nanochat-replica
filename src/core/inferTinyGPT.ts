import { RegexBPETokenizer } from "./tokenizer";
import { WebManifest, viewF32 } from "./webModel";

type Tensors = Record<string, Float32Array>;

function gelu(x: number): number {
  // tanh approximation
  const c = 0.044715;
  const s = Math.sqrt(2 / Math.PI);
  const y = s * (x + c * x * x * x);
  return 0.5 * x * (1 + Math.tanh(y));
}

function layerNorm(x: Float32Array, eps = 1e-5): Float32Array {
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length || 1;
  let v = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mean;
    v += d * d;
  }
  v /= x.length || 1;
  const inv = 1 / Math.sqrt(v + eps);
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = (x[i] - mean) * inv;
  return y;
}

function softmaxInPlace(a: Float32Array) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const e = Math.exp(a[i] - m);
    a[i] = e;
    s += e;
  }
  const inv = 1 / (s || 1);
  for (let i = 0; i < a.length; i++) a[i] *= inv;
}

function matmulVec(out: Float32Array, W: Float32Array, inDim: number, x: Float32Array) {
  // W: [outDim, inDim] row-major, out.length = outDim
  for (let i = 0; i < out.length; i++) {
    let s = 0;
    const base = i * inDim;
    for (let j = 0; j < inDim; j++) s += W[base + j] * x[j];
    out[i] = s;
  }
}

function addInPlace(a: Float32Array, b: Float32Array) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

function takeRow(mat: Float32Array, cols: number, row: number): Float32Array {
  const out = new Float32Array(cols);
  out.set(mat.subarray(row * cols, (row + 1) * cols));
  return out;
}

function setRow(mat: Float32Array, cols: number, row: number, x: Float32Array) {
  mat.set(x, row * cols);
}

function sampleFromLogits(logits: Float32Array, temperature: number, topK: number, rng: () => number): number {
  const scaled = new Float32Array(logits.length);
  const invT = 1 / Math.max(1e-6, temperature);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] * invT;

  // top-k filter
  let cutoff = -Infinity;
  if (topK > 0 && topK < scaled.length) {
    const arr = Array.from(scaled);
    arr.sort((a, b) => b - a);
    cutoff = arr[topK - 1];
  }
  for (let i = 0; i < scaled.length; i++) {
    if (scaled[i] < cutoff) scaled[i] = -1e9;
  }

  softmaxInPlace(scaled);
  const r = rng();
  let cum = 0;
  for (let i = 0; i < scaled.length; i++) {
    cum += scaled[i];
    if (r <= cum) return i;
  }
  return scaled.length - 1;
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

export type TinyGPTWeb = {
  manifest: WebManifest;
  tokenizer: RegexBPETokenizer;
  tensors: Tensors;
  generate: (prompt: string, opts: { maxNewTokens: number; temperature: number; topK: number; seed: number }) => string;
};

export function loadTensors(weightsBuf: ArrayBuffer, manifest: WebManifest): Tensors {
  const t: Tensors = {};
  for (const [name, info] of Object.entries(manifest.tensors)) {
    t[name] = viewF32(weightsBuf, info.offset, info.nbytes);
  }
  return t;
}

export function createTinyGPTWeb(manifest: WebManifest, tokenizer: RegexBPETokenizer, tensors: Tensors): TinyGPTWeb {
  const cfg = manifest.config;
  const { n_layer, n_head, n_embd, block_size, vocab_size } = cfg;
  const headDim = n_embd / n_head;

  function forward(ids: number[]): Float32Array {
    const T = Math.min(ids.length, block_size);
    const X = new Float32Array(T * n_embd);

    // embeddings
    const tokW = tensors["tok_emb.weight"]; // [vocab, n_embd]
    const posW = tensors["pos_emb.weight"]; // [block, n_embd]
    for (let t = 0; t < T; t++) {
      const id = ids[t] % vocab_size;
      const tokOff = id * n_embd;
      const posOff = t * n_embd;
      const dst = t * n_embd;
      for (let i = 0; i < n_embd; i++) X[dst + i] = tokW[tokOff + i] + posW[posOff + i];
    }

    // blocks
    for (let l = 0; l < n_layer; l++) {
      // LN1
      const ln1w = tensors[`blocks.${l}.ln1.weight`];
      const ln1b = tensors[`blocks.${l}.ln1.bias`];
      const Xn1 = new Float32Array(T * n_embd);
      for (let t = 0; t < T; t++) {
        const row = takeRow(X, n_embd, t);
        const y = layerNorm(row);
        for (let i = 0; i < n_embd; i++) y[i] = y[i] * ln1w[i] + ln1b[i];
        setRow(Xn1, n_embd, t, y);
      }

      // attn: qkv + causal attention + proj
      const qkvW = tensors[`blocks.${l}.attn.qkv.weight`]; // [3*n_embd, n_embd]
      const projW = tensors[`blocks.${l}.attn.proj.weight`]; // [n_embd, n_embd]

      const Q = new Float32Array(T * n_embd);
      const K = new Float32Array(T * n_embd);
      const V = new Float32Array(T * n_embd);
      const tmp = new Float32Array(3 * n_embd);
      for (let t = 0; t < T; t++) {
        const x = takeRow(Xn1, n_embd, t);
        matmulVec(tmp, qkvW, n_embd, x);
        Q.set(tmp.subarray(0, n_embd), t * n_embd);
        K.set(tmp.subarray(n_embd, 2 * n_embd), t * n_embd);
        V.set(tmp.subarray(2 * n_embd, 3 * n_embd), t * n_embd);
      }

      const attOut = new Float32Array(T * n_embd);
      const scores = new Float32Array(T);
      for (let h = 0; h < n_head; h++) {
        for (let i = 0; i < T; i++) {
          // scores[j] = q_i dot k_j / sqrt(headDim), with causal mask
          for (let j = 0; j < T; j++) {
            if (j > i) {
              scores[j] = -1e9;
              continue;
            }
            let s = 0;
            const qi = i * n_embd + h * headDim;
            const kj = j * n_embd + h * headDim;
            for (let d = 0; d < headDim; d++) s += Q[qi + d] * K[kj + d];
            scores[j] = s / Math.sqrt(headDim);
          }
          softmaxInPlace(scores);

          // weighted sum of V
          const outOff = i * n_embd + h * headDim;
          for (let d = 0; d < headDim; d++) {
            let s = 0;
            for (let j = 0; j < T; j++) {
              const vj = j * n_embd + h * headDim + d;
              s += scores[j] * V[vj];
            }
            attOut[outOff + d] = s;
          }
        }
      }

      // proj
      const projOut = new Float32Array(T * n_embd);
      const y = new Float32Array(n_embd);
      for (let t = 0; t < T; t++) {
        const x = takeRow(attOut, n_embd, t);
        matmulVec(y, projW, n_embd, x);
        setRow(projOut, n_embd, t, y);
      }

      // residual 1: X = X + projOut
      addInPlace(X, projOut);

      // LN2
      const ln2w = tensors[`blocks.${l}.ln2.weight`];
      const ln2b = tensors[`blocks.${l}.ln2.bias`];
      const Xn2 = new Float32Array(T * n_embd);
      for (let t = 0; t < T; t++) {
        const row = takeRow(X, n_embd, t);
        const z = layerNorm(row);
        for (let i = 0; i < n_embd; i++) z[i] = z[i] * ln2w[i] + ln2b[i];
        setRow(Xn2, n_embd, t, z);
      }

      // MLP: fc -> gelu -> proj
      const fcW = tensors[`blocks.${l}.mlp.fc.weight`]; // [4n, n]
      const prW = tensors[`blocks.${l}.mlp.proj.weight`]; // [n, 4n]
      const hid = new Float32Array(4 * n_embd);
      const out = new Float32Array(n_embd);
      const mlpOut = new Float32Array(T * n_embd);
      for (let t = 0; t < T; t++) {
        const x = takeRow(Xn2, n_embd, t);
        matmulVec(hid, fcW, n_embd, x);
        for (let i = 0; i < hid.length; i++) hid[i] = gelu(hid[i]);
        matmulVec(out, prW, 4 * n_embd, hid);
        setRow(mlpOut, n_embd, t, out);
      }

      // residual 2
      addInPlace(X, mlpOut);
    }

    // final ln + head
    const lnfw = tensors["ln_f.weight"];
    const lnfb = tensors["ln_f.bias"];
    const last = takeRow(X, n_embd, Math.max(0, Math.min(T - 1, T - 1)));
    const z = layerNorm(last);
    for (let i = 0; i < n_embd; i++) z[i] = z[i] * lnfw[i] + lnfb[i];

    const headW = tensors["lm_head.weight"]; // [vocab, n_embd]
    const logits = new Float32Array(vocab_size);
    for (let v = 0; v < vocab_size; v++) {
      let s = 0;
      const base = v * n_embd;
      for (let i = 0; i < n_embd; i++) s += headW[base + i] * z[i];
      logits[v] = s;
    }
    return logits;
  }

  function generate(prompt: string, opts: { maxNewTokens: number; temperature: number; topK: number; seed: number }): string {
    const rng = mulberry32(opts.seed);
    let ids = tokenizer.encode(prompt);
    for (let i = 0; i < opts.maxNewTokens; i++) {
      const ctx = ids.slice(Math.max(0, ids.length - block_size));
      const logits = forward(ctx);
      const next = sampleFromLogits(logits, opts.temperature, opts.topK, rng);
      ids = ids.concat([next]);
      // stop if eos
      const eos = tokenizer.specialTokens.get("<|eos|>");
      if (eos !== undefined && next === eos) break;
    }
    return tokenizer.decode(ids);
  }

  return { manifest, tokenizer, tensors, generate };
}

