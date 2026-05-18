import { WebManifest, viewF32 } from "./webModel";

/** Minimal tokenizer surface for generation (character BPE or GPT-2 tiktoken). */
export type TextTokenizer = {
  encode(text: string): number[];
  decode(ids: number[]): string;
  specialTokens?: Map<string, number>;
};

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

/** Min-heap (size n) over heap[0..n); smallest value at root. */
function heapSiftDown(heap: Float32Array, i: number, n: number) {
  while (true) {
    let m = i;
    const l = i * 2 + 1;
    const r = i * 2 + 2;
    if (l < n && heap[l] < heap[m]) m = l;
    if (r < n && heap[r] < heap[m]) m = r;
    if (m === i) break;
    const t = heap[i];
    heap[i] = heap[m];
    heap[m] = t;
    i = m;
  }
}

function heapBuildMin(heap: Float32Array, n: number) {
  for (let i = Math.floor(n / 2) - 1; i >= 0; i--) heapSiftDown(heap, i, n);
}

/**
 * k-th largest value in arr (1-indexed k), O(n log k) — avoids full sort on vocab-sized arrays.
 * Uses a min-heap of the k largest elements seen so far.
 */
function kthLargest(arr: Float32Array, k: number): number {
  if (k <= 0 || k > arr.length) return -Infinity;
  const heap = new Float32Array(k);
  for (let i = 0; i < k; i++) heap[i] = arr[i];
  heapBuildMin(heap, k);
  for (let i = k; i < arr.length; i++) {
    const x = arr[i];
    if (x > heap[0]) {
      heap[0] = x;
      heapSiftDown(heap, 0, k);
    }
  }
  return heap[0];
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

function addRowInPlace(mat: Float32Array, cols: number, row: number, delta: Float32Array) {
  const off = row * cols;
  for (let i = 0; i < cols; i++) mat[off + i] += delta[i];
}

function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
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
  const invT = 1 / Math.max(1e-6, temperature);
  for (let i = 0; i < logits.length; i++) logits[i] *= invT;

  // top-k filter (heap k-th largest; full vocab sort was dominant cost for GPT-2)
  let cutoff = -Infinity;
  if (topK > 0 && topK < logits.length) {
    cutoff = kthLargest(logits, topK);
  }
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] < cutoff) logits[i] = -1e9;
  }

  softmaxInPlace(logits);
  const r = rng();
  let cum = 0;
  for (let i = 0; i < logits.length; i++) {
    cum += logits[i];
    if (r <= cum) return i;
  }
  return logits.length - 1;
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

export type GenerateOpts = {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  seed: number;
};

export type GenerateCallbacks = {
  onProgress?: (step: number, max: number) => void;
  signal?: AbortSignal;
};

export type TinyGPTWeb = {
  manifest: WebManifest;
  tokenizer: TextTokenizer;
  tensors: Tensors;
  generate: (prompt: string, opts: GenerateOpts) => string;
  generateAsync: (prompt: string, opts: GenerateOpts, callbacks?: GenerateCallbacks) => Promise<string>;
};

export function loadTensors(weightsBuf: ArrayBuffer, manifest: WebManifest): Tensors {
  const t: Tensors = {};
  for (const [name, info] of Object.entries(manifest.tensors)) {
    t[name] = viewF32(weightsBuf, info.offset, info.nbytes);
  }
  return t;
}

export function createTinyGPTWeb(manifest: WebManifest, tokenizer: TextTokenizer, tensors: Tensors): TinyGPTWeb {
  const cfg = manifest.config;
  const { n_layer, n_head, n_embd, block_size, vocab_size } = cfg;
  const headDim = n_embd / n_head;

  /** When `onlyLast`, compute logits for the final context position only (autoregressive decode). */
  function forward(ids: number[], onlyLast = false): Float32Array {
    const T = Math.min(ids.length, block_size);
    const lastIdx = T - 1;
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
      const qkvB = tensors[`blocks.${l}.attn.qkv.bias`];
      const projW = tensors[`blocks.${l}.attn.proj.weight`]; // [n_embd, n_embd]
      const projB = tensors[`blocks.${l}.attn.proj.bias`];

      const Q = new Float32Array(T * n_embd);
      const K = new Float32Array(T * n_embd);
      const V = new Float32Array(T * n_embd);
      const tmp = new Float32Array(3 * n_embd);
      for (let t = 0; t < T; t++) {
        const x = takeRow(Xn1, n_embd, t);
        matmulVec(tmp, qkvW, n_embd, x);
        if (qkvB) {
          for (let i = 0; i < tmp.length; i++) tmp[i] += qkvB[i];
        }
        Q.set(tmp.subarray(0, n_embd), t * n_embd);
        K.set(tmp.subarray(n_embd, 2 * n_embd), t * n_embd);
        V.set(tmp.subarray(2 * n_embd, 3 * n_embd), t * n_embd);
      }

      const attOut = new Float32Array(T * n_embd);
      const scores = new Float32Array(T);
      const attnStart = onlyLast ? lastIdx : 0;
      const attnEnd = onlyLast ? lastIdx + 1 : T;
      for (let h = 0; h < n_head; h++) {
        for (let i = attnStart; i < attnEnd; i++) {
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

      const y = new Float32Array(n_embd);
      if (onlyLast) {
        const x = takeRow(attOut, n_embd, lastIdx);
        matmulVec(y, projW, n_embd, x);
        if (projB) {
          for (let i = 0; i < n_embd; i++) y[i] += projB[i];
        }
        addRowInPlace(X, n_embd, lastIdx, y);
      } else {
        const projOut = new Float32Array(T * n_embd);
        for (let t = 0; t < T; t++) {
          const x = takeRow(attOut, n_embd, t);
          matmulVec(y, projW, n_embd, x);
          if (projB) {
            for (let i = 0; i < n_embd; i++) y[i] += projB[i];
          }
          setRow(projOut, n_embd, t, y);
        }
        addInPlace(X, projOut);
      }

      const ln2w = tensors[`blocks.${l}.ln2.weight`];
      const ln2b = tensors[`blocks.${l}.ln2.bias`];
      const fcW = tensors[`blocks.${l}.mlp.fc.weight`];
      const fcB = tensors[`blocks.${l}.mlp.fc.bias`];
      const prW = tensors[`blocks.${l}.mlp.proj.weight`];
      const prB = tensors[`blocks.${l}.mlp.proj.bias`];
      const hid = new Float32Array(4 * n_embd);
      const out = new Float32Array(n_embd);

      if (onlyLast) {
        const row = takeRow(X, n_embd, lastIdx);
        const z = layerNorm(row);
        for (let i = 0; i < n_embd; i++) z[i] = z[i] * ln2w[i] + ln2b[i];
        matmulVec(hid, fcW, n_embd, z);
        if (fcB) {
          for (let i = 0; i < hid.length; i++) hid[i] += fcB[i];
        }
        for (let i = 0; i < hid.length; i++) hid[i] = gelu(hid[i]);
        matmulVec(out, prW, 4 * n_embd, hid);
        if (prB) {
          for (let i = 0; i < n_embd; i++) out[i] += prB[i];
        }
        addRowInPlace(X, n_embd, lastIdx, out);
      } else {
        const Xn2 = new Float32Array(T * n_embd);
        for (let t = 0; t < T; t++) {
          const row = takeRow(X, n_embd, t);
          const z = layerNorm(row);
          for (let i = 0; i < n_embd; i++) z[i] = z[i] * ln2w[i] + ln2b[i];
          setRow(Xn2, n_embd, t, z);
        }
        const mlpOut = new Float32Array(T * n_embd);
        for (let t = 0; t < T; t++) {
          const x = takeRow(Xn2, n_embd, t);
          matmulVec(hid, fcW, n_embd, x);
          if (fcB) {
            for (let i = 0; i < hid.length; i++) hid[i] += fcB[i];
          }
          for (let i = 0; i < hid.length; i++) hid[i] = gelu(hid[i]);
          matmulVec(out, prW, 4 * n_embd, hid);
          if (prB) {
            for (let i = 0; i < n_embd; i++) out[i] += prB[i];
          }
          setRow(mlpOut, n_embd, t, out);
        }
        addInPlace(X, mlpOut);
      }
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

  function decodeStep(ids: number[], rng: () => number, opts: GenerateOpts): number {
    const from = Math.max(0, ids.length - block_size);
    const ctx = from === 0 ? ids : ids.slice(from);
    const logits = forward(ctx, true);
    return sampleFromLogits(logits, opts.temperature, opts.topK, rng);
  }

  function generate(prompt: string, opts: GenerateOpts): string {
    const rng = mulberry32(opts.seed);
    const ids = tokenizer.encode(prompt);
    const manifestEos = manifest.eos_token_id;
    const charTokEos = tokenizer.specialTokens?.get("<|eos|>");
    for (let i = 0; i < opts.maxNewTokens; i++) {
      const next = decodeStep(ids, rng, opts);
      ids.push(next);
      if (manifestEos !== undefined && next === manifestEos) break;
      if (charTokEos !== undefined && next === charTokEos) break;
    }
    return tokenizer.decode(ids);
  }

  async function generateAsync(prompt: string, opts: GenerateOpts, callbacks?: GenerateCallbacks): Promise<string> {
    const rng = mulberry32(opts.seed);
    const ids = tokenizer.encode(prompt);
    const manifestEos = manifest.eos_token_id;
    const charTokEos = tokenizer.specialTokens?.get("<|eos|>");
    const max = opts.maxNewTokens;

    for (let i = 0; i < max; i++) {
      if (callbacks?.signal?.aborted) {
        throw new DOMException("Generation cancelled", "AbortError");
      }
      callbacks?.onProgress?.(i + 1, max);
      const next = decodeStep(ids, rng, opts);
      ids.push(next);
      if (manifestEos !== undefined && next === manifestEos) break;
      if (charTokEos !== undefined && next === charTokEos) break;
      await yieldToMain();
    }
    return tokenizer.decode(ids);
  }

  return { manifest, tokenizer, tensors, generate, generateAsync };
}

