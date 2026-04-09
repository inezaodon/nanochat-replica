export type TensorInfo = {
  shape: number[];
  offset: number;
  nbytes: number;
};

export type WebManifest = {
  format: "f32";
  weights: string;
  tensors: Record<string, TensorInfo>;
  config: {
    vocab_size: number;
    block_size: number;
    n_layer: number;
    n_head: number;
    n_embd: number;
    dropout: number;
  };
};

export async function fetchJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return (await r.json()) as T;
}

export async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.arrayBuffer();
}

export function viewF32(buf: ArrayBuffer, offsetBytes: number, nbytes: number): Float32Array {
  return new Float32Array(buf, offsetBytes, nbytes / 4);
}

