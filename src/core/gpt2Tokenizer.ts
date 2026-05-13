import { decode, encode } from "gpt-tokenizer/encoding/r50k_base";

/** Same BPE family as OpenAI GPT-2 / tiktoken "gpt2" (r50k_base ranks). Pure JS — Vite-friendly. */
const encodeOpts = { allowedSpecial: "all" as const };

export class Gpt2TiktokenTokenizer {
  encode(text: string): number[] {
    return encode(text, encodeOpts);
  }

  decode(ids: number[]): string {
    return decode(ids);
  }

  free(): void {
    /* no native handles */
  }
}
