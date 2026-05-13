/**
 * Curated study notes from course practice answer keys (CSE 10124 style).
 * Source PDFs are optional locally; the app ships this static list for deploy.
 */
export type ExamFactSource = "exam01" | "practice02";

export interface ExamFact {
  id: string;
  /** Short label shown in the UI */
  sourceLabel: string;
  source: ExamFactSource;
  text: string;
}

export const EXAM_FACTS: ExamFact[] = [
  {
    id: "e01-next-token",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Large language models often generate text autoregressively: at each step they predict the next token given all tokens generated so far, then append that token and repeat.",
  },
  {
    id: "e01-train-vs-infer",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Training updates weights on large datasets to reduce prediction error; inference freezes weights and only runs the forward pass to produce outputs for new inputs.",
  },
  {
    id: "e01-token-vs-word",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "A token is whatever the tokenizer emits—a whole word, a subword, or a character chunk. Natural words and model tokens need not line up one-to-one.",
  },
  {
    id: "e01-oov-bpe",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Word-level vocabularies hit out-of-vocabulary issues on unseen words. Subword tokenization (for example BPE) expresses rare words as sequences of known fragments instead.",
  },
  {
    id: "e01-bpe-merge",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Byte Pair Encoding starts from a small symbol set (often bytes or characters), then repeatedly merges the most frequent adjacent pair in the training corpus to build larger units.",
  },
  {
    id: "e01-bpe-bytes",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Starting BPE from bytes means any UTF-8 string can be represented as tokens, so there is always a fallback path even when a rare word never appeared whole in training data.",
  },
  {
    id: "e01-embeddings",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Token embeddings are learned dense vectors indexed by token ID. They replace impractically huge sparse one-hot rows and let the model place related tokens nearer in vector space.",
  },
  {
    id: "e01-cross-entropy",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Cross-entropy for language modeling rewards assigning probability mass to the true next token. Very confident wrong predictions are penalized much more than mild uncertainty.",
  },
  {
    id: "e01-linear-stack",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Stacking linear layers without a nonlinearity collapses to a single linear map. Nonlinear activations between layers are what let depth express richer functions.",
  },
  {
    id: "e01-softmax",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "The language-model head emits logits—unnormalized scores per vocabulary entry. Softmax turns logits into a valid probability distribution over next tokens.",
  },
  {
    id: "e01-self-attn",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Self-attention lets each position build a weighted mix of every other position’s representations. Learned queries, keys, and values determine those mixing weights.",
  },
  {
    id: "e01-mha",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Multi-head attention runs several attention operations in parallel at reduced per-head width, concatenates them, and projects back—different heads can specialize different relational patterns.",
  },
  {
    id: "e01-causal",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "Decoder-only models use causal masking so a position cannot attend to future tokens. That matches left-to-right generation and prevents peeking at the target being predicted.",
  },
  {
    id: "e01-block",
    source: "exam01",
    sourceLabel: "Exam 01 practice key",
    text: "A standard transformer block chains multi-head attention, residual add and layer norm, a position-wise feed-forward network, then another residual add and norm—repeated for depth.",
  },
  {
    id: "p02-post-train",
    source: "practice02",
    sourceLabel: "Practice packet 02 key",
    text: "Post-training continues after large-scale self-supervised pre-training: smaller curated datasets and different objectives steer behavior—format, safety, and assistant style—without relearning language from scratch.",
  },
  {
    id: "p02-base-chat",
    source: "practice02",
    sourceLabel: "Practice packet 02 key",
    text: "A base model is optimized for next-token continuation on broad text, so a bare question may continue as prose or another question. Chat models add post-training so answers match conversational intent.",
  },
  {
    id: "p02-pipeline",
    source: "practice02",
    sourceLabel: "Practice packet 02 key",
    text: "A common modern stack is supervised fine-tuning on instructions, training a reward model from human preference comparisons, then reinforcement learning from human feedback (for example PPO) or direct preference optimization.",
  },
  {
    id: "p02-sft-mask",
    source: "practice02",
    sourceLabel: "Practice packet 02 key",
    text: "Instruction tuning usually applies the language-modeling loss only on assistant response tokens, masking the prompt. That teaches completions without wasting gradients on rewriting the user message.",
  },
];
