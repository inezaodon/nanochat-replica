# nanochat-replica

Small GPT‑style language model you can train on **your own text** (default: a multi-source corpus, not Shakespeare-only) and run **entirely in the browser**, built as a learning playground for tokenization, embeddings, and transformer blocks.

---

## Features

- **Tiny GPT**: small GPT‑style decoder you configure (`--n_layer`, etc.); default training text is `data/training_corpus.txt` (mixed public-domain + original snippets).
- **Character tokenizer**: Character-level tokenization, shared between Python and TypeScript.
- **Browser inference**: Pure TypeScript forward pass and sampling – no server required.
- **React UI**: Responsive, single‑page interface with a “Small LLM” playground.
- **Course labs preserved**: Original Lab 01/02/03 HTML exports wired in as docs.

---

## Project layout

- `src/react/` – React app (`App`, `Home`, `LLMPlayground`, and docs page).
- `src/core/` – Tokenizer, tiny GPT inference, and web manifest helpers.
- `llm/` – PyTorch training + export code:
  - `train.py` – train a tiny GPT on UTF-8 text (`--data` file or comma-separated list; default `data/training_corpus.txt`)
  - `model.py` – GPT config and modules (attention, MLP, blocks)
  - `tokenizer_bpe.py` – character-level tokenizer with JSON export
  - `expand_corpus.py` – merge Hugging Face / Gutenberg / local text into one training file
- `public/models/tiny-gpt/` – exported weights consumed by the web app.
- `public/Lab_*.html` – static notebook exports from the original course.

---

## Getting started

### 1. Run the web app

```bash
npm install
npm run dev
```

Then open the URL shown in the terminal (usually `http://localhost:5173`) and click **Small LLM**.

The repo already includes a small set of pre‑trained weights under `public/models/tiny-gpt/`, so you can:

1. Click **Load model**
2. Type a prompt
3. Click **Generate**

---

### 2. (Optional) Retrain the tiny GPT yourself

Create and activate a virtual environment, then install Python deps:

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Run a training job (CPU example):

```bash
python -m llm.train \
  --data data/training_corpus.txt \
  --device cpu \
  --out_dir checkpoints/tiny-gpt
```

This writes:

- `checkpoints/tiny-gpt/model.pt`
- `checkpoints/tiny-gpt/tokenizer.json`

You can pass **several files** (concatenated in order), e.g.  
`--data data/training_corpus.txt,data/course/jabberwocky.txt`  
after `python -m llm.fetch_course_datasets`, or point `--data` at `data/shakespeare.txt` alone if you want that baseline.

### 2b. Expand the corpus (Hugging Face, Gutenberg, local)

Install deps (`datasets` is listed in `requirements.txt`). Then for example:

```bash
python -m llm.expand_corpus \
  --out data/corpus_expanded.txt \
  --include-local data/training_corpus.txt \
  --hf-preset wikitext-103 ag_news imdb yelp tweet_sentiment \
  --gutenberg 11 1342 84 \
  --max-chars-per-preset 120000 \
  --max-chars-gutenberg 200000
```

Train on the merged file:

```bash
python -m llm.train --data data/corpus_expanded.txt --device cuda --out_dir checkpoints/tiny-gpt
```

**Kaggle:** install the [Kaggle API](https://github.com/Kaggle/kaggle-api), place `kaggle.json` under `~/.kaggle/`, run `kaggle datasets download …`, unzip, then add those files with repeated `--include-local path.csv`. **Licenses** differ by dataset; you are responsible for terms of use.

### 3. Export to the browser

```bash
python -m llm.export_web \
  --ckpt checkpoints/tiny-gpt/model.pt \
  --tokenizer checkpoints/tiny-gpt/tokenizer.json \
  --out_dir public/models/tiny-gpt
```

Now refresh the web app and use **Load model** again – it will pick up your newly trained weights.

---

## HTML labs (course material)

Static exports from the original labs are kept as references:

- `public/Lab_01_Tokenization.html`
- `public/Lab_02_Embedding.html`
- `public/Lab_03_Transformer_Block.html`
- `public/legacy/GPT2_Replica_12Heads.html` (GPT‑2 multi‑head attention visual demo)

You can open these directly (or via the **Course labs** tab in the UI) to connect the code to the teaching material.

---

## Deployment (GitHub Pages)

A workflow in `.github/workflows/deploy-gh-pages.yml` builds the Vite app and deploys it to GitHub Pages.

1. Ensure **Settings → Pages → Source** is set to **GitHub Actions**.
2. Push to `main`:

```bash
git push origin main
```

3. GitHub Actions will run **Deploy to GitHub Pages** and publish the site.

If you fork this repo or rename it, update the `base` option in `vite.config.ts` to match your new repository name.

