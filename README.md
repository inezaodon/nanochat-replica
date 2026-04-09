# nanochat-replica

Small GPT‑style language model you can train on Shakespeare and run **entirely in the browser**, built as a learning playground for tokenization, embeddings, and transformer blocks.

---

## Features

- **Tiny GPT**: 2‑layer GPT‑style decoder with 12 attention heads and positional embeddings.
- **Character tokenizer**: Character-level tokenization, shared between Python and TypeScript.
- **Browser inference**: Pure TypeScript forward pass and sampling – no server required.
- **React UI**: Responsive, single‑page interface with a “Small LLM” playground.
- **Course labs preserved**: Original Lab 01/02/03 HTML exports wired in as docs.

---

## Project layout

- `src/react/` – React app (`App`, `Home`, `LLMPlayground`, and docs page).
- `src/core/` – Tokenizer, tiny GPT inference, and web manifest helpers.
- `llm/` – PyTorch training + export code:
  - `train.py` – train a tiny GPT on `data/shakespeare.txt`
  - `model.py` – GPT config and modules (attention, MLP, blocks)
  - `tokenizer_bpe.py` – character-level tokenizer with JSON export
  - `export_web.py` – export weights + tokenizer for the browser
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
  --data data/shakespeare.txt \
  --device cpu \
  --out_dir checkpoints/tiny-gpt
```

This writes:

- `checkpoints/tiny-gpt/model.pt`
- `checkpoints/tiny-gpt/tokenizer.json`

Export the checkpoint to browser‑friendly weights:

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

