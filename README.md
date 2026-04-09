# nanochat-replica
In the era infused with AI, reading text is one of the best things computers can do. I am embarking on this project to learn and understand how the concepts behind LLMs work.

## HTML demos
- `public/Lab_01_Tokenization.html`
- `public/Lab_02_Embedding.html`
- `public/Lab_03_Transformer_Block.html`
- `public/legacy/GPT2_Replica_12Heads.html` (GPT‑2-style multi-head attention demo with **12 heads**)

## Shippable web app
This repo now includes a small Vite + TypeScript web app that reuses the Lab 1/2/3 ideas as interactive pages.

### Run locally
```bash
npm install
npm run dev
```

### Build for deployment
```bash
npm run build
npm run preview
```

### Deploy to GitHub Pages
- Push to `main` and the workflow in `.github/workflows/deploy-gh-pages.yml` will publish.
- If you change the repo name, update `base` in `vite.config.ts`.
