# Cortex

**Turn any corpus into a thinking brain — then read its mind for the non-obvious connections.**

Cortex embeds your notes, docs, or an AI-agent trace into a live 3D neural map:
each **concept becomes a neuron**, each **embedding similarity becomes a synapse**,
and a curiosity engine *thinks* through it in real time — attending, spreading
activation, and making dream-jumps across the graph. Every leap it makes is
captured as an **evidence-backed Insight Card** that rolls up into an exportable
**Insight Digest**. It runs **100% in your browser** — no backend, no upload, no CDN.

![Cortex — a curiosity engine thinking in real time](docs/cortex-demo.gif)

**▶ [Live demo](https://cmonaco0317.github.io/cortex-mind/)** — runs entirely in your browser; nothing is uploaded.

---

## Why it exists

Most knowledge graphs are decorative — a pretty picture you look at once. Cortex is
built around the **output**: the graph is context, the *digest of non-obvious
connections* is the product. A curiosity engine (inspired by spreading-activation
models of thought) wanders your concepts and surfaces pairs you wouldn't have put
together, ranked most-surprising-first, each with a "why" and an angle to pursue.

## Try it

```bash
cd frontend
npm ci          # exact, reproducible install (pinned lockfile)
npm run dev     # then open the printed localhost URL
```

The default view is a 237-concept demo brain (AI / neuroscience / cognition) — hit
**⚡ all** in the Insight Digest panel to surface every connection at once.

> Serve it over http (the `npm run dev` URL, or the built `dist/`). Opening the
> files directly with `file://` breaks ES-module loading.

## Bring your own data

Click **＋ your data**, then drop a folder, pick files, paste text, or hand it an
**AI-agent trace** (`.md`, `.txt`, `.json`, `.jsonl`). Cortex embeds it **locally**
and builds the brain in-browser.

![Build a brain from your own data](docs/cortex-byo.png)

- **Provably local.** The embedding model (`all-MiniLM-L6-v2`) and the ONNX-runtime
  WebAssembly are **vendored into the app** — open your browser's Network tab while
  it embeds and you'll see requests to *only your own origin*, never a CDN.
- **Secret-safe.** API keys, tokens, and private keys are auto-redacted on ingest,
  before anything becomes a neuron or a shareable card.
- **Fault-tolerant.** If a file won't parse, Cortex tells you exactly why instead of
  silently producing an empty brain.

## The Insight Digest — the artifact

Each dream-jump emits an Insight Card: `concept A × concept B · why it's non-obvious
· an angle to explore`, with the source snippets as evidence. Copy or download the
whole digest as Markdown, or export any card as a watermarked share image.

![An exported Insight Card](docs/cortex-share-example.png)

## How it works

| Layer | What it is |
|---|---|
| **Neurons** | your concepts, embedded with `all-MiniLM-L6-v2` (384-dim), laid out in 3D via PCA |
| **Synapses** | k-nearest-neighbour cosine similarity between concept embeddings |
| **Bridges** | long-range, moderately-similar pairs — the seeds of the insight leaps |
| **Curiosity engine** | live spreading-activation with an attend → spread → dream-jump policy; the firing is emergent, not scripted |
| **Insight Digest** | the ranked, evidence-backed export — the product |

## Tech

- **Local embeddings** — [`@xenova/transformers`](https://github.com/xenova/transformers.js) (all-MiniLM-L6-v2, quantized), running in WASM
- **Rendering** — `three.js` with UnrealBloom postprocessing
- **Build** — Vite + TypeScript; unit-tested with Vitest
- **Zero external runtime dependencies** — the model and wasm are committed under
  [`frontend/public/`](frontend/public); see [PROVENANCE](frontend/public/models/PROVENANCE.md)
  for exact files, sources, and licenses. Nothing is fetched at runtime.

## License

Cortex's own code is [MIT](LICENSE). Vendored components keep their own licenses:
the embedding model is Apache-2.0 and ONNX Runtime Web is MIT — details in
[PROVENANCE](frontend/public/models/PROVENANCE.md).

---

<sub>made with Cortex — a local curiosity engine. Nothing leaves your machine.</sub>
