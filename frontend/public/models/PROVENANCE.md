# Vendored runtime — provenance & licenses

Cortex embeds your text **locally in the browser** with zero CDN calls. To make
that provable and to survive with no active maintainer, the embedding model and
the ONNX-runtime WebAssembly are committed into the repo rather than fetched at
runtime. Here is exactly what's vendored, from where, and under what license.

## Embedding model — `models/Xenova/all-MiniLM-L6-v2/`

- **Source:** https://huggingface.co/Xenova/all-MiniLM-L6-v2 (revision `main`)
- **License:** Apache-2.0 (ONNX port of `sentence-transformers/all-MiniLM-L6-v2`,
  also Apache-2.0)
- **Files & sizes (byte-exact from the Hugging Face manifest):**
  - `config.json` (650)
  - `tokenizer.json` (711,661)
  - `tokenizer_config.json` (366)
  - `special_tokens_map.json` (125)
  - `vocab.txt` (231,508)
  - `onnx/model_quantized.onnx` (22,972,370) — the 8-bit quantized model the app loads

## ONNX Runtime Web — `../ort/`

- **Source:** the `onnxruntime-web@1.14.0` npm package (`dist/`), the exact version
  pinned in `package-lock.json` and used by `@xenova/transformers@2.17.2`
- **License:** MIT
- **Files:** `ort-wasm-simd.wasm` (SIMD, primary), `ort-wasm.wasm` (non-SIMD fallback
  for iOS 16.4.x / older browsers). The app forces `numThreads = 1`, so only these
  two non-threaded binaries are ever requested.

## How the app loads these (see `src/cortex/ingest.ts`)

```
env.allowLocalModels = true;
env.allowRemoteModels = false;              // hard no-CDN
env.localModelPath = `${BASE_URL}models/`;
env.backends.onnx.wasm.wasmPaths = `${BASE_URL}ort/`;
env.backends.onnx.wasm.numThreads = 1;
```

## Reproducing this vendoring

```bash
# from frontend/
MD=public/models/Xenova/all-MiniLM-L6-v2
mkdir -p "$MD/onnx" public/ort
BASE=https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt onnx/model_quantized.onnx; do
  curl -fsSL "$BASE/$f" -o "$MD/$f"
done
cp node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm public/ort/
cp node_modules/onnxruntime-web/dist/ort-wasm.wasm public/ort/
```
