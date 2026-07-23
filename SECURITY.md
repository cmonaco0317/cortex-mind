# Security

## Dependency advisories: what reaches a visitor

`npm audit` currently reports 11 advisories on this project, including three
criticals. **None of them reach someone who loads the page.** That's a specific,
checkable claim rather than a reassurance, so here is the check.

### The chain that looks alarming

```
@xenova/transformers → onnxruntime-web → onnx-proto → protobufjs
```

`protobufjs@6.11.6` carries a critical advisory (arbitrary code execution) plus
several highs. It is a transitive dependency of the embedding runtime, so the
obvious conclusion is that a shipped, user-facing library has a critical CVE.

That conclusion is wrong, and it's worth showing why rather than asserting it.

### Why it doesn't ship

The ONNX model is not parsed by JavaScript here. It's parsed inside the **vendored
WebAssembly runtime** (`public/ort/ort-wasm.wasm`, `ort-wasm-simd.wasm`), which uses
ORT's own C++ protobuf implementation. The JS `protobufjs` package is tree-shaken out
of the build entirely.

Verify it yourself — every one of these returns nothing:

```bash
cd frontend && npm run build
grep -rIl "protobufjs"      dist/     # 0 files
grep -rIl "onnx-proto"      dist/     # 0 files
grep -rIl "google/protobuf" dist/     # 0 files
grep -rIl '$protobuf'       dist/     # 0 files
```

The same applies to `sharp`: it's a Node native module that cannot execute in a
browser at all. (A string search for `sharp` in the bundle *does* hit — but it's a
minifier-generated variable name, `const sharp={}`, not the library.)

### What the advisories actually put at risk

Not visitors — **the build machine**. `vite`, `vitest`, `esbuild`, `sharp` and the
rest execute during `npm ci` and `npm run build`, on a developer's laptop or in CI.
That's a real if narrower exposure, and it's the reason the non-breaking upgrades
were applied (14 advisories → 11) rather than shrugged off.

### Why the remaining 11 are still there

`npm audit fix --force` resolves them by downgrading `@xenova/transformers` from
2.17.2 to **1.4.2** — a major-version downgrade that breaks the local-model loading
this project depends on. Taking a breaking change to silence advisories on code that
never ships would trade a real feature for a cosmetic number.

They stay, documented, until the upstream chain updates.

## Runtime security properties

- **No network at runtime.** The embedding model and the ONNX WebAssembly are
  vendored under `frontend/public/`, and `env.allowRemoteModels = false` is set
  explicitly. Open the Network tab while it embeds: requests go to your own origin
  and nowhere else. This is the load-bearing privacy claim and it's observable.
- **No backend, no account, no telemetry.** Your corpus is never transmitted.
- **Corpus text is never rendered as HTML.** Insight cards use `textContent`, so a
  note containing markup cannot become script.
- **Secrets are redacted on ingest** — before text becomes a neuron, an insight, or a
  shareable card. Agent traces routinely contain API keys; without this the
  watermarked share image would be a leak vector.

## agent-insights

`agent-insights/` reads Claude Code session logs, so it carries its own tripwire:
paths are one-way hashed, model names collapse to a coarse family, no prompt text is
retained, and the run **hard-fails** (`SystemExit`) if anything path-, email- or
secret-shaped would be emitted.

That tripwire is covered by a 16-shape test matrix, each asserted blocked as a value,
as a dictionary key, and nested inside a list — plus the inverse, that ordinary
counts and ratios still pass, since a tripwire that fires on normal metrics is one
that gets switched off.

The matrix exists because reading the pattern wasn't enough. It was originally
`sk-[A-Za-z0-9]{20,}`, which does **not** match a real Anthropic key: `sk-ant-api03-…`
breaks the alphanumeric run with a hyphen after three characters. The single likeliest
credential to appear in a Claude Code log was the one shape that got through. It
surfaced only by running real key formats at it.

## Reporting

Open an issue. This is a personal project with no SLA — please don't rely on it for
anything where a security response time matters.
