# Security

## Dependency advisories: what reaches a visitor

`npm audit` currently reports **0 advisories**. That number is the least interesting
part of this section, and it is deliberately not the argument — for a while the count
was 11, including three criticals, and **none of those reached someone who loaded the
page either**. What matters is which code actually ships, so here is that check, kept
because it is what the claim rests on.

### The chain that looks alarming

```
@xenova/transformers → onnxruntime-web → onnx-proto → protobufjs
```

`protobufjs@6.11.6` carried a critical advisory (arbitrary code execution) plus
several highs. It is a transitive dependency of the embedding runtime, so the
obvious conclusion was that a shipped, user-facing library had a critical CVE.

That conclusion was wrong, and it's worth showing why rather than asserting it —
especially now that the advisory is gone, since a green `npm audit` is not evidence
of anything on its own.

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
That's a real if narrower exposure, and this repo auto-deploys from CI on every push
to `main`, so it is the exposure worth actually closing.

### How they were closed, and the trap that was avoided

**Do not run `npm audit fix --force` here.** It "resolves" the chain by downgrading
`@xenova/transformers` from 2.17.2 to **1.4.2** — a major-version downgrade that
breaks the local-model loading this project depends on. Trading a real feature for a
cosmetic number is the wrong trade, and it stayed the wrong trade even while the
count sat at 11.

What worked instead, without touching `@xenova/transformers` at all:

- **The two vulnerable transitives are pinned forward** in `frontend/package.json`
  `overrides` — `protobufjs@7.6.5`, `sharp@0.35.3`. The deprecated parent stays at
  2.17.2. This was always available; it just isn't what `audit fix` reaches for.
- **The build tooling took its major upgrades** — `vite` 5.4.21 → 6.4.3, `vitest` and
  `@vitest/coverage-v8` 2.1.9 → 3.2.7. These are the packages that actually execute
  during a build, so they are the ones where the exposure was real.

Verified before it was committed, because a major bump of the bundler is exactly the
kind of change that passes CI and breaks the artifact: `tsc` clean, 36 tests green,
`npm ci` reproducible from the committed lock, bundle within 1 kB of the previous
build with an identical CSS hash, and the built page loaded in a browser and rendered
the graph (237 neurons, 1,393 synapses) with an empty console. Node engines were
checked against the Node 20 the deploy workflow pins.

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
