# CLAUDE.md — `cortex-mind`

## What this project is

A 100% client-side, in-browser curiosity engine. `frontend/src/cortex/` embeds a
corpus locally and ranks "non-obvious" concept pairs. `build_brain.py` is an
offline pipeline that produces the same artifact via Ollama. `agent-insights/` is
a separate stdlib-only Python tool that also lives in its own repo.

## Hard constraints — violating these breaks the product's central claim

1. **No network calls from `frontend/src/`, ever.** "Provably local" is the
   headline claim: the embedding model and ONNX WebAssembly are vendored under
   `frontend/public/`. The only permitted `fetch` calls are same-origin
   (`brains.json`, `brain-*.json`). No CDN, no telemetry, no analytics, no
   remote model loading.
2. **No new runtime dependencies.** Build-time devDependencies only, and only if
   a concern requires one.
3. **`build_brain.py` and `frontend/src/cortex/` implement the same algorithm in
   two languages.** Passage splitting, domain assignment, and the bridge score
   must stay in lockstep. If you change one, change the other in the same commit,
   and say so in the commit message. (Making this enforceable is §2.4.)
4. **`agent-insights/` is stdlib-only** and is byte-identical to the standalone
   `cmonaco0317/agent-insights` repo. Do not add dependencies to it, and do not
   edit it here without noting that the other copy needs the same change.

## How to run things

```bash
python3 -m pytest -q                    # from repo root — 91 tests
cd frontend && npm ci && npm test       # vitest
cd frontend && npm run build            # must stay green; CI deploys from dist/
```

`build_brain.py` and `blind_test.py` additionally need `pip install numpy`.

## Rules for how you work

1. **Every fix needs a regression test that fails before your change and passes
   after.** Write the failing test first, run it, show me the failure, then fix.
   If a concern cannot be expressed as a test, say so explicitly.
2. **Never fix by weakening a test or loosening an assertion.** If a test fails
   because the code is wrong, fix the code.
3. **Never fix by tuning a constant until a metric moves.** Specifically: the
   `1.15` cross-domain bonus and the `0.35` same-document discount are not the
   problem and are not the fix. Changing them to shift a correlation figure is
   the fix quietly failing, and you must report it as such.
4. **Never fix by widening a claim to match what you built.** The README serves
   the claim; the code does not serve the README.
5. **"I could not make this true, so I removed the claim" is a successful
   outcome.** Report it as success. Do not leave aspirational prose in place with
   a hedge bolted on. An inert term deleted from the formula and from the README
   is a better result than an inert term kept.
6. **Measure before you change scoring.** Do not touch the bridge score before
   the measurement script exists and has recorded the current numbers.
7. **Show me output, do not describe it.** When you say something passes, paste
   the actual command and its actual output. When you report a distribution or a
   correlation, paste the script's real output. This is the rule most likely to
   slip late in a session — hold it.
8. **One commit per concern**, message naming the failure it fixes, in the style
   already in this repo (`fix: make the insight engine actually rank, and the
   claims actually true`).
9. **No scope creep.** Do not refactor, restyle, or "improve" anything outside
   the concern you are working.
10. **Do not tell me the work went well.** End with the report format in the
    remediation file, including everything you could not do.
11. **Stop and ask** rather than guessing, if a fix would require a network call,
    a new dependency, regenerating a committed brain JSON, or changing a public
    claim.
