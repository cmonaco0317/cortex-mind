# Remediation prompt — `cortex-mind`

**This file is the task list. The standing rules live in `CLAUDE.md` at this repo
root — this file does not restate them, on purpose. Read `CLAUDE.md` first.**

Section numbers match the master review document, so gaps in the numbering are
expected.

---

## §-1 — Setup and session protocol (read this first)

**Before the first session, once per repo:**

1. Confirm `CLAUDE.md` exists in the repo root. If it does not, **stop and tell
   me** — the standing rules live there and this file deliberately does not
   restate them. Do not proceed from this file alone.
2. Confirm this file is committed at the repo root, not only pasted into chat, so
   it survives context compaction and can be re-read later.
3. `git checkout -b remediation` — never work these on `main`.
4. Install dependencies per the commands in `CLAUDE.md`, and confirm the existing
   test suite is green before changing anything. A red starting state means stop
   and tell me.

**Every session:**

- The rules in `CLAUDE.md` govern. Read it before starting work.
- **Do two or three concerns per session, not the whole file.** Work them in the
  order below. When the batch is done, write the §4 report, commit, and stop — do
  not start the next concern. A fresh session picks up from this same file.
- Announce which concerns you are taking at the start, and do not silently widen
  the batch.
- Before each fix, show me the failing test. After each fix, show me the passing
  run, as real pasted output.
- If context is running low mid-concern, stop at a clean commit and say where you
  stopped. Do not rush the remainder.

**Batching for this repo (8 concerns):**

| Session | Concerns | Why together |
|---|---|---|
| 1 | §2.1, §2.7 | CI first; the audit-claim reconciliation is asserted by that same job. |
| 2 | §2.2 step 1 only | Build and run the measurement script. Record the numbers. Change nothing else. |
| 3 | §2.2 steps 2–3, §2.3 | The scoring fix and the domain fix interact — do them together, measured. |
| 4 | §2.4 | Cross-language conformance, once §2.3 has settled what a domain is. |
| 5 | §2.5, §2.6 | Demo regeneration and the blind test, both downstream of the new scoring. |
| 6 | §2.8 | Documentation truth pass — last. |

§2.2 is a design change, not a patch. Do not compress it into a session with
other work.

---

## §2 — Repo: `cortex-mind`

### 2.1 CI builds but never tests

`.github/workflows/deploy.yml` runs `npm ci && npm run build` and deploys. It
never runs the 78 pytest tests, never runs vitest, and never runs the greps that
`SECURITY.md` presents as proof that `protobufjs` is absent from the bundle. A
commit that breaks every test in the repo deploys green, and the tree-shaking
proof has a shelf life of one dependency bump.

Fix: add a `test` job that runs `python3 -m pytest -q` and `npm test`, fails on 0
collected tests in either, and executes the `SECURITY.md` greps against the built
`dist/` as assertions. Make `deploy` depend on `test`.

Acceptance: deploy cannot run when a test fails or when `protobufjs` reappears in
`dist/`.

### 2.2 The surprise score is degenerate — two of three factors do nothing

Measured on the shipped `frontend/public/brain-safe.json` (237 concepts, 59
insights):

```
overlap:      0.000 for 54 of 59 insights;  max across the whole deck = 0.043
crossDomain:  true for 59 of 59
sim:          0.678 → 0.737     (total spread 0.06)
Spearman(score, cosine) = 0.85
```

`(1 − overlap)` ranges over [0.957, 1.000] — it cannot reorder anything.
`crossDomain` is constant, and a constant does not rank. What remains is cosine
similarity with a scalar on it.

This is structural, not corpus-specific. Candidates are drawn from
`order[i].slice(lo, hi)` = ranks 12–59 (ingest.ts:377-385) while overlap is
computed over `NBR = 12` nearest neighbours (ingest.ts:365). You are asking
whether two nodes share top-12 neighbours *after* selecting only pairs that are
rank-12-or-worse for each other. Zero is the near-certain answer.

Work in this order:

**Step 1 — instrument before changing anything.** Add a script (`analyze_brain.py`
or a vitest-run TS equivalent) that takes a brain JSON and reports, per factor:
min / median / max, variance, fraction at the degenerate value, and
**Spearman correlation between the final score and plain cosine**. Commit it, run
it on the shipped brain, and record the current numbers. This is the acceptance
instrument for everything below — do not change the formula before it exists.

**Step 2 — make the structural term actually measure something.** The candidate
window and the overlap window must overlap enough for the term to discriminate.
Options: compute overlap over a neighbourhood large enough to reach the candidate
band; or replace set-overlap with a measure that has range on this data — shared
second-degree neighbours, community/cluster assignment from the embeddings, or
graph distance in the kNN synapse graph excluding the candidate edge. Choose one,
justify it in a comment, and show its measured distribution.

**Step 3 — prove the score is not cosine in disguise.** After the fix, the
instrument must show every factor carrying real variance and the score
**demonstrably reordering** relative to cosine. Report the Spearman figure and
how much the top-10 set changes. If you cannot get a factor off its degenerate
value, delete that factor from the formula and from the README rather than
shipping an inert multiplier.

**Do not** achieve this by tuning `1.15` or `0.35`. Constants are not the
problem; the windows are.

### 2.3 `domain` is a folder name, and the README says it is an embedding measurement

`ingest.ts:242`:

```ts
const domain = parts.length > 1 ? slug(parts[parts.length - 2]).slice(0, 20) : "note";
```

The README (line ~56) states all three terms are "measured in the **full
embedding space**." That is false for the cross-domain term: it is string
inequality on a directory name. Worse — paste text, use "pick files," or supply
an agent trace, and every concept gets `domain: "note"`, so `cross` is always
false and the bonus is dead for the most likely first-use path.

Fix (prefer the first):
- **Derive domains from the embeddings** — cluster the vectors (k-means or
  similar, k chosen from the data) and use cluster identity as the domain. This
  makes the README claim true, makes the term work for pasted text, and removes
  the dependency on the user having tidy folders. Keep the folder name as a
  display label if useful, clearly distinguished from the scoring input.
- If you cannot, rename the field to what it is (`sourceFolder`), stop calling it
  cross-domain, and correct the README sentence.

Acceptance: with the §2.2 instrument, a pasted-text corpus must show a
non-constant domain factor.

### 2.4 The two "mirrored" implementations have already diverged

`build_brain.py:284` uses `rel.parts[0]` (top-level folder). `ingest.ts:242` uses
the immediate parent. The same nested corpus therefore gets different domain
labels, different bonuses, and different rankings depending on which pipeline ran
it. The only contract between them is the comment "mirrors splitPassages() in
frontend/src/cortex/text.ts."

Fix: pick one definition and make both match. Then add a **cross-language
conformance test**: a small fixture corpus committed to the repo, plus a golden
JSON of expected passages, domains, and ranked bridge pairs, asserted by both the
pytest suite and vitest. Any future divergence must turn CI red.

### 2.5 The demo does not demonstrate the product

`brain-safe.json` has `meta.dim: 768`. MiniLM-L6-v2 is 384. The default brain was
built by `build_brain.py` via Ollama `nomic-embed-text`, and its `why`/`angle`
text is llama3.2 output. So every visitor who clicks the live demo evaluates the
**offline pipeline** while reading README claims about the in-browser one — including
the note that card text is "composed from measurements, not generated," which is
true of a path the demo does not exercise.

Fix (prefer the first): regenerate the default brain through the **browser**
ingest path so it is 384-dim with composed card text, and make the demo the thing
the README describes. If you keep a pre-baked 768-dim brain, it must be labelled
unmistakably **in the app UI** — not only in the README — as offline-pipeline
output with generated explanations, and the README must say so adjacent to the
demo link rather than three sections below it.

### 2.6 The blind test cannot detect an effect

`blind_test.py` pits a scored bridge against "a plain nearest-neighbour." While
the score is ≈1.15 × cosine (§2.2), treatment and control are the same estimator
sampled from different slices of one sorted list. Inconclusive at n=20 is not an
underpowering problem; the design forecloses the result.

Fix, **after** §2.2 lands:
- Re-verify the baseline is genuinely distinct from the new score, using the 2.2
  instrument. State the measured distinction in the harness output.
- Compute and print the **n required** to detect a stated effect size at a stated
  power, and refuse to report a verdict below that n — print the requirement
  instead of an inconclusive verdict, so an underpowered run cannot read as
  evidence either way.
- Keep and strengthen the existing refusal to define a pass bar.

### 2.7 The README contradicts SECURITY.md

README (§Security) says `npm audit` "reports advisories including criticals."
`SECURITY.md:5` says it "currently reports **0 advisories**." The deps were fixed
and the front page was not updated.

Fix: reconcile from the actual current `npm audit` output, and have the §2.1 CI
job assert the claim so this cannot drift again.

### 2.8 Truth pass over README

After the above, re-read `README.md` end to end against the code. Particular
attention to: the surprise formula description, the "measured in the full
embedding space" sentence, the 384-dim claim in the How-it-works table, and the
test-count figure. Anything you cannot point at a test for, delete.

---

---

## §4 — Report format (end the session with this)

A table, one row per numbered concern:

| Concern | What changed | Test that proves it | Status |
|---|---|---|---|

`Status` is one of: **fixed** / **claim deleted instead** / **not done**.

Then, in prose:

1. Any concern where you changed a threshold, tolerance, or constant, and why
   that is not the fix quietly failing.
2. Any measured number that came out worse than the previous claim implied —
   above all the post-fix Spearman figure from §2.2, and the measured variance of
   each scoring factor. Report these first, not last.
3. Anything you believe the review got wrong, with the evidence.
4. What is still untrue in the README after your changes.

Do not summarize this as a success. List what remains.
