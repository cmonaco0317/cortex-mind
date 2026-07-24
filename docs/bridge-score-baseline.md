# Bridge-score baseline

Recorded 2026-07-23, before any change to the scoring, by

```bash
python3 analyze_brain.py
```

This is the "before" that every later scoring change is measured against. It is
committed rather than pasted into a conversation so the comparison is still
possible in six months. Regenerate it with the command above; the instrument is
deterministic and reads only the committed brain.

```
brain    : frontend/public/brain-safe.json
meta     : 237 concepts, 1393 synapses, 59 insights, dim 768

FACTORS
  sim (cosine)   min   0.6780   median   0.6950   max   0.7367   spread  0.0587   var 2.257e-04
  overlap        min   0.0000   median   0.0000   max   0.0435   spread  0.0435   var 1.493e-04
                 54/59 (92%) sit at the degenerate value 0
  1 - overlap    min   0.9565   median   1.0000   max   1.0000   spread  0.0435   var 1.493e-04
                 54/59 (92%) sit at the degenerate value 1
  crossDomain    true for 59/59 (100%)   CONSTANT — cannot reorder anything
  sameDocument   ABSENT from this artifact — build_brain.py emits it, so this
                 brain predates the term entirely and it cannot be measured here.

  final score    min   0.7797   median   0.7981   max   0.8473   spread  0.0676   var 2.361e-04

IS THE SCORE JUST COSINE?
  Spearman(score, cosine) = 0.8469
  top-5   agreement with a plain cosine ranking: 3/5 (60%)
  top-10  agreement with a plain cosine ranking: 6/10 (60%)
  top-20  agreement with a plain cosine ranking: 15/20 (75%)

FORMULA CONSISTENCY (recomputed from the stored factors)
  max |recomputed - stored| = 1.00e-04
  stored scores match the documented formula
```

## What this says

The score is `sim x (1 - overlap) x (1.15 if crossDomain else 1) x (0.35 if sameDocument else 1)`.
On the shipped deck:

- **`crossDomain` is constant.** True for 59 of 59. A constant is a scalar, not a
  ranking factor; it contributes exactly nothing to the order.
- **`overlap` is degenerate for 92% of the deck** — exactly 0.0 for 54 of 59,
  maximum 0.0435 across the whole deck. Candidates are drawn from ranks 12-59
  while overlap is computed over the 12 nearest neighbours, so the question
  being asked is whether two nodes share top-12 neighbours *after* selecting
  only pairs that are rank-12-or-worse for each other. Zero is the near-certain
  answer.
- **`sameDocument` is not in this artifact at all.** `build_brain.py` emits it,
  so the committed brain predates the term. It cannot be measured here.
- **`sim` carries a spread of 0.0587**, which is the entire dynamic range the
  formula has to work with.

Spearman(score, cosine) = **0.8469**, which is not 1.0 — so the formula is not
*purely* cosine. The reason is worth stating precisely, because it is easy to
get backwards: `(1 - overlap)` spans [0.9565, 1.0], and while that range is
trivial in absolute terms, `sim`'s own spread is only 0.0587. A 4.35% multiplier
is therefore large relative to the spread it competes with, and it does reorder
the deck: **only 6 of the top 10 insights survive as a plain cosine top 10.**

So the honest summary is not "two of three factors do nothing". It is: one
factor does nothing at all (`crossDomain`), one is inert for 92% of pairs but
decisive for the handful where it fires (`overlap`), one could not be measured
(`sameDocument`), and the remaining variance is cosine.
