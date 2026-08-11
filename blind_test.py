#!/usr/bin/env python3
"""
Blind A/B evaluation harness for the curiosity engine.

Measures whether the engine's "insight leaps" are genuinely more novel/useful than
what plain cosine-similarity nearest-neighbours already return. Builds a
self-contained HTML rating sheet that puts the two BLIND, side by side, for a rater
to judge — then scores the result.

For each source concept it shows two connections:
  - BASELINE: the nearest neighbour by cosine similarity (the "obvious" match).
  - ENGINE:   the engine's own top-scored bridge for that source, from the SAME
              rank_bridges() the product ships.
Order is randomised and which-is-which is hidden. The rater marks which (if any)
is a genuinely novel + useful connection.

The engine arm used to be `rng.choice(order[i, 15:50])` -- a RANDOM pick from a
rank band, which never touched the scoring at all. The harness therefore could
not have detected a change to the score in either direction, because the score
was not in the experiment. Both arms were one estimator sampled from two slices
of one sorted list, and the design foreclosed the result before any rater saw it.
The harness now prints, on every run, how far apart the arms actually are: how
many resolve to the same concept (must be 0) and where the engine's pick sits in
the plain-cosine ordering (the baseline is rank 0 by definition).

Scoring is an exact two-sided binomial test against chance on the decisive ratings
(the ones where a rater picked a side). It deliberately does NOT use a fixed
percentage bar: an earlier version passed the engine at >=30% wins, which is at or
below chance once "neither" is an option, so it could report PASS while the baseline
was actually preferred more than twice as often.

It also refuses to report a verdict it cannot earn. Detecting a 65/35 preference
at 80% power (two-sided, alpha=0.05) needs 82 decisive ratings; below that the
page prints the REQUIREMENT instead of a conclusion. "Inconclusive" at n=20 reads
as weak evidence of no effect, and it is not -- it is no evidence at all.

Read even a powered result narrowly: one rater on one corpus is one rater on one
corpus, and the harness says so in its own output rather than implying otherwise.

Usage (runs 100% locally; the HTML never leaves your machine):
  blind_test.py --corpus cortex/corpus_safe.json --out cortex/blind-test.html
  blind_test.py --ingest ~/notes --out cortex/blind-test.html --n 20
"""

import argparse
import html
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_brain import embed, ingest_folder, rank_bridges  # noqa: E402


def load_corpus(args: argparse.Namespace) -> list[dict]:
    if args.ingest:
        return ingest_folder(args.ingest, args.max)
    return json.load(open(args.corpus))


# Detecting a preference this strong, at this power, with a two-sided exact
# binomial test. Stated up front rather than chosen after seeing the data.
EFFECT = 0.65  # a 65/35 split among decisive ratings
ALPHA = 0.05
POWER = 0.80


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def _reject_region(n: int, alpha: float) -> set[int]:
    """Two-sided exact-binomial rejection region against p=0.5: the most extreme
    outcomes whose total probability under H0 stays within alpha."""
    ks = sorted(range(n + 1), key=lambda k: (abs(k - n / 2), k), reverse=True)
    region, mass = set(), 0.0
    for k in ks:
        m = _binom_pmf(k, n, 0.5)
        if mass + m > alpha:
            break
        region.add(k)
        mass += m
    return region


def required_n(effect: float = EFFECT, alpha: float = ALPHA, power: float = POWER) -> int:
    """Smallest number of DECISIVE ratings with at least `power` chance of
    detecting a true `effect` split. Exact, not a normal approximation."""
    for n in range(4, 2001):
        region = _reject_region(n, alpha)
        if not region:
            continue
        achieved = sum(_binom_pmf(k, n, effect) for k in region)
        if achieved >= power:
            return n
    return 2001


def build_pairs(corpus: list[dict], n_rows: int, seed: int) -> list[dict]:
    vecs = []
    for i, c in enumerate(corpus):
        vecs.append(embed(c.get("text") or c.get("label") or c["id"]))
        if (i + 1) % 25 == 0 or i + 1 == len(corpus):
            print(f"  embedded {i + 1}/{len(corpus)}", flush=True)
    X = np.asarray(vecs, dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)

    rng = np.random.default_rng(seed)
    n = len(corpus)
    lo = min(12, max(2, n // 4))
    hi = min(60, n - 1)
    # The ENGINE arm is the engine's own top-scored bridge for that source, taken
    # from the SAME rank_bridges() the product ships. It used to be
    # `rng.choice(order[i, 15:50])` -- a random pick from a rank slice, which
    # never touched the scoring at all: the harness could not have detected a
    # change to the score, in either direction, because the score was not in it.
    ranked = rank_bridges(sim, order, Xn, corpus, lo, hi)
    best_for: dict[int, int] = {}
    for c in ranked:  # already sorted most-surprising-first
        for a, b in ((c["i"], c["j"]), (c["j"], c["i"])):
            if a not in best_for:
                best_for[a] = b

    sources = rng.choice(n, size=min(n_rows, n), replace=False)
    rows = []
    # Every concept is used at most once across the whole test — as a source OR an
    # answer option — so no "hub" concept recurs as an option across questions
    # (which would bias the rater). Sources are reserved up front.
    used: set[int] = {int(s) for s in sources}
    for i in sources:
        i = int(i)
        # baseline = nearest still-unused neighbour (the "obvious" match)
        baseline = next(
            (int(j) for j in order[i] if int(j) not in used and int(j) != i), None
        )
        if baseline is None:
            continue
        engine = best_for.get(i)
        if engine is None or engine in used or engine in (i, baseline):
            continue
        used.add(baseline)
        used.add(engine)
        rows.append(
            {
                "source": i,
                "baseline": baseline,
                "engine": engine,
                # how far down the plain-cosine ordering the engine's pick sits;
                # rank 0 would mean the two arms are the same estimator
                "engine_cosine_rank": int(np.where(order[i] == engine)[0][0]),
            }
        )
    return rows


def report_distinctness(rows: list[dict]) -> dict:
    """State, in the harness's own output, how far apart the two arms actually are.

    §2.6's point: while the score was ~1.15 x cosine, treatment and control were
    one estimator sampled from two slices of one sorted list, so the design
    foreclosed the result before any rater saw it. A harness that cannot show its
    arms are distinct is not evidence of anything.
    """
    ranks = [r["engine_cosine_rank"] for r in rows]
    same = sum(1 for r in rows if r["engine"] == r["baseline"])
    return {
        "n_rows": len(rows),
        "arms_identical": same,
        "engine_cosine_rank_min": min(ranks) if ranks else None,
        "engine_cosine_rank_median": int(sorted(ranks)[len(ranks) // 2]) if ranks else None,
        "engine_cosine_rank_max": max(ranks) if ranks else None,
    }


def render_html(corpus: list[dict], rows: list[dict], name: str) -> str:
    # Values below are inserted via textContent in the page JS, so they are NOT
    # HTML-escaped here (escaping would render literal &#x27; etc.). Show the full
    # concept text so the rater can judge the connection fairly.
    def snippet(idx: int) -> str:
        return (corpus[idx].get("text") or "").strip()

    def label(idx: int) -> str:
        return corpus[idx].get("label", corpus[idx]["id"])

    # Each row ships its two options with a hidden "kind" so scoring is automatic.
    data = []
    for r in rows:
        opts = [
            {
                "kind": "engine",
                "label": label(r["engine"]),
                "text": snippet(r["engine"]),
            },
            {
                "kind": "baseline",
                "label": label(r["baseline"]),
                "text": snippet(r["baseline"]),
            },
        ]
        # (order is shuffled client-side so the rater can't infer which is which)
        data.append(
            {
                "source_label": label(r["source"]),
                "source_text": snippet(r["source"]),
                "options": opts,
            }
        )

    payload = json.dumps(data).replace("</", "<\\/")  # safe to embed inline in <script>
    effect, alpha, power = EFFECT, ALPHA, POWER
    req_n = required_n()
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Cortex — blind insight test ({html.escape(name)})</title>
<style>
  body{{background:#05060d;color:#cfe6f2;font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:0 auto;padding:28px}}
  h1{{font-size:19px;color:#9fe6ff}} .sub{{color:#5f8296;font-size:13px;margin-bottom:22px}}
  .row{{border:1px solid rgba(80,160,200,.2);border-radius:10px;padding:16px;margin-bottom:16px}}
  .src{{font-weight:600;color:#eafaff}} .src small{{display:block;color:#6f95a8;font-weight:400;margin-top:3px}}
  .opts{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}}
  .opt{{border:1px solid rgba(80,160,200,.25);border-radius:8px;padding:10px;cursor:pointer}}
  .opt.sel{{border-color:#4fd6f5;background:rgba(30,90,120,.3)}}
  .opt b{{color:#bde8ff}} .opt small{{display:block;color:#6f95a8;margin-top:4px}}
  .neither{{margin-top:8px;font-size:12px;color:#6f95a8;cursor:pointer}}
  .neither.sel{{color:#4fd6f5}}
  button{{background:#1c4a5e;color:#dff;border:1px solid #4fd6f5;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer;margin-top:8px}}
  #verdict{{margin-top:20px;padding:18px;border-radius:10px;font-size:16px;display:none}}
  .go{{background:rgba(30,120,60,.3);border:1px solid #4ade80;color:#c6f6d5}}
  .stop{{background:rgba(140,40,40,.3);border:1px solid #f87171;color:#fecaca}}
  .meh{{background:rgba(90,90,110,.3);border:1px solid #a1a1aa;color:#e4e4e7}}
  #verdict small{{display:block;margin-top:8px;opacity:.85;font-size:13px;line-height:1.5}}
</style></head><body>
<h1>Cortex — blind insight test</h1>
<div class="sub">For each concept, pick the connection that is a <b>genuinely novel &amp; useful</b> insight
(better than an obvious "related item"). If neither is, pick "neither". You can't tell which is the engine — that's the point.
Corpus: <b>{html.escape(name)}</b> · {len(rows)} items. 100% local.</div>
<div id="rows"></div>
<button onclick="score()">Score it</button>
<div id="verdict"></div>
<script>
const DATA = {payload};
// Power requirement, computed in Python before this page was written, so the
// bar is fixed in advance rather than chosen after seeing the ratings.
const EFFECT = {effect};
const ALPHA = {alpha};
const POWER = {power};
const REQUIRED_N = {req_n};
const shuffled = DATA.map(r => {{
  const o = r.options.slice();
  if (Math.random() < 0.5) o.reverse();
  return {{...r, options:o}};
}});
const picks = new Array(shuffled.length).fill(null);
const root = document.getElementById('rows');
shuffled.forEach((r,ri) => {{
  const div = document.createElement('div'); div.className='row';
  const src = document.createElement('div'); src.className='src';
  src.innerHTML = '<span></span>';
  src.firstChild.textContent = r.source_label;
  const s2 = document.createElement('small'); s2.textContent = r.source_text; src.appendChild(s2);
  const opts = document.createElement('div'); opts.className='opts';
  r.options.forEach((o,oi) => {{
    const el = document.createElement('div'); el.className='opt';
    const b=document.createElement('b'); b.textContent=o.label;
    const sm=document.createElement('small'); sm.textContent=o.text;
    el.append(b,sm);
    el.onclick=()=>{{picks[ri]={{kind:o.kind}};[...opts.children].forEach(c=>c.classList.remove('sel'));el.classList.add('sel');nb.classList.remove('sel');}};
    opts.appendChild(el);
  }});
  const nb=document.createElement('div'); nb.className='neither'; nb.textContent='· neither is genuinely novel/useful';
  nb.onclick=()=>{{picks[ri]={{kind:'neither'}};[...opts.children].forEach(c=>c.classList.remove('sel'));nb.classList.add('sel');}};
  div.append(src,opts,nb); root.appendChild(div);
}});
// Exact two-sided binomial test against p=0.5. Used instead of a fixed
// percentage bar: the old rule passed the engine at >=30% wins, which is at or
// BELOW chance once "neither" is an option -- it could print PASS while the
// baseline was actually preferred more than twice as often.
function logC(n,k){{ let s=0; for(let i=1;i<=k;i++) s += Math.log(n-k+i) - Math.log(i); return s; }}
function binomTwoSided(k,n){{
  if(n===0) return 1;
  const pmf = (i)=> Math.exp(logC(n,i) - n*Math.LN2);
  const obs = pmf(k); let p=0;
  for(let i=0;i<=n;i++){{ const pi=pmf(i); if(pi <= obs*(1+1e-9)) p += pi; }}
  return Math.min(1,p);
}}
function score(){{
  const done = picks.filter(Boolean).length;
  if (done < shuffled.length) {{ alert('Rate all '+shuffled.length+' items first ('+done+' done)'); return; }}
  const engineWins   = picks.filter(p=>p.kind==='engine').length;
  const baselineWins = picks.filter(p=>p.kind==='baseline').length;
  const neither      = picks.filter(p=>p.kind==='neither').length;
  const decisive     = engineWins + baselineWins;
  const p            = binomTwoSided(engineWins, decisive);
  const v = document.getElementById('verdict');
  v.style.display='block';

  const tally = '<b>engine ' + engineWins + ' · baseline ' + baselineWins
    + ' · neither ' + neither + '</b> (of ' + shuffled.length + ' rated)<br>'
    + 'Two-sided binomial test on the ' + decisive + ' decisive ratings: p = ' + p.toFixed(3) + '.';
  const caveat = '<small>One rater, n=' + shuffled.length + '. This measures whether '
    + 'THIS rater preferred the engine\\u2019s leaps to the nearest-neighbour baseline on '
    + 'THIS corpus \\u2014 not that the engine is good in general. A single-rater result is '
    + 'suggestive at best; it is not evidence of a general effect.</small>';

  if (decisive === 0) {{
    v.className='meh';
    v.innerHTML = tally + '<br>\\u2014 No decisive ratings: neither side was preferred anywhere. Inconclusive.' + caveat;
  }} else if (decisive < REQUIRED_N) {{
    // Underpowered is not the same as inconclusive. An underpowered run cannot
    // support a claim in EITHER direction, and reporting it as "inconclusive"
    // invites reading it as weak evidence of no effect. Print what would have
    // been required instead of a verdict that cannot be earned at this n.
    v.className='meh';
    v.innerHTML = tally + '<br>\\u26D4 <b>NO VERDICT \\u2014 UNDERPOWERED.</b> '
      + 'Detecting a ' + Math.round(EFFECT*100) + '/' + Math.round((1-EFFECT)*100)
      + ' preference at ' + Math.round(POWER*100) + '% power (two-sided, \\u03B1=' + ALPHA + ') '
      + 'needs <b>' + REQUIRED_N + ' decisive ratings</b>; this run has ' + decisive + '. '
      + 'No conclusion is reported in either direction, because none can be.' + caveat;
  }} else if (p >= 0.05) {{
    v.className='meh';
    v.innerHTML = tally + '<br>\\u2014 <b>INCONCLUSIVE</b> at this sample size. The split is '
      + 'consistent with a coin flip, so this run supports no claim in either direction.' + caveat;
  }} else if (engineWins > baselineWins) {{
    v.className='go';
    v.innerHTML = tally + '<br>\\u2705 The engine\\u2019s leaps were preferred significantly more often than the baseline.' + caveat;
  }} else {{
    v.className='stop';
    v.innerHTML = tally + '<br>\\uD83D\\uDED1 The <b>baseline</b> was preferred significantly more often than the engine.' + caveat;
  }}
}}
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus")
    src.add_argument("--ingest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20, help="number of items to rate")
    ap.add_argument("--max", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    corpus = load_corpus(args)
    print(f"corpus: {len(corpus)} concepts; embedding…", flush=True)
    rows = build_pairs(corpus, args.n, args.seed)
    open(args.out, "w").write(
        render_html(corpus, rows, name=os.path.basename(args.out))
    )
    dist = report_distinctness(rows)
    req = required_n()
    print(
        f"wrote {args.out}: {len(rows)} blind rating items. Open it and rate to get the verdict."
    )
    # State the two things that decide whether this run can mean anything, in the
    # harness's own output, before any rater is involved.
    print("\narms distinctness (§2.6) — the engine arm is rank_bridges()' own top")
    print(f"  pick for each source, NOT a random draw from a rank band:")
    print(f"  rows                          : {dist['n_rows']}")
    print(f"  arms identical (must be 0)    : {dist['arms_identical']}")
    print(
        "  engine pick's rank in the plain-cosine ordering: "
        f"min {dist['engine_cosine_rank_min']}, "
        f"median {dist['engine_cosine_rank_median']}, "
        f"max {dist['engine_cosine_rank_max']}"
    )
    print(
        f"\npower (§2.6): detecting a {EFFECT:.0%}/{1 - EFFECT:.0%} preference at "
        f"{POWER:.0%} power, two-sided alpha={ALPHA}, needs {req} DECISIVE ratings."
    )
    if len(rows) < req:
        print(
            f"  this sheet has {len(rows)} items, so even a unanimous result cannot "
            f"reach that bar. The page will report NO VERDICT rather than a\n"
            f"  conclusion it has not earned. Raise --n to at least {req} "
            f"(and expect some ratings to be 'neither')."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
