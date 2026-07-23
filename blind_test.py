#!/usr/bin/env python3
"""
Blind A/B evaluation harness for the curiosity engine.

Measures whether the engine's "insight leaps" are genuinely more novel/useful than
what plain cosine-similarity nearest-neighbours already return. Builds a
self-contained HTML rating sheet that puts the two BLIND, side by side, for a rater
to judge — then scores the result.

For each of ~20 source concepts it shows two connections:
  - BASELINE: the nearest neighbour by cosine similarity (the "obvious" match).
  - ENGINE:   a moderately-similar-but-distant concept (a non-obvious leap).
Order is randomised and which-is-which is hidden. The rater marks which (if any)
is a genuinely novel + useful connection.

Scoring is an exact two-sided binomial test against chance on the decisive ratings
(the ones where a rater picked a side). It deliberately does NOT use a fixed
percentage bar: an earlier version passed the engine at >=30% wins, which is at or
below chance once "neither" is an option, so it could report PASS while the baseline
was actually preferred more than twice as often.

Read the result narrowly. One rater at n=20 can show that *this* rater preferred one
side on *this* corpus. It cannot establish a general effect, and the harness says so
in its own output rather than implying otherwise.

Usage (runs 100% locally; the HTML never leaves your machine):
  blind_test.py --corpus cortex/corpus_safe.json --out cortex/blind-test.html
  blind_test.py --ingest ~/notes --out cortex/blind-test.html --n 20
"""

import argparse
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_brain import embed, ingest_folder  # noqa: E402


def load_corpus(args: argparse.Namespace) -> list[dict]:
    if args.ingest:
        return ingest_folder(args.ingest, args.max)
    return json.load(open(args.corpus))


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
        # engine = a moderately-similar-but-distant, still-unused concept (a leap)
        band = [
            int(j)
            for j in order[i, 15:50]
            if int(j) not in used and int(j) not in (i, baseline)
        ]
        if not band:
            continue
        engine = int(rng.choice(band))
        used.add(baseline)
        used.add(engine)
        rows.append({"source": i, "baseline": baseline, "engine": engine})
    return rows


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
    print(
        f"wrote {args.out}: {len(rows)} blind rating items. Open it and rate to get the verdict."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
