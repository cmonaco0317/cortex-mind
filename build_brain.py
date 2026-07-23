#!/usr/bin/env python3
"""
Build a "brain map" (neurons + synapses + 3D layout) from a corpus.

Each concept becomes a NEURON. Its text is embedded with a local Ollama model
(nomic-embed-text) — nothing leaves the machine. Cosine similarity defines
SYNAPSES (k-nearest-neighbours + a few long-range "insight" bridges). A PCA
projection gives a semantically meaningful 3D layout (related concepts sit near
each other), shaped into an organic brain-like ellipsoid.

The output JSON is consumed by the frontend curiosity engine, which runs live
spreading-activation over this real semantic graph — the firing is not scripted.

Bring your own corpus, two ways:
  1. A concept JSON: [{"id","label","domain","text"}, ...]
       build_brain.py --corpus mine.json --out frontend/public/brain-mine.json --name mine
  2. Point it at a folder of notes/docs (markdown/text) — one file per concept:
       build_brain.py --ingest ~/notes --out frontend/public/brain-notes.json --name notes

Add --manifest frontend/public/brains.json to register the brain for the UI picker.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
GENERATE_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"

TEXT_EXTS = {".md", ".markdown", ".txt", ".mdx", ".rst"}
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "__pycache__",
    ".obsidian",
}


def embed(text: str) -> list[float]:
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["embedding"]


def _trim(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > limit * 0.6 else cut).rstrip(" ,;:") + "…"


def _fallback_insight(
    a: dict[str, Any],
    b: dict[str, Any],
    sim: float | None = None,
    overlap: float | None = None,
) -> dict[str, str]:
    """Used when the local LLM is unavailable or returns nothing usable.

    Every clause states a number that was actually measured. The previous version
    asserted that each pair "rarely appear together" without checking anything,
    and contradicted itself on same-domain pairs.
    """
    da, db = a.get("domain", "?"), b.get("domain", "?")
    cross = da != db
    where = (
        f"bridging {da} and {db}"
        if cross
        else f"inside {da}, between two groups that otherwise don't touch"
    )
    if sim is None or overlap is None:
        return {
            "why": f"A long-range link {where}.",
            "angle": f"What would '{a['label']}' look like if reframed through '{b['label']}'?",
        }
    relatedness = (
        "Strongly related"
        if sim >= 0.75
        else "Clearly related" if sim >= 0.55 else "Loosely related"
    )
    separation = (
        "they share no near neighbours at all"
        if overlap == 0
        else f"they share only {round(overlap * 100)}% of their near neighbours"
    )
    return {
        "why": (
            f"{relatedness} (cosine {sim:.2f}), yet {separation} — "
            f"a link {where} that neither one's own neighbourhood would surface."
        ),
        "angle": f"What would '{a['label']}' look like if reframed through '{b['label']}'?",
    }


def llm_insight(
    a: dict[str, Any],
    b: dict[str, Any],
    sim: float | None = None,
    overlap: float | None = None,
) -> dict[str, str]:
    """Precompute (at build time, locally) why a long-range bridge is non-obvious.

    Runs on the local LLM so the shipped brain never needs a runtime model call —
    the corpus text never leaves the machine.
    """
    prompt = (
        "Two concepts from a knowledge base are being connected as a surprising, "
        "non-obvious insight.\n"
        f"A: {a['label']} — {a.get('text', '')[:200]}\n"
        f"B: {b['label']} — {b.get('text', '')[:200]}\n"
        "In ONE sentence, explain why linking A and B is non-obvious yet meaningful. "
        "Then give ONE short question or angle this connection opens up. "
        'Return JSON: {"why": "...", "angle": "..."}'
    )
    body = json.dumps(
        {
            "model": LLM_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.7},
        }
    ).encode()
    req = urllib.request.Request(
        GENERATE_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(json.loads(resp.read())["response"])
        why = _trim(str(data.get("why", "")), 240)
        angle = _trim(str(data.get("angle", "")), 200)
        fb = _fallback_insight(a, b, sim, overlap)
        return {"why": why or fb["why"], "angle": angle or fb["angle"]}
    except (OSError, ValueError, KeyError):
        return _fallback_insight(a, b, sim, overlap)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "concept"


def _clean_markdown(raw: str) -> tuple[str, str]:
    """Return (title, plain_text) from a markdown/text document."""
    lines = raw.splitlines()
    title = ""
    for ln in lines:
        m = re.match(r"^#{1,6}\s+(.*)", ln.strip())
        if m:
            title = m.group(1).strip()
            break
    text = raw
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code fences
    text = re.sub(r"`[^`]*`", " ", text)  # inline code
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links/images
    text = re.sub(r"[#>*_~\-|]", " ", text)  # md punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


# Passage splitting — mirrors splitPassages() in frontend/src/cortex/text.ts.
# Both paths must build the same graph from the same folder, or the offline CLI
# and the in-browser flow would disagree about what a "concept" even is.
PASSAGE_TARGET = 900  # a paragraph longer than this is cut on sentence ends
PASSAGE_MIN = 30  # below this a fragment is glued to its neighbour
MAX_PASSAGES_PER_DOC = 40
PASSAGE_TEXT_CAP = 1200


def _strip_markdown(raw: str) -> str:
    text = re.sub(r"```.*?```", " ", raw, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", " ", text, flags=re.M)
    text = re.sub(r"[#>*_~|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _chunk_sentences(text: str, size: int) -> list[str]:
    sentences = re.findall(r"[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$", text) or [text]
    out: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > size:
            out.append(buf.strip())
            buf = ""
        buf += s
    if buf.strip():
        out.append(buf.strip())
    return [x for x in out if x]


def _split_passages(raw: str, doc_title: str = "") -> list[tuple[str, str]]:
    """Break a document into (heading, text) passages, each about ONE thing.

    A blank line is the author saying "new idea", so passages are NOT greedily
    packed to PASSAGE_TARGET across paragraph breaks — only true fragments are
    merged.
    """
    sections: list[tuple[str, list[str]]] = []
    heading, body = doc_title, []
    for ln in raw.splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.*)", ln)
        if m:
            if " ".join(body).strip():
                sections.append((heading, body))
            heading, body = m.group(1).strip(), []
        else:
            body.append(ln)
    if " ".join(body).strip() or not sections:
        sections.append((heading, body))

    out: list[list[str]] = []  # [heading, text] so trailing fragments can merge
    for head, lines in sections:
        paras = [
            p
            for p in (
                _strip_markdown(x) for x in re.split(r"\n\s*\n", "\n".join(lines))
            )
            if p
        ]
        buf = ""

        def flush() -> None:
            nonlocal buf
            t = buf.strip()
            buf = ""
            if not t:
                return
            if len(t) < PASSAGE_MIN and out and out[-1][0] == head:
                out[-1][1] = (out[-1][1] + " " + t).strip()
                return
            out.append([head, t])

        for p in paras:
            if len(p) > PASSAGE_TARGET:
                flush()
                out.extend(
                    [head, piece] for piece in _chunk_sentences(p, PASSAGE_TARGET)
                )
                continue
            buf = f"{buf} {p}" if buf else p
            if len(buf) >= PASSAGE_MIN:
                flush()
        flush()
    return [(h, t) for h, t in out if len(t) >= PASSAGE_MIN]


def ingest_folder(root: str, max_concepts: int) -> list[dict[str, Any]]:
    """Turn a folder of notes/docs into a concept corpus (one file per concept)."""
    base = Path(root).expanduser().resolve()
    concepts: list[dict[str, Any]] = []
    files = sorted(
        p
        for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() in TEXT_EXTS
        and not any(part in SKIP_DIRS for part in p.parts)
    )
    for p in files:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        title, text = _clean_markdown(raw)
        if len(text) < 30:
            continue  # skip near-empty files
        rel = p.relative_to(base)
        domain = rel.parts[0] if len(rel.parts) > 1 else "note"
        label = title or p.stem.replace("-", " ").replace("_", " ").title()
        # One file used to become one concept, truncated to 600 chars. A long
        # note therefore lost most of its content and mean-pooled its remaining
        # ideas into a single blurred vector. Mirrors splitPassages() in
        # frontend/src/cortex/text.ts so both paths build the same graph.
        passages = _split_passages(raw, title)[:MAX_PASSAGES_PER_DOC]
        for i, (heading, body) in enumerate(passages):
            sub = heading if (heading and heading != title) else None
            concepts.append(
                {
                    "id": _slug(
                        str(rel.with_suffix(""))
                        + (f"-{i}" if len(passages) > 1 else "")
                    ),
                    "label": (sub or label)[:60],
                    "domain": _slug(domain)[:24] or "note",
                    "text": ((sub + ". " if sub else label + ". ") + body)[
                        :PASSAGE_TEXT_CAP
                    ],
                    "source": label[:60],
                }
            )
    if len(concepts) > max_concepts:
        print(
            f"note: {len(concepts)} files found; capping to first {max_concepts}",
            flush=True,
        )
        concepts = concepts[:max_concepts]
    return concepts


def build_map(
    corpus: list[dict[str, Any]],
    name: str,
    k: int,
    radii: str,
    gen_insights: bool = True,
) -> dict[str, Any]:
    n = len(corpus)
    if n < 4:
        raise SystemExit("need at least 4 concepts to build a brain")
    print(f"corpus: {n} concepts", flush=True)

    vecs = []
    for i, c in enumerate(corpus):
        text = c.get("text") or c.get("label") or c["id"]
        vecs.append(embed(text))
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  embedded {i + 1}/{n}", flush=True)
    X = np.asarray(vecs, dtype=np.float64)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # PCA -> 3D layout.
    Xc = Xn - Xn.mean(axis=0)
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt[:3].T
    coords = coords / (coords.std(axis=0) + 1e-9)
    rx, ry, rz = (float(v) for v in radii.split(","))
    coords[:, 0] *= rx
    coords[:, 1] *= ry
    coords[:, 2] *= rz
    rng = np.random.default_rng(7)
    coords += rng.normal(0.0, 1.4, coords.shape)

    # Synapses: kNN + long-range bridges.
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -1.0)
    edges: dict[tuple[int, int], dict[str, Any]] = {}

    def add_edge(a: int, b: int, w: float, long: bool) -> None:
        key = (a, b) if a < b else (b, a)
        cur = edges.get(key)
        if cur is None or w > cur["w"]:
            edges[key] = {
                "s": key[0],
                "t": key[1],
                "w": round(float(w), 4),
                "long": long,
            }

    kk = min(k, n - 1)
    order = np.argsort(-sim, axis=1)
    for i in range(n):
        for j in order[i, :kk]:
            add_edge(i, int(j), sim[i, int(j)], long=False)

    # ---- surprise scoring (mirrors frontend/src/cortex/ingest.ts) -----------
    #
    # A non-obvious connection is one that is genuinely RELATED yet sits in two
    # neighbourhoods that never touch -- two clusters meeting at a single point:
    #
    #     surprise = relatedness x (1 - neighbourhood overlap) x domain bonus
    #
    # all measured in the FULL embedding space. This previously picked source
    # concepts at RANDOM, ranked them by distance in the 3D PCA layout (a
    # projection artifact: a pair lands far apart in 3D precisely when PCA
    # discarded the axis on which they agree), and never sorted the result at
    # all -- despite the docs promising "ranked most-surprising-first".
    nbr = min(12, max(2, n - 1))
    nbr_sets = [set(order[i, :nbr].tolist()) for i in range(n)]
    lo = min(12, max(2, n // 4))  # scale band so small corpora still bridge
    hi = min(60, n - 1)

    cands: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in order[i, lo:hi].tolist():
            key = (min(i, j), max(i, j))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rel = float(sim[i, j])
            if rel <= 0:  # unrelated isn't surprising, it's noise
                continue
            inter = len(nbr_sets[i] & nbr_sets[j])
            union = len(nbr_sets[i]) + len(nbr_sets[j]) - inter
            ov = inter / union if union else 0.0
            cross = corpus[i].get("domain") != corpus[j].get("domain")
            # Two passages of the SAME note are the commonest high-similarity
            # pair once documents are split, and the least interesting: the
            # author already put those ideas side by side, so nothing was
            # discovered. Discounted so they can't crowd out real bridges.
            src_i, src_j = corpus[i].get("source"), corpus[j].get("source")
            same_doc = src_i is not None and src_i == src_j
            cands.append(
                {
                    "i": i,
                    "j": j,
                    "sim": rel,
                    "overlap": ov,
                    "cross": cross,
                    "same_doc": same_doc,
                    "score": rel
                    * (1.0 - ov)
                    * (1.15 if cross else 1.0)
                    * (0.35 if same_doc else 1.0),
                }
            )
    cands.sort(key=lambda c: -c["score"])

    # Diversity guard: without it the single most bridgeable concept takes every
    # slot and the deck is one node over and over.
    max_per, used, bridges = 2, {}, []
    for c in cands:
        if len(bridges) >= max(8, n // 4):
            break
        if used.get(c["i"], 0) >= max_per or used.get(c["j"], 0) >= max_per:
            continue
        used[c["i"]] = used.get(c["i"], 0) + 1
        used[c["j"]] = used.get(c["j"], 0) + 1
        add_edge(c["i"], c["j"], c["sim"], long=True)
        bridges.append(c)

    neurons = [
        {
            "id": c["id"],
            "label": c.get("label", c["id"]),
            "domain": c.get("domain", "note"),
            "x": round(float(coords[i, 0]), 2),
            "y": round(float(coords[i, 1]), 2),
            "z": round(float(coords[i, 2]), 2),
            "snippet": (c.get("text") or "")[:140],
        }
        for i, c in enumerate(corpus)
    ]
    synapses = list(edges.values())

    # Insight cards: precompute (locally) why each long-range bridge is
    # non-obvious. These are what the curiosity engine surfaces as captured
    # output — the toy->tool line.
    insights: list[dict[str, Any]] = []
    if gen_insights:
        # `bridges` is already sorted most-surprising-first, so the emitted cards
        # are too -- and each carries the measurements its claim rests on.
        print(f"generating {len(bridges)} insight cards (local LLM)…", flush=True)
        for idx, c in enumerate(bridges):
            card = llm_insight(
                corpus[c["i"]], corpus[c["j"]], sim=c["sim"], overlap=c["overlap"]
            )
            insights.append(
                {
                    "s": min(c["i"], c["j"]),
                    "t": max(c["i"], c["j"]),
                    "why": card["why"],
                    "angle": card["angle"],
                    "score": round(c["score"], 4),
                    "evidence": {
                        "sim": round(c["sim"], 4),
                        "overlap": round(c["overlap"], 4),
                        "crossDomain": bool(c["cross"]),
                        "sameDocument": bool(c["same_doc"]),
                    },
                }
            )
            if (idx + 1) % 10 == 0 or idx + 1 == len(bridges):
                print(f"  insight {idx + 1}/{len(bridges)}", flush=True)

    return {
        "meta": {
            "name": name,
            "count": n,
            "dim": int(X.shape[1]),
            "k": kk,
            "synapses": len(synapses),
            "insights": len(insights),
        },
        "neurons": neurons,
        "synapses": synapses,
        "insights": insights,
    }


def update_manifest(manifest_path: str, name: str, count: int, synapses: int) -> None:
    p = Path(manifest_path)
    entries: list[dict[str, Any]] = []
    if p.exists():
        try:
            entries = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            entries = []
    entries = [e for e in entries if e.get("name") != name]
    entries.append({"name": name, "label": name, "count": count, "synapses": synapses})
    entries.sort(key=lambda e: e["name"])
    p.write_text(json.dumps(entries, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", help="concept JSON: [{id,label,domain,text},...]")
    src.add_argument(
        "--ingest", help="folder of notes/docs (md/txt) — one file per concept"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="brain")
    ap.add_argument("--k", type=int, default=8, help="synapses per neuron (kNN)")
    ap.add_argument("--radii", default="26,18,21", help="ellipsoid radii x,y,z")
    ap.add_argument("--max", type=int, default=800, help="cap concepts when ingesting")
    ap.add_argument(
        "--manifest", help="brains.json to register this brain for the UI picker"
    )
    ap.add_argument(
        "--no-insights",
        action="store_true",
        help="skip local-LLM insight cards (faster)",
    )
    args = ap.parse_args()

    if args.ingest:
        corpus = ingest_folder(args.ingest, args.max)
    else:
        corpus = json.load(open(args.corpus))

    out = build_map(
        corpus, args.name, args.k, args.radii, gen_insights=not args.no_insights
    )
    json.dump(out, open(args.out, "w"))
    m = out["meta"]
    print(
        f"wrote {args.out}: {m['count']} neurons, {m['synapses']} synapses "
        f"({sum(1 for e in out['synapses'] if e['long'])} long-range)",
        flush=True,
    )
    if args.manifest:
        update_manifest(args.manifest, args.name, m["count"], m["synapses"])
        print(f"registered '{args.name}' in {args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
