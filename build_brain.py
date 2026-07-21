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


def _fallback_insight(a: dict[str, Any], b: dict[str, Any]) -> dict[str, str]:
    da, db = a.get("domain", "?"), b.get("domain", "?")
    bridge = f"across {da} and {db}" if da != db else f"within {da}"
    return {
        "why": f"'{a['label']}' and '{b['label']}' rarely appear together, yet the link sits {bridge}.",
        "angle": f"What would '{a['label']}' look like if reframed through '{b['label']}'?",
    }


def llm_insight(a: dict[str, Any], b: dict[str, Any]) -> dict[str, str]:
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
        fb = _fallback_insight(a, b)
        return {"why": why or fb["why"], "angle": angle or fb["angle"]}
    except (OSError, ValueError, KeyError):
        return _fallback_insight(a, b)


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
        concepts.append(
            {
                "id": _slug(str(rel.with_suffix(""))),
                "label": label[:60],
                "domain": _slug(domain)[:24] or "note",
                "text": (label + ". " + text)[:600],
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

    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    lo = min(12, max(2, n // 4))  # scale band so small corpora still bridge
    for i in rng.choice(n, size=max(8, n // 4), replace=False):
        cand = order[i, lo : min(60, n - 1)]
        if len(cand) == 0:
            continue
        j = int(cand[int(np.argmax(dist[i, cand]))])
        add_edge(int(i), j, sim[i, j], long=True)

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
        longs = [e for e in synapses if e["long"]]
        print(f"generating {len(longs)} insight cards (local LLM)…", flush=True)
        for idx, e in enumerate(longs):
            card = llm_insight(corpus[e["s"]], corpus[e["t"]])
            insights.append(
                {"s": e["s"], "t": e["t"], "why": card["why"], "angle": card["angle"]}
            )
            if (idx + 1) % 10 == 0 or idx + 1 == len(longs):
                print(f"  insight {idx + 1}/{len(longs)}", flush=True)

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
