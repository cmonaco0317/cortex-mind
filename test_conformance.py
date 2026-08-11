#!/usr/bin/env python3
"""
Cross-language conformance: the Python half (§2.4).

`build_brain.py` and `frontend/src/cortex/` implement the same algorithm twice.
The only contract between them used to be a comment saying "mirrors
splitPassages()", and they had already drifted: the same nested corpus got
different domain labels depending on which pipeline ran it.

`conformance/fixture.json` holds the inputs and `conformance/golden.json` the
expected output. This module asserts the Python implementations against the
golden; `frontend/tests/cortex-conformance.test.ts` asserts the TypeScript ones
against the SAME file. Either side drifting turns CI red.

The golden is regenerated deliberately, never to make a failing test pass — a
diff here means the two languages disagree, or a deliberate algorithm change
that must be applied to both.
"""

import json
import math
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = json.load(open(os.path.join(HERE, "conformance", "fixture.json"), encoding="utf-8"))
GOLDEN = json.load(open(os.path.join(HERE, "conformance", "golden.json"), encoding="utf-8"))

np = pytest.importorskip("numpy")  # noqa: F841  (build_brain imports it)
import build_brain as B  # noqa: E402


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_folder_domain_matches_the_golden():
    """The divergence §2.4 names: this used the top-level folder, the browser used
    the immediate parent, with different caps and different empty-name fallbacks."""
    got = [B.folder_domain(p) for p in FIXTURE["paths"]]
    assert got == GOLDEN["folderDomains"]


def test_passage_splitting_matches_the_golden():
    got = []
    for d in FIXTURE["docs"]:
        title, _ = B._clean_markdown(d["text"])
        got.append(
            [{"heading": h or "", "text": t} for h, t in B._split_passages(d["text"], title)]
        )
    assert got == GOLDEN["passages"]


def test_domain_clustering_matches_the_golden():
    got = B.cluster_domains([_norm(v) for v in FIXTURE["vectors"]])
    assert got == GOLDEN["clusters"]


def test_bridge_ranking_matches_the_golden():
    """The whole scoring path — candidate selection, overlap, the same-note
    discount, the sort, and the diversity guard — on fixed vectors."""
    by_text = {c["text"]: v for c, v in zip(FIXTURE["concepts"], FIXTURE["vectors"])}
    orig_embed, orig_llm = B.embed, B.llm_insight
    try:
        B.embed = lambda t: by_text[t]
        B.llm_insight = lambda a, b, sim=None, overlap=None, cross=False: {
            "why": "",
            "angle": "",
        }
        brain = B.build_map(FIXTURE["concepts"], "conformance", 8, "26,18,21", gen_insights=True)
    finally:
        B.embed, B.llm_insight = orig_embed, orig_llm

    got = [
        {
            "s": i["s"],
            "t": i["t"],
            "score": i["score"],
            "sim": i["evidence"]["sim"],
            "overlap": i["evidence"]["overlap"],
            "crossDomain": i["evidence"]["crossDomain"],
            "sameDocument": i["evidence"]["sameDocument"],
        }
        for i in brain["insights"]
    ]
    assert got == GOLDEN["bridges"]


def test_the_golden_is_not_vacuous():
    """A golden that encodes an empty or degenerate result would pass forever
    while proving nothing. Each of these is a property the golden must keep, so
    a regeneration that quietly flattens it fails here instead of sailing through.

    NOT asserted: that the same-note discount fires. It does not in this fixture
    — 0 of 30 bridges have sameDocument set, because two passages of one document
    are among each other's NEAREST neighbours and the candidate band starts below
    that rank, so they are filtered out before scoring. The discount is exercised
    by the vitest formula test instead; pretending otherwise here would be a
    tautology dressed as coverage.
    """
    assert len(GOLDEN["folderDomains"]) >= 5
    assert len(set(GOLDEN["folderDomains"])) > 1, "every path slugged the same way"
    assert sum(len(p) for p in GOLDEN["passages"]) >= 6
    assert any(len(p) > 1 for p in GOLDEN["passages"]), "nothing actually split"
    assert len(set(GOLDEN["clusters"])) > 1, "clustering collapsed to one group"
    assert len(GOLDEN["bridges"]) >= 10

    scores = [b["score"] for b in GOLDEN["bridges"]]
    assert scores == sorted(scores, reverse=True), "bridges are not ranked"
    overlaps = {b["overlap"] for b in GOLDEN["bridges"]}
    assert len(overlaps) > 1, "overlap is constant — the §2.2 regression is back"
    crosses = {b["crossDomain"] for b in GOLDEN["bridges"]}
    assert len(crosses) > 1, "crossDomain is constant across every bridge"
