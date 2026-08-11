#!/usr/bin/env python3
"""
Tests for build_brain.py's embedding-derived domain clustering (§2.3).

build_brain.py imports numpy at module load, so these skip where numpy is
absent (the stdlib-only CI job) and run where it is present — locally, and in
the cross-language conformance job §2.4 will add. The mirrored TypeScript
implementation (clusterDomains in frontend/src/cortex/ingest.ts) is exercised by
vitest, which runs in CI regardless; §2.4 locks the two together with a golden
fixture asserted from both languages.
"""

import math

import pytest

np = pytest.importorskip("numpy")  # noqa: F841  (build_brain imports it)
import build_brain as B  # noqa: E402


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _clustered(clusters=4, per=10, dim=8, seed=42):
    """Deterministic well-separated clusters: one-hot centre plus small noise.
    No RNG library, so it is reproducible and matches a TS generator."""
    s = seed

    def jit():
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF - 0.5

    vecs = []
    for k in range(clusters):
        for _ in range(per):
            vecs.append(
                [(1.0 if d % clusters == k else 0.0) + 0.15 * jit() for d in range(dim)]
            )
    return [_norm(v) for v in vecs]


def test_cluster_domains_is_deterministic():
    """No RNG, so the browser path and this pipeline can agree byte for byte."""
    unit = _clustered()
    assert B.cluster_domains(unit) == B.cluster_domains(unit)


def test_cluster_domains_is_non_constant_on_a_flat_corpus():
    """The failure §2.3 names: pasted text and flat corpora have no folders, so
    every concept was domain "note" and crossDomain was uniformly false. Derived
    from the embeddings instead, the labels must vary."""
    unit = _clustered()
    assert len(set(B.cluster_domains(unit))) > 1


def test_cluster_domains_recovers_separated_clusters():
    unit = _clustered(clusters=4, per=10)
    dom = B.cluster_domains(unit)
    for k in range(4):
        block = dom[k * 10 : (k + 1) * 10]
        assert len(set(block)) == 1, "cluster %d was split: %r" % (k, block)
    assert len(set(dom)) == 4


def test_cluster_domains_handles_degenerate_sizes():
    assert B.cluster_domains([]) == []
    assert B.cluster_domains([[1.0, 0.0]]) == [0]
