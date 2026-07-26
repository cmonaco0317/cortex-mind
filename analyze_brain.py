#!/usr/bin/env python3
"""
analyze_brain.py — what each factor of the surprise score actually contributes.

The score is

    score = sim x (1 - overlap) x (0.35 if sameDocument else 1)

and it is only a *ranking* to the extent that each of those terms varies across
the deck. A term that is constant, or that ranges over a hundredth, cannot
reorder anything — it is a scalar wearing the costume of a factor.

crossDomain is reported for context but is NOT a factor: it was a x1.15
multiplier that, once the overlap window was fixed, changed ~6 of 59 selected
pairs and never the top 10 -- cross-cluster pairs ARE the low-overlap pairs, so
it measured, worse, what (1 - overlap) already measures. It is an embedding-
derived cluster label now (§2.3), shown as evidence, not multiplied in.

This script measures that rather than asserting it. For every factor it reports
the spread, the variance, and the fraction sitting on the degenerate value, and
then the one number that says whether the whole formula earns its keep: the
Spearman correlation between the final score and plain cosine similarity. If
that is ~1.0, the score IS cosine, and everything else is decoration.

It is the acceptance instrument for any change to the scoring, so it is
deliberately dependency-free (no numpy) and reads only a committed brain JSON.

    python3 analyze_brain.py [frontend/public/brain-safe.json]
"""

import json
import sys
from pathlib import Path

DEFAULT_BRAIN = Path(__file__).parent / "frontend" / "public" / "brain-safe.json"

SAME_DOC_DISCOUNT = 0.35


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _variance(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / (n - 1)


def _ranks(xs):
    """Ranks with ties averaged, so Spearman is correct on a deck full of 0.0."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho. Returns None when either side is constant (rho undefined)."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((r - ma) ** 2 for r in ra) ** 0.5
    db = sum((r - mb) ** 2 for r in rb) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def describe(name, values, degenerate=None):
    """One line per factor: range, variance, and how much of the deck is inert."""
    lo, hi = min(values), max(values)
    var = _variance(values)
    line = "  %-14s min %8.4f   median %8.4f   max %8.4f   spread %7.4f   var %.3e" % (
        name,
        lo,
        _median(values),
        hi,
        hi - lo,
        var,
    )
    if degenerate is not None:
        stuck = sum(1 for v in values if v == degenerate)
        line += "\n  %-14s %d/%d (%.0f%%) sit at the degenerate value %g" % (
            "",
            stuck,
            len(values),
            100.0 * stuck / len(values),
            degenerate,
        )
    if var == 0.0:
        line += "\n  %-14s CONSTANT — this factor cannot reorder anything" % ""
    print(line)


def analyze(path):
    brain = json.loads(Path(path).read_text(encoding="utf-8"))
    insights = brain.get("insights") or []
    meta = brain.get("meta") or {}
    if not insights:
        print("no insights in %s" % path)
        return 1

    print("brain    : %s" % path)
    print(
        "meta     : %d concepts, %d synapses, %d insights, dim %s"
        % (
            meta.get("count", -1),
            meta.get("synapses", -1),
            len(insights),
            meta.get("dim", "?"),
        )
    )
    print()

    sims = [i["evidence"]["sim"] for i in insights]
    overlaps = [i["evidence"]["overlap"] for i in insights]
    crosses = [bool(i["evidence"]["crossDomain"]) for i in insights]
    scores = [i["score"] for i in insights]
    has_same_doc = all("sameDocument" in i["evidence"] for i in insights)
    same_docs = (
        [bool(i["evidence"]["sameDocument"]) for i in insights]
        if has_same_doc
        else None
    )

    print("FACTORS")
    describe("sim (cosine)", sims)
    describe("overlap", overlaps, degenerate=0.0)
    describe("1 - overlap", [1.0 - o for o in overlaps], degenerate=1.0)

    n_cross = sum(crosses)
    print(
        "  %-14s true for %d/%d (%.0f%%)   informational — not a score factor"
        % ("crossDomain", n_cross, len(crosses), 100.0 * n_cross / len(crosses))
    )
    if same_docs is None:
        print(
            "  %-14s ABSENT from this artifact — build_brain.py emits it, so this\n"
            "  %-14s brain predates the term entirely and it cannot be measured here."
            % ("sameDocument", "")
        )
    else:
        n_same = sum(same_docs)
        print(
            "  %-14s true for %d/%d (%.0f%%)%s"
            % (
                "sameDocument",
                n_same,
                len(same_docs),
                100.0 * n_same / len(same_docs),
                "   CONSTANT — cannot reorder anything"
                if n_same in (0, len(same_docs))
                else "",
            )
        )
    print()
    describe("final score", scores)
    print()

    print("IS THE SCORE JUST COSINE?")
    rho = spearman(scores, sims)
    print(
        "  Spearman(score, cosine) = %s"
        % ("undefined (a side is constant)" if rho is None else "%.4f" % rho)
    )
    for k in (5, 10, 20):
        k = min(k, len(insights))
        by_score = {
            id(x): None for x in sorted(insights, key=lambda i: -i["score"])[:k]
        }
        by_sim = sorted(insights, key=lambda i: -i["evidence"]["sim"])[:k]
        shared = sum(1 for x in by_sim if id(x) in by_score)
        print(
            "  top-%-3d agreement with a plain cosine ranking: %d/%d (%.0f%%)"
            % (k, shared, k, 100.0 * shared / k)
        )
    print()

    print("FORMULA CONSISTENCY (recomputed from the stored factors)")
    worst = 0.0
    for idx, ins in enumerate(insights):
        e = ins["evidence"]
        recomputed = (
            e["sim"]
            * (1.0 - e["overlap"])
            * (SAME_DOC_DISCOUNT if e.get("sameDocument") else 1.0)
        )
        worst = max(worst, abs(recomputed - ins["score"]))
    print("  max |recomputed - stored| = %.2e" % worst)
    print(
        "  %s"
        % (
            "stored scores match the documented formula"
            if worst < 5e-4
            else "STORED SCORES DO NOT MATCH THE DOCUMENTED FORMULA "
            "(this brain predates the current scoring — regenerate it, §2.5)"
        )
    )
    return 0


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT_BRAIN
    return analyze(path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
