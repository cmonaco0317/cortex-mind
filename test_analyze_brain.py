#!/usr/bin/env python3
"""
Tests for analyze_brain.py — the measurement instrument for the surprise score.

The instrument is the acceptance criterion for every later change to the
scoring, so it has to be trustworthy before it is trusted. Two things matter:
that Spearman is right on inputs whose answer is known independently (including
the tie-heavy case, since 92% of the shipped deck's overlap values are exactly
0.0 and naive ranking gets that wrong), and that a degenerate factor is actually
reported as degenerate rather than quietly averaged away.
"""

import json

import analyze_brain as A


def test_spearman_matches_hand_computable_cases():
    # rho is a float sum, so it lands within rounding of the exact answer rather
    # than on it; 1e-12 is far tighter than any ranking defect could hide in.
    assert abs(A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12
    assert abs(A.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-12
    # monotone but non-linear: Spearman is 1.0 where Pearson would not be
    assert abs(A.spearman([1, 2, 3, 4], [1, 4, 9, 16]) - 1.0) < 1e-12


def test_spearman_averages_ties_rather_than_ordering_them_arbitrarily():
    """54 of 59 shipped overlap values are exactly 0.0. Ranking ties by input
    order instead of averaging them invents a correlation that isn't there."""
    rho = A.spearman([0.0, 0.0, 0.0, 1.0], [5.0, 5.0, 5.0, 9.0])
    assert abs(rho - 1.0) < 1e-12
    # a constant side has no rank variance, so rho is undefined, not 0.0
    assert A.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_variance_and_median_on_a_known_sample():
    assert A._median([3, 1, 2]) == 2
    assert A._median([4, 1, 3, 2]) == 2.5
    assert abs(A._variance([2, 4, 4, 4, 5, 5, 7, 9]) - 4.5714285714) < 1e-9
    assert A._variance([7]) == 0.0
    assert A._variance([7, 7, 7]) == 0.0


def test_it_runs_on_the_shipped_brain(capsys):
    """The committed artifact must stay readable by the instrument — this is
    what makes the recorded baseline reproducible rather than a one-off."""
    assert A.analyze(A.DEFAULT_BRAIN) == 0
    out = capsys.readouterr().out
    for expected in (
        "sim (cosine)",
        "overlap",
        "crossDomain",
        "final score",
        "Spearman(score, cosine)",
        "FORMULA CONSISTENCY",
    ):
        assert expected in out, "instrument stopped reporting %r" % expected


def test_a_constant_factor_is_called_out_as_unable_to_rank(tmp_path, capsys):
    """The failure this whole instrument exists to surface: a factor that is the
    same for every insight is decoration, and must be named as such. overlap is
    the one that carries this risk now that crossDomain is informational."""
    brain = {
        "meta": {"count": 2, "synapses": 1, "insights": 2, "dim": 8},
        "insights": [
            {
                "s": 0,
                "t": 1,
                "score": round(0.5 * 1.0, 4),
                "evidence": {"sim": 0.5, "overlap": 0.0, "crossDomain": True},
            },
            {
                "s": 0,
                "t": 1,
                "score": round(0.9 * 1.0, 4),
                "evidence": {"sim": 0.9, "overlap": 0.0, "crossDomain": True},
            },
        ],
    }
    p = tmp_path / "brain-constant.json"
    p.write_text(json.dumps(brain), encoding="utf-8")

    assert A.analyze(p) == 0
    out = capsys.readouterr().out
    # overlap constant at its degenerate value must be called out
    assert "CONSTANT" in out, "a constant overlap was not reported as constant"
    assert "sit at the degenerate value" in out
    # with only cosine varying, the score IS cosine and the instrument must say so
    assert "Spearman(score, cosine) = 1.0000" in out
    # crossDomain is context now, not a factor
    assert "informational — not a score factor" in out


def test_it_reports_when_stored_scores_contradict_the_documented_formula(
    tmp_path, capsys
):
    """A brain whose scores were produced by a different formula than the one
    documented must not pass silently — that is how a scoring change ships
    without anyone noticing the artifact went stale."""
    brain = {
        "meta": {"count": 2, "synapses": 1, "insights": 1, "dim": 8},
        "insights": [
            {
                "s": 0,
                "t": 1,
                "score": 0.1234,  # unrelated to sim x (1 - overlap)
                "evidence": {"sim": 0.9, "overlap": 0.0, "crossDomain": True},
            }
        ],
    }
    p = tmp_path / "brain-inconsistent.json"
    p.write_text(json.dumps(brain), encoding="utf-8")

    assert A.analyze(p) == 0
    out = capsys.readouterr().out
    assert "STORED SCORES DO NOT MATCH THE DOCUMENTED FORMULA" in out
