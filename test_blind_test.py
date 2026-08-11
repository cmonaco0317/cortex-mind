#!/usr/bin/env python3
"""
Tests for the blind-test harness's power analysis (§2.6).

The harness used to emit "INCONCLUSIVE" at n=20 for a comparison whose design
foreclosed the result: treatment and control were one estimator sampled from two
slices of one sorted list, and the engine arm was a *random* pick from a rank
band, so the score was not in the experiment at all. Two things had to become
true — the arms must be measurably distinct, and the harness must know how many
ratings a verdict would actually require.

An underpowered run reported as "inconclusive" reads as weak evidence of no
effect. It is not: it is no evidence at all, and the harness now says so.
"""

import math

import pytest

pytest.importorskip("numpy")  # blind_test imports numpy
import blind_test as BT  # noqa: E402


def test_binomial_pmf_matches_hand_computed_values():
    assert abs(BT._binom_pmf(0, 4, 0.5) - 0.0625) < 1e-12
    assert abs(BT._binom_pmf(2, 4, 0.5) - 0.375) < 1e-12
    assert abs(sum(BT._binom_pmf(k, 9, 0.37) for k in range(10)) - 1.0) < 1e-12


def test_rejection_region_at_n20_is_the_textbook_one():
    """Two-sided exact binomial against p=0.5 at alpha=0.05, n=20: the region is
    {0..5} u {15..20}, total mass 0.0414. Adding 6 and 14 would take it to
    0.115, past alpha."""
    region = BT._reject_region(20, 0.05)
    assert region == set(range(0, 6)) | set(range(15, 21))
    mass = sum(BT._binom_pmf(k, 20, 0.5) for k in region)
    assert mass <= 0.05
    assert abs(mass - 0.0414) < 5e-4


def test_rejection_region_never_exceeds_alpha():
    for n in (5, 12, 20, 33, 82, 150):
        region = BT._reject_region(n, 0.05)
        assert sum(BT._binom_pmf(k, n, 0.5) for k in region) <= 0.05


def test_required_n_falls_as_the_effect_gets_larger():
    """A bigger effect is easier to detect. If this inverts, the power maths is
    backwards and every n it prints is wrong."""
    ns = [BT.required_n(effect=e) for e in (0.60, 0.65, 0.70, 0.80)]
    assert ns == sorted(ns, reverse=True), ns
    # and the figure the harness actually prints
    assert BT.required_n(effect=0.65, alpha=0.05, power=0.80) == 82


def test_required_n_rises_with_the_demanded_power():
    assert BT.required_n(power=0.80) < BT.required_n(power=0.95)


def test_the_stated_n_really_achieves_the_stated_power():
    """The claim is 'this many decisive ratings gives >=80% chance of detecting a
    65/35 split'. Verify it directly rather than trusting the search, and verify
    one fewer rating does NOT — otherwise the number is not a threshold."""
    n = BT.required_n(effect=0.65, alpha=0.05, power=0.80)
    region = BT._reject_region(n, 0.05)
    achieved = sum(BT._binom_pmf(k, n, 0.65) for k in region)
    assert achieved >= 0.80

    below = BT._reject_region(n - 1, 0.05)
    assert sum(BT._binom_pmf(k, n - 1, 0.65) for k in below) < 0.80


def test_distinctness_report_flags_arms_that_are_the_same_concept():
    """The failure the report exists to catch: if the engine arm and the baseline
    arm resolve to the same concept, the comparison is vacuous."""
    rows = [
        {"source": 0, "baseline": 5, "engine": 9, "engine_cosine_rank": 40},
        {"source": 1, "baseline": 7, "engine": 7, "engine_cosine_rank": 0},
    ]
    d = BT.report_distinctness(rows)
    assert d["arms_identical"] == 1
    assert d["engine_cosine_rank_min"] == 0
    assert d["engine_cosine_rank_max"] == 40


def test_distinctness_report_on_genuinely_separated_arms():
    rows = [
        {"source": i, "baseline": 100 + i, "engine": 200 + i, "engine_cosine_rank": r}
        for i, r in enumerate((17, 31, 47, 62, 79))
    ]
    d = BT.report_distinctness(rows)
    assert d["arms_identical"] == 0
    assert d["engine_cosine_rank_min"] == 17
    assert d["engine_cosine_rank_median"] == 47
    assert d["n_rows"] == 5
