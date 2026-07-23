#!/usr/bin/env python3
"""
Tests for taxonomy.py — the free card set + the archetype.

The card prose is shipped text, so it is held to the same structural rules as
the rest of the pipeline: no card may name a model BUILD (the design collapses
model names to a coarse family, and a hardcoded version is both false for anyone
who ran a different one and stale every release), and a card that counts
"smaller model" turns must count only genuinely smaller families.
"""

import re

import taxonomy as T

# A dated build string in shipped prose: "Opus 4.8", "claude-opus-4-8",
# "Haiku 4.5". Matches a family name followed by a version numeral; the ordinary
# "99.8% Opus — nothing but the top model" phrasing has no numeral after the
# name and is left alone.
MODEL_VERSION = re.compile(
    r"\b(claude|opus|sonnet|haiku|gpt|gemini)[\s\-_.]*v?\d", re.I
)

# Fires every card in the library except the single-pass "precision" branch,
# which is mutually exclusive with "craft" on the same metric.
_MAXIMAL = {
    "sessions": 40,
    "assistant_turns": 10000,
    "real_user_prompts": 400,
    "tool_calls": 4000,
    "pounce_median_sec": 3.0,
    "corrections_caught": 60,
    "reversal_count": 24,
    "reversal_rate_per_100": 6.0,
    "read": 600,
    "edit": 800,
    "write": 200,
    "read_to_edit_ratio": 0.8,
    "reread_pct": 32.0,
    "top_tools": {"Bash": 1500, "Edit": 800, "Read": 600},
    "top_mcp_servers": {"chrome": 150, "github": 200},
    "mcp_servers_used": 20,
    "mcp_calls": 700,
    "models": {"opus": 9900, "haiku": 100},
    "opus_pct": 99.0,
    "tokens": {"cache_read": 200_000_000, "output": 1_000_000},
    "distinct_files_edited": 300,
    "most_churned_file_edits": 60,
    "max_turns_in_session": 1000,
    "avg_session_min": 600.0,
    "workflow_calls": 40,
    "agent_calls": 60,
    "todo_calls": 0,
    "hour_histogram": {str(h): (200 if h in (2, 3, 4, 20) else 0) for h in range(24)},
    "weekday_histogram": {str(d): (100 if d in (5, 6) else 40) for d in range(7)},
}

# The single-pass editor — the one card the maximal profile cannot reach.
_PRECISION = dict(
    _MAXIMAL,
    edit=120,
    write=30,
    distinct_files_edited=100,
    most_churned_file_edits=4,
    reread_pct=10.0,
)

_PROFILES = {"maximal": _MAXIMAL, "precision": _PRECISION}

# (category, metric) for every card branch. Fails if a card is added without a
# profile that reaches it — the audit is only worth as much as its coverage.
ALL_CARDS = {
    ("taste", "model_mix"),
    ("reflex", "pounce"),
    ("psyche", "reversals"),
    ("instinct", "read_edit"),
    ("style", "todo"),
    ("automation", "dispatch"),
    ("craft", "edits_per_file"),
    ("psyche", "corrections"),
    ("precision", "edits_per_file"),
    ("devotion", "weekend"),
    ("instinct", "bash"),
    ("operator", "browser"),
    ("tempo", "hours"),
    ("endurance", "max_turns"),
    ("discipline", "reread"),
    ("scale", "cache"),
    ("stack", "mcp"),
}


def test_profile_matrix_reaches_every_card():
    seen = set()
    for m in _PROFILES.values():
        seen |= {(c["category"], c["metric"]) for c in T.build_cards(m)}
    assert seen == ALL_CARDS


def test_no_card_names_a_model_build():
    """The taste card asserted "Opus 4.8" for everyone.

    An operator who ran 4.6 got a card stating 4.8, it goes stale every release,
    and it contradicts the design rule that model names collapse to a coarse
    family (extract.model_family) before anything is emitted.
    """
    for name, m in _PROFILES.items():
        for c in T.build_cards(m):
            for field in ("hero", "title", "sub"):
                assert not MODEL_VERSION.search(str(c[field])), f"{name}: {c[field]}"


def test_no_archetype_prose_names_a_model_build():
    for name, m in _PROFILES.items():
        arch = T.compute_archetype(m)
        assert arch is not None, name
        prose = [arch["name"], arch["tagline"], arch["definition"], *arch["traits"]]
        for s in prose:
            assert not MODEL_VERSION.search(str(s)), f"{name}: {s}"


def test_taste_card_counts_only_genuinely_smaller_families():
    """`other` is the placeholder/unknown-codename bucket (extract collapses
    <synthetic> and unrecognised build strings into it), so counting it as
    "ran on a smaller model" overstates the downshift that actually happened —
    the same inflation report._smaller_models already guards against."""
    import report as R

    m = dict(_MAXIMAL, models={"opus": 9950, "other": 50}, assistant_turns=10000)
    sub = next(c for c in T.build_cards(m) if c["metric"] == "model_mix")["sub"]
    assert R._smaller_models(m) == 0
    assert sub.startswith("0 turns out of 10,000")

    m2 = dict(_MAXIMAL, models={"opus": 9900, "haiku": 60, "sonnet": 40})
    sub2 = next(c for c in T.build_cards(m2) if c["metric"] == "model_mix")["sub"]
    assert sub2.startswith("100 turns out of 10,000")
