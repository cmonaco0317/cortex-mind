#!/usr/bin/env python3
"""
Tests for report.py — the Operator Report engine (Phase 1 paid tier).

These encode the LOCKED design guarantees from OPERATOR-REPORT.md as PROPERTIES,
not one golden case + denylists (an adversarial review showed the first cut was
gameable). In particular:
  - taxes fire on signal PAIRS: each single conjunct alone must NOT fire the tax
  - honesty is STRUCTURAL: no outcome-quality verdicts, no cross-user comparisons,
    no "wrong turn"-as-reality — checked by pattern, not a fixed phrase list
  - honest attribution: smaller-model count excludes placeholder buckets; no
    false "zero todos"; the parallel/chunking-artifact tax does not exist
  - no broken numbers ship (the "1-in-0" bug); numbers are actually the real values
  - the diversity guard is load-bearing across edges AND taxes
  - the archetype never ships from absent/thin data (Night Builder guard)
  - privacy tripwire is scoped (benign slashes render; real paths/emails crash)
"""

import json
import re

import pytest

import report as R

# A SYNTHETIC "The Pouncer" profile — fictional numbers chosen to exercise every
# Pouncer rule (no real user's data). Un-collapsed model keys are intentional: they
# test that _smaller_models is robust to raw build strings, not just collapsed families.
POUNCER = {
    "sessions": 40,
    "assistant_turns": 10000,
    "real_user_prompts": 400,
    "pounce_median_sec": 3.0,
    "corrections_caught": 60,
    "reversal_count": 24,
    "reversal_rate_per_100": 6.0,
    "tool_calls": 4000,
    "top_tools": {"Bash": 1500, "Edit": 800, "Read": 600, "TaskUpdate": 50},
    "read": 600,
    "edit": 800,
    "write": 200,
    "read_to_edit_ratio": 0.8,
    "reread_pct": 32.0,
    "bash": 1500,
    "opus_pct": 99.8,
    "models": {
        "claude-opus-4-8": 9980,
        "<synthetic>": 8,
        "claude-haiku-4-5-20251001": 20,
    },
    "mcp_servers_used": 20,
    "mcp_calls": 700,
    "distinct_files_edited": 300,
    "most_churned_file_edits": 60,
    "max_turns_in_session": 1000,
    "avg_session_min": 600.0,
    "workflow_calls": 40,
    "agent_calls": 60,
    "todo_calls": 0,
    "max_parallel_tools": 1,
    "hour_histogram": {str(h): (200 if h in (2, 3, 4, 20) else 0) for h in range(24)},
}

EMPTY = {"sessions": 0}

# Structural honesty guards (patterns, not a fixed phrase denylist).
CROSS_USER = re.compile(
    r"\b(most|other|average)\s+(people|operators|users)\b|\bthan (most|other|everyone)\b|\bunlike\b|percentile|% of users",
    re.I,
)
OUTCOME_VERDICT = re.compile(
    r"\bwrong (turn|path|way)\b|went wrong|never (lets|loses|misses)\b|\bit works\b|always works|\boptimal\b|\bmistake\b|you should have|best practice|doing it wrong",
    re.I,
)
BROKEN_NUM = re.compile(r"1[- ]in[- ]0\b|1[- ]in[- ]1\b|\bNaN\b|\bundefined\b")


def _items(rep):
    return rep["edges"] + rep["taxes"] + rep["moves"]


def _blob(rep):
    return " ".join(it["title"] + " " + it["body"] for it in _items(rep))


# ------------------------------------------------------------------ archetype
def test_archetype_is_the_pouncer():
    assert R.build_report(POUNCER)["archetype"]["name"] == "The Pouncer"


def test_archetype_measures_style_not_usage_volume():
    """An archetype must describe HOW you work, not HOW MUCH.

    The thresholds were absolute counts (corrections_caught >= 40), so anyone
    with enough sessions became "The Pouncer" by accumulation — the label was a
    proxy for usage volume. These two profiles invert under the fix: the heavy
    user who rarely corrects is no longer a Pouncer, and the light user who
    corrects constantly now is.
    """
    import taxonomy as T

    base = dict(
        reversal_rate_per_100=6.0,
        reversal_count=24,
        read_to_edit_ratio=0.8,
        tool_calls=400,
        top_tools={"Bash": 100},
        edit=100,
        distinct_files_edited=50,
        reread_pct=32.0,
        opus_pct=99.0,
        workflow_calls=2,
        agent_calls=2,
        hour_histogram={str(h): (50 if h in (2, 3, 4, 20) else 0) for h in range(24)},
    )
    # 60 corrections looks like a lot, but it's 3 per 100 prompts — not hovering.
    heavy_low_rate = dict(
        base, assistant_turns=10000, real_user_prompts=2000, corrections_caught=60
    )
    # 25 corrections looks like few, but it's 25 per 100 prompts — constant hovering.
    light_high_rate = dict(
        base, assistant_turns=500, real_user_prompts=100, corrections_caught=25
    )

    assert heavy_low_rate["corrections_caught"] >= 40  # the OLD rule would fire
    assert T.compute_archetype(heavy_low_rate)["name"] != "The Pouncer"

    assert light_high_rate["corrections_caught"] < 40  # the OLD rule would NOT fire
    assert T.compute_archetype(light_high_rate)["name"] == "The Pouncer"


def test_archetype_needs_a_minimum_sample():
    """Absolute counts survive only as sample-size floors. Three prompts is not
    a personality, however lopsided the ratio."""
    import taxonomy as T

    tiny = dict(
        assistant_turns=8,
        real_user_prompts=4,
        corrections_caught=4,  # 100 per 100 prompts, but n=4
        reversal_rate_per_100=6.0,
        reversal_count=1,
        read_to_edit_ratio=0.8,
        tool_calls=10,
        top_tools={"Bash": 2},
        edit=4,
        distinct_files_edited=2,
        reread_pct=10.0,
        opus_pct=99.0,
        hour_histogram={"14": 8},
    )
    arch = T.compute_archetype(tiny)
    assert arch is None or arch["name"] != "The Pouncer"


def test_archetype_dropped_on_thin_data():
    assert R.build_report(EMPTY)["archetype"] is None


def test_night_builder_not_asserted_for_afternoon_operator():
    # before_1pm==0 (peak at 3pm=hour 15) but zero nocturnal activity: taxonomy
    # would label "The Night Builder"; the report must refuse the false headline.
    m = dict(
        EMPTY,
        sessions=6,
        assistant_turns=1000,
        hour_histogram={str(h): (300 if h == 15 else 0) for h in range(24)},
    )
    arch = R.build_report(m)["archetype"]
    assert arch is None or arch["name"] != "The Night Builder"


# ------------------------------------------------------------------ edges
def test_edges_fire_deduped_and_capped():
    rep = R.build_report(POUNCER)
    assert 1 <= len(rep["edges"]) <= R.MAX_EDGES
    fams = [e["family"] for e in rep["edges"]]
    assert len(fams) == len(set(fams))
    assert any(e["family"] == "reflex" for e in rep["edges"])
    assert any(e["family"] == "dispatch" for e in rep["edges"])


# ------------------------------------------------------------------ taxes: pairs
def test_taxes_fire_deduped_and_capped():
    rep = R.build_report(POUNCER)
    assert 1 <= len(rep["taxes"]) <= R.MAX_TAXES
    fams = [t["family"] for t in rep["taxes"]]
    assert len(fams) == len(set(fams))
    keys = {t["key"] for t in rep["taxes"]}
    assert "steer" in keys
    assert "onegear" in keys


def _tax_keys(m):
    return {t["key"] for t in R.build_report(m)["taxes"]}


def test_steer_tax_needs_both_legs_not_one():
    base = dict(EMPTY, assistant_turns=1000)
    # only edit-first (no corrections) -> off
    assert "steer" not in _tax_keys(
        dict(base, read_to_edit_ratio=0.5, corrections_caught=0)
    )
    # only corrections (reads-first) -> off
    assert "steer" not in _tax_keys(
        dict(base, read_to_edit_ratio=3.0, corrections_caught=99)
    )
    # both -> on
    assert "steer" in _tax_keys(
        dict(base, read_to_edit_ratio=0.5, corrections_caught=99)
    )


def test_onegear_tax_needs_dominance_and_a_near_zero_alternative():
    base = dict(EMPTY, assistant_turns=1000)
    # high opus BUT a real downshift sample -> off (they have the data)
    assert "onegear" not in _tax_keys(
        dict(base, opus_pct=99.0, models={"opus": 700, "sonnet": 300})
    )
    # high opus AND near-zero smaller sample -> on
    assert "onegear" in _tax_keys(
        dict(base, opus_pct=99.9, models={"opus": 998, "haiku": 2})
    )


def test_strongest_one_gear_signal_ships_the_tax_and_its_move():
    """The tax was INVERTED: it could not fire for the operator it most describes.

    100% Opus with zero smaller-model turns is the strongest possible form of the
    signal. That body renders the zero as a word ("none of your turns have ever
    run on a smaller model") — no numeral — and the old personalized-or-nothing
    gate regex-scanned rendered prose for a digit, so it silently dropped the tax.
    Because _moves() only keeps moves whose key is in `fired`, the move went with
    it. A weaker 99%/4-turn profile shipped both. Both directions are asserted
    here so the inversion can't come back.
    """
    strongest = dict(
        EMPTY,
        assistant_turns=400,
        opus_pct=100.0,
        models={"opus": 400},
        read_to_edit_ratio=1.0,
    )
    weaker = dict(strongest, opus_pct=99.0, models={"opus": 396, "haiku": 4})

    assert R._smaller_models(strongest) == 0
    assert R._smaller_models(weaker) == 4

    for label, m in (("0 smaller-model turns", strongest), ("4 turns", weaker)):
        rep = R.build_report(m)
        assert "onegear" in {t["key"] for t in rep["taxes"]}, label
        assert "onegear" in {mv["key"] for mv in rep["moves"]}, label

    # The exact cause, pinned: this body ships even though it contains no digit.
    body = next(t for t in R.build_report(strongest)["taxes"] if t["key"] == "onegear")[
        "body"
    ]
    assert not R._has_number(body)


# --------------------------------------------------- attribution / honesty
def test_smaller_models_excludes_placeholder_and_codename_buckets():
    # <synthetic> and an unknown codename ('other' family) are NOT smaller models.
    m = {
        "models": {
            "claude-opus-4-8": 100,
            "<synthetic>": 50,
            "claude-fable-5": 30,
            "claude-haiku-4-5": 2,
        }
    }
    assert R._smaller_models(m) == 2


def test_no_false_zero_todos_claim():
    # TaskUpdate=67 is present; the report must never claim the operator plans
    # nothing. (Advising "not a todo list" as a format is fine — that's not a claim
    # about their behavior.)
    blob = _blob(R.build_report(POUNCER)).lower()
    for false_claim in (
        "zero todos",
        "no todos",
        "todos ever",
        "not a single todo",
        "never make a list",
    ):
        assert false_claim not in blob


def test_no_parallel_or_chunking_artifact_tax():
    # max_parallel_tools is a transcript/model property, not an operator choice.
    keys = {t["key"] for t in R.build_report(POUNCER)["taxes"]}
    assert "parallel" not in keys
    assert "1 at a time" not in _blob(R.build_report(POUNCER)).lower()


def test_every_body_cites_a_number():
    # A profile whose numbers all render as numerals. NOT the shipping gate —
    # see test_every_template_declares_and_renders_the_numbers_it_cites.
    for it in _items(R.build_report(POUNCER)):
        assert R._has_number(it["body"]), it["title"]


def test_bodies_cite_the_real_metric_values():
    # Not just "a digit exists" — the actual computed values must appear.
    blob = _blob(R.build_report(POUNCER))
    for token in ("60", "0.8", "1,000", "100", "300"):
        assert token in blob, token


# Every rule in the library, and every prose BRANCH inside a rule, must be
# exercised below. The one-gear bug lived in a branch no profile ever rendered,
# so a per-function unit test could not see it.
EDGE_FAMILIES = {
    "reflex",
    "dispatch",
    "endurance",
    "bash",
    "churn",
    "reread",
    "planning",
}
TAX_KEYS = {"steer", "onegear", "churn"}

# A cited zero is allowed to render as the word that stands in for it — that
# readability choice is exactly what the old numeral scan punished.
_ZERO_WORDS = ("none", "not one", "zero", "never", "no ")

_SURGEON = dict(
    EMPTY,
    sessions=20,
    assistant_turns=2000,
    distinct_files_edited=100,
    edit=120,
    write=30,
    reread_pct=10.0,
    read_to_edit_ratio=2.0,
    models={"opus": 1900, "haiku": 100},
    opus_pct=95.0,
)

_ALL_OPUS = dict(
    EMPTY,
    sessions=10,
    assistant_turns=400,
    opus_pct=100.0,
    models={"opus": 400},
    read_to_edit_ratio=1.0,
)

_NO_REVERSALS = dict(
    EMPTY,
    assistant_turns=1000,
    read_to_edit_ratio=0.8,
    corrections_caught=40,
    reversal_rate_per_100=0.0,
)

# name -> profile. Collectively these must fire every rule; the coverage
# assertions below fail if a new rule is added without one.
_RULE_COVERAGE = {
    "pouncer": POUNCER,
    "surgeon": _SURGEON,  # the churn EDGE (low edits-per-file)
    "all_opus": _ALL_OPUS,  # the zero-smaller-model branch (renders as a word)
    "no_reversals": _NO_REVERSALS,  # the steer move without its reversal clause
}


def _renders(v, body: str) -> bool:
    """The cited value actually appears in the prose it was handed to."""
    f = float(v)
    if f == 0 and any(w in body.lower() for w in _ZERO_WORDS):
        return True
    forms = {str(v), str(round(f, 1))}
    if f.is_integer():
        forms.add(R.fmt(int(f)))
    return any(s in body for s in forms)


def test_every_template_declares_and_renders_the_numbers_it_cites():
    """Personalized-or-nothing, audited across EVERY branch of the library.

    Two directions, because the bug needed both: a template must declare the
    metric values it was handed (so the gate can check inputs rather than
    scan prose), and every declared value must actually reach the body (so
    declaring a cite can't become a way to smuggle an unpersonalized template
    past the gate).
    """
    seen_edges: set[str] = set()
    seen_taxes: set[str] = set()
    seen_moves: set[str] = set()

    for name, m in _RULE_COVERAGE.items():
        fired = {t["key"] for t in R._taxes(m)}
        items = (
            [("edge", e, seen_edges) for e in R._edges(m)]
            + [("tax", t, seen_taxes) for t in R._taxes(m)]
            + [("move", mv, seen_moves) for mv in R._moves(m) if mv["key"] in fired]
        )
        assert items, name
        for kind, it, bucket in items:
            where = f"{name}/{kind}/{it['title']}"
            assert R._personalized(it), where
            for v in it["cites"]:
                assert _renders(v, it["body"]), f"{where}: {v!r} missing from body"
            bucket.add(it.get("key") or it["family"])

    assert seen_edges == EDGE_FAMILIES
    assert seen_taxes == TAX_KEYS
    assert seen_moves == TAX_KEYS


def test_an_undeclared_template_cannot_ship():
    # The gate is structural, so it must reject a body that cites nothing —
    # including one stuffed with digits that came from somewhere else.
    assert not R._personalized({"body": "42 of 99 turns", "cites": ()})
    assert not R._personalized({"body": "42 of 99 turns"})
    assert not R._personalized({"body": "x", "cites": (None,)})
    assert not R._personalized({"body": "x", "cites": (3, float("nan"))})
    assert R._personalized({"body": "x", "cites": (0,)})  # zero is a real value


def test_no_broken_or_placeholder_numbers():
    # Fire the steer tax with zero reversals -> the move must not print "1-in-0".
    m = dict(
        EMPTY,
        assistant_turns=1000,
        read_to_edit_ratio=0.8,
        corrections_caught=40,
        reversal_rate_per_100=0.0,
    )
    rep = R.build_report(m)
    assert "steer" in {t["key"] for t in rep["taxes"]}
    assert not BROKEN_NUM.search(_blob(rep)), _blob(rep)


def test_no_cross_user_comparison_structural():
    assert not CROSS_USER.search(_blob(R.build_report(POUNCER)))


def test_no_outcome_quality_verdict_structural():
    assert not OUTCOME_VERDICT.search(_blob(R.build_report(POUNCER)))


def test_every_tax_reads_as_a_tradeoff_not_a_scold():
    # Trap 1, enforced mechanically: every tax body names the choice/tradeoff.
    markers = (
        "tradeoff",
        "preference",
        "fine default",
        "on purpose",
        "cost is",
        "cost of",
    )
    for t in R.build_report(POUNCER)["taxes"]:
        assert any(mk in t["body"].lower() for mk in markers), t["title"]


# ------------------------------------------------------------------ moves
def test_moves_map_only_to_fired_taxes():
    rep = R.build_report(POUNCER)
    tax_keys = {t["key"] for t in rep["taxes"]}
    move_keys = {mv["key"] for mv in rep["moves"]}
    assert move_keys == tax_keys


def test_move_success_metric_is_fewer_wrong_starts_not_fewer_corrections():
    # Coherence: the reflex edge says "lean in"; the steer move must NOT tell them
    # to shrink the very metric the edge praises.
    move = next(mv for mv in R.build_report(POUNCER)["moves"] if mv["key"] == "steer")
    assert (
        "flagged in the first place" in move["body"] or "moved earlier" in move["body"]
    )


# ------------------------------------------------------------------ footer
def test_footer_always_present():
    assert "hypothesis" in R.build_report(POUNCER)["footer"].lower()
    assert R.build_report(EMPTY)["footer"]


# ------------------------------------------------------------------ diversity guard
def test_diversify_is_load_bearing_across_edge_and_tax():
    edges = [{"family": "shared", "priority": 90, "title": "E", "body": "1"}]
    taxes = [
        {"family": "shared", "priority": 60, "key": "k", "title": "T", "body": "2"}
    ]
    e, t = R._diversify(edges, taxes)
    assert len(e) == 1 and len(t) == 0  # higher-priority edge keeps the family


# ------------------------------------------------------------------ privacy tripwire
def test_privacy_tripwire_catches_real_leaks():
    with pytest.raises(SystemExit):
        R._audit({"body": "/Users/someone/secret"})
    with pytest.raises(SystemExit):
        R._audit({"body": "reach me at name@example.com"})
    with pytest.raises(SystemExit):
        R._audit({"edges": [{"body": "peek at /home/user/keys"}]})  # nested


def test_privacy_tripwire_is_scoped_to_real_leaks():
    # Benign prose with slashes / at-signs must render, not crash.
    R._audit({"body": "read/write, CI/CD, and/or an @mention are fine"})


def test_build_report_runs_the_tripwire():
    R.build_report(POUNCER)  # _audit runs internally; must not raise on clean prose


# ------------------------------------------------------------------ render
def test_low_signal_input_is_graceful():
    rep = R.build_report(EMPTY)
    assert rep["edges"] == [] and rep["taxes"] == [] and rep["moves"] == []
    assert "OPERATOR REPORT" in R.render_html(rep, EMPTY)


def test_render_is_self_contained_and_complete():
    rep = R.build_report(POUNCER)
    html = R.render_html(rep, POUNCER)
    assert "The Pouncer" in html
    for it in rep["edges"] + rep["taxes"]:
        assert R._esc(it["title"]) in html
    assert rep["footer"][:40] in html
    assert "<script src" not in html
    assert "<link " not in html
    assert "cdn" not in html.lower()
    assert html.count("</script>") == 1  # no data-injected early close


def test_build_is_deterministic():
    a = json.dumps(R.build_report(POUNCER), sort_keys=True, ensure_ascii=False)
    b = json.dumps(R.build_report(POUNCER), sort_keys=True, ensure_ascii=False)
    assert a == b


# ------------------------------------------------------------------ breadth
# The engine must produce a coherent report for operators who are NOT the sample —
# a thin/empty report for anyone-but-a-Pouncer was the real product risk.
_MODERATE = dict(
    sessions=30,
    assistant_turns=4000,
    tool_calls=2500,
    top_tools={},
    read=300,
    edit=400,
    write=100,
    read_to_edit_ratio=0.75,
    reread_pct=12,
    bash=200,
    opus_pct=99.6,
    models={"opus": 3984, "haiku": 16},
    distinct_files_edited=120,
    most_churned_file_edits=10,
    max_turns_in_session=350,
    workflow_calls=5,
    agent_calls=10,
    todo_calls=0,
    corrections_caught=22,
    pounce_median_sec=6.0,
    reversal_rate_per_100=3.0,
    hour_histogram={str(h): (150 if 9 <= h <= 18 else 0) for h in range(24)},
)


def test_non_pouncer_operators_get_a_complete_report():
    profiles = {
        "director": dict(
            _MODERATE,
            workflow_calls=60,
            agent_calls=70,
            read_to_edit_ratio=1.4,
            corrections_caught=3,
        ),
        "night": dict(
            _MODERATE,
            hour_histogram={str(h): (300 if h in (2, 3, 4) else 0) for h in range(24)},
            corrections_caught=3,
            read_to_edit_ratio=1.5,
        ),
        "terminal": dict(
            _MODERATE,
            bash=1200,
            top_tools={"Bash": 1200},
            corrections_caught=3,
            read_to_edit_ratio=1.4,
        ),
    }
    for name, m in profiles.items():
        rep = R.build_report(m)
        assert rep["archetype"] is not None, name
        assert len(rep["edges"]) >= 1, name  # not a header-only report
        assert len(rep["taxes"]) >= 1, name  # heavy-Opus users hit one-gear at least
        assert {mv["key"] for mv in rep["moves"]} == {
            t["key"] for t in rep["taxes"]
        }, name


def test_planning_edge_fires_for_a_planner():
    m = dict(
        _MODERATE,
        read_to_edit_ratio=1.6,
        corrections_caught=3,
        top_tools={"TaskUpdate": 40},
    )
    fams = [e["family"] for e in R.build_report(m)["edges"]]
    assert "planning" in fams


def test_planning_edge_yields_to_the_steer_tax_same_family():
    # The sample profile has 50 TaskUpdate AND fires the steer tax; the higher-priority
    # win the 'planning' family so the report never both praises and taxes it.
    rep = R.build_report(POUNCER)
    assert any(t["key"] == "steer" for t in rep["taxes"])
    assert not any(e["family"] == "planning" for e in rep["edges"])


def test_planning_edge_excludes_bare_task_dispatch():
    # The bare `Task` tool is subagent DISPATCH (already counted as agent_calls),
    # NOT planning — it must not fire the planning edge or double-count.
    m = dict(
        _MODERATE,
        todo_calls=0,
        top_tools={"Task": 80},
        read_to_edit_ratio=1.6,
        corrections_caught=3,
    )
    assert R._planning_calls(m) == 0
    assert not any(e["family"] == "planning" for e in R.build_report(m)["edges"])


def test_balanced_operator_gets_an_honest_no_tax_note():
    # a genuine model-router with no strong tax: honest empty-state, not a blank,
    # and scoped to what the report actually checks (no false "clean bill").
    m = dict(
        _MODERATE,
        models={"opus": 3000, "sonnet": 1000},
        read_to_edit_ratio=1.4,
        corrections_caught=3,
        most_churned_file_edits=6,
    )
    rep = R.build_report(m)
    assert rep["taxes"] == []
    html = R.render_html(rep, m)
    assert "balanced on the things it measures" in html


def test_insufficient_data_renders_an_honest_banner():
    html = R.render_html(R.build_report(EMPTY), EMPTY)
    assert "NOT ENOUGH YET" in html or "enough sessions" in html.lower()


def test_privacy_tripwire_blocks_every_secret_shape():
    """The tripwire is the only thing standing between a session log and a
    shareable card, so it gets an explicit matrix rather than a spot check.

    `sk-ant-api03-…` is first on purpose: the original pattern was
    sk-[A-Za-z0-9]{20,}, which does NOT match a real Anthropic key — the run of
    alphanumerics after "sk-" is broken by a hyphen after three characters. The
    single most likely secret to appear in a Claude Code log was the one shape
    the tripwire let through.
    """
    import extract as E

    # Assembled at runtime, not written as literals. A credential scanner
    # matches on SHAPE, so it can't tell a real token from a synthetic one of
    # the same shape — and a fixture without the real shape wouldn't test the
    # tripwire properly. Concatenating keeps each runtime string realistic while
    # leaving nothing matchable in the source, so a gitleaks run on this repo
    # reports real findings instead of its own test data.
    A = "A" * 28
    leaky = [
        "sk-" + "ant-api03-" + A,
        "sk-" + "proj-" + A,
        "sk-" + A,
        "sk_" + "live_" + "A" * 20,
        "rk_" + "live_" + "A" * 20,
        "ghp_" + "A" * 36,
        "github_" + "pat_" + "A" * 24,
        "AKIA" + "IOSFODNN7EXAMPLE",
        "AIza" + "Sy" + "A" * 33,
        "xoxb-" + "1234567890-abcdefghij",
        "eyJhbGciOiJI." + "eyJzdWIiOiIx." + "SflKxwRJSM",
        "-----BEGIN " + "RSA PRIVATE KEY-----",
        "Bearer " + "abcdefghijklmnopqrstuvwx",
        "/Users/someone/notes.md",
        "C:\\Users\\someone\\notes.md",
        "someone@example.com",
    ]
    for v in leaky:
        with pytest.raises(SystemExit):
            E._audit({"value": v})
        with pytest.raises(SystemExit):  # and as a KEY, not just a value
            E._audit({v: 1})
        with pytest.raises(SystemExit):  # and nested inside a list
            E._audit({"runs": [{"ok": 1}, {"v": v}]})


def test_privacy_tripwire_does_not_block_legitimate_metrics():
    """A tripwire that fires on ordinary aggregates would just get disabled."""
    import extract as E

    E._audit(
        {
            "sessions": 40,
            "tool_calls": 4000,
            "opus_pct": 99.8,
            "models": {"opus": 100, "haiku": 3},
            "top_tools": {"Bash": 1500, "TaskUpdate": 50},
            "top_files": {E.hpath("/Users/someone/x.md"): 3},
            "reread_pct": 32.0,
        }
    )


def test_paths_are_one_way_hashed():
    import extract as E

    h = E.hpath("/Users/someone/secret-project/plan.md")
    assert "/" not in h and "someone" not in h and "secret" not in h
    assert h == E.hpath("/Users/someone/secret-project/plan.md")  # stable
    assert h != E.hpath("/Users/someone/secret-project/other.md")  # distinguishes


def test_model_family_collapses_build_identifiers():
    """Exact build strings can carry unreleased codenames; only the family ships."""
    import extract as E

    assert E.model_family("claude-opus-4-8") == "opus"
    assert E.model_family("claude-haiku-4-5-20251001") == "haiku"
    assert E.model_family("claude-fable-5") == "other"
    assert E.model_family("<synthetic>") == "other"
