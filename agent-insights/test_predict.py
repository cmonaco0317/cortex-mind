#!/usr/bin/env python3
"""
Tests for predict.py — the Predictive "Next-Move" engine (v2, rebuilt after review).

Synthetic ordered-event streams (no personal data). Locks the invariants the
adversarial review demanded:
  - ATTRIBUTION: tool-level patterns are framed as the AGENT's, never "your habit/
    reflex"; "you" is reserved for corrections/reversals/session choices
  - it is actually PREDICTIVE (context-conditioned) and LIFT-ranked (not majority-class)
  - Wilson intervals gate recommendations (no winner's-curse; 0/n isn't printed as 0%)
  - the rocky proxy is full-window only (no right-censoring), broadened beyond corrections
  - signature excludes the universal top-2 pair; novelty excludes read-only probes
  - the footer carries the actor caveat; honesty holds; thin data is graceful
"""

import json
import re

import pytest

import predict as P

OUTCOME_VERDICT = re.compile(
    r"\boptimal\b|\bbest (move|practice)\b|you should\b|it works\b|guaranteed", re.I
)
CROSS_USER = re.compile(
    r"\b(most|other|average) (people|operators|users)\b|than most|percentile", re.I
)
MISATTRIB = re.compile(
    r"your reflex|your habit|the groove you fall|autopilot vs|you reach for", re.I
)


def tool(n):
    return ("tool", n)


ERR, COR, REV = ("error", None), ("correction", None), ("reversal", None)


def sess(events):
    return P._session_stats(events)


def _blob(nudges):
    return " ".join(n["title"] + " " + n["body"] for n in nudges).lower()


# ------------------------------------------------------------------ stats
def test_wilson_never_reports_zero_for_small_sample():
    lo, hi = P._wilson(0, 12)
    assert lo == 0.0 and hi > 15  # 0/12 is consistent with a real rate, not "0%"
    lo2, hi2 = P._wilson(6, 12)
    assert lo2 < 50 < hi2  # wide interval on a coin-flip at n=12


# ------------------------------------------------------------------ predictive core
def test_high_lift_move_is_conditional_and_beats_base():
    # (Plan,Grep) -> Edit is near-certain; Edit's base rate is diluted by Bash filler.
    corpus = [
        sess([tool("Plan"), tool("Grep"), tool("Edit"), tool("Bash")] * 4)
        for _ in range(4)
    ]
    hl = P.high_lift_moves(corpus)
    assert hl, "a high-lift conditional move should surface"
    assert hl[0]["lift"] >= P.MIN_LIFT
    assert hl[0]["n"] >= P.MIN_CTX


def test_predict_next_conditions_on_context():
    corpus = [
        sess([tool("Grep"), tool("Read"), tool("Edit"), tool("Bash")] * 4)
        for _ in range(4)
    ]
    preds = P.predict_next(corpus, ["Grep", "Read"])
    assert preds and preds[0]["next"] == "Edit"


def test_predict_next_empty_on_unknown_context():
    corpus = [sess([tool("Grep"), tool("Edit")] * 6) for _ in range(3)]
    assert P.predict_next(corpus, ["Nonexistent"]) == []


# ------------------------------------------------------------------ signature: not a truism
def test_signature_excludes_the_universal_top2_pair():
    # Bash+Edit dominate; a Bash/Edit-only chain is universal and must be skipped.
    corpus = [sess([tool("Edit"), tool("Bash")] * 20) for _ in range(5)]
    sig = P.signature_chain(corpus)
    assert sig is None  # only the two most-used tools -> not a fingerprint


def test_signature_surfaces_a_distinctive_chain():
    seq = ([tool("Edit"), tool("Bash")] * 6) + (
        [tool("Grep"), tool("Read"), tool("Edit")] * 4
    )
    corpus = [sess(list(seq)) for _ in range(5)]
    sig = P.signature_chain(corpus)
    assert sig and any(t in sig["chain"] for t in ("Grep", "Read"))


# ------------------------------------------------------------------ proxy: full-window only
def test_rocky_proxy_ignores_truncated_windows():
    # a branch at the very end of a session has no observable window -> not counted rocky
    corpus = [
        sess([COR, tool("Bash")]) for _ in range(10)
    ]  # Bash is terminal each time
    tr = P.post_trigger(corpus)
    if "correction" in tr:
        bash = next((d for d in tr["correction"]["dist"] if d["tool"] == "Bash"), None)
        assert bash is None or bash["rocky_n"] == 0  # no full window ever observed


def test_rocky_proxy_counts_followup_steer():
    # after correction -> Bash, then another correction within the window = rocky hit
    corpus = [
        sess([COR, tool("Bash"), tool("Edit"), COR, tool("Read")]) for _ in range(10)
    ]
    tr = P.post_trigger(corpus)
    bash = next(d for d in tr["correction"]["dist"] if d["tool"] == "Bash")
    assert bash["rocky_n"] >= 8 and bash["rocky_pct"] == 100


# ------------------------------------------------------------------ nudges: attribution + honesty
def _corpus():
    planned = [
        sess(
            [tool("TodoWrite"), tool("Grep"), tool("Read"), tool("Edit"), tool("Bash")]
            * 3
        )
        for _ in range(8)
    ]
    rocky = [
        sess(
            [tool("Grep"), tool("Read"), tool("Edit"), tool("Bash")] * 2
            + [ERR, tool("Bash"), tool("Edit"), COR, tool("Read"), tool("Edit")]
        )
        for _ in range(8)
    ]
    rare = [
        sess([tool("Agent"), tool("Grep"), tool("Read"), tool("Edit")] * 2)
        for _ in range(3)
    ]
    return planned + rocky + rare


def test_nudges_fire_capped_one_per_family():
    nudges = P.build_nudges(P.mine(_corpus()))
    assert 1 <= len(nudges) <= P.MAX_NUDGES
    fams = [n["family"] for n in nudges]
    assert len(fams) == len(set(fams))


def test_tool_level_nudges_attribute_to_the_agent_not_the_operator():
    blob = _blob(P.build_nudges(P.mine(_corpus())))
    assert not MISATTRIB.search(
        blob
    ), "no 'your habit/reflex/you reach for' on model-level moves"
    # the signature nudge, if present, must name the agent explicitly
    for n in P.build_nudges(P.mine(_corpus())):
        if n["family"] == "signature":
            assert "agent" in n["body"].lower() and "not you" in n["body"].lower()


def test_every_nudge_cites_a_number_and_has_confidence():
    for n in P.build_nudges(P.mine(_corpus())):
        assert P._has_number(n["body"]), n["title"]
        assert n["confidence"] in ("low", "medium", "high")


def test_recommendations_are_caveated():
    for n in P.build_nudges(P.mine(_corpus())):
        if "one experiment" in n["body"].lower():
            b = n["body"].lower()
            assert any(c in b for c in ("correlation", "not proof", "clears the noise"))


def test_no_outcome_verdict_or_cross_user():
    blob = _blob(P.build_nudges(P.mine(_corpus())))
    assert not OUTCOME_VERDICT.search(blob)
    assert not CROSS_USER.search(blob)


def test_footer_carries_the_actor_caveat():
    f = P.FOOTER.lower()
    assert "agent" in f and "you drive" in f and "you decide" in f


# ------------------------------------------------------------------ novelty
def test_novelty_excludes_readonly_probes():
    # a rare read-only probe must NOT be picked as the explore/exploit slot
    corpus = [sess([tool("Edit"), tool("Bash")] * 5 + [COR] * 3) for _ in range(6)]
    corpus += [
        sess([tool("mcp__x__list_connected_browsers"), tool("Edit"), tool("Bash")])
        for _ in range(3)
    ]
    nov = P.novelty(corpus)
    assert nov is None or "list_connected_browsers" not in nov["tool"]


# ------------------------------------------------------------------ assembly / render
def test_build_next_moves_and_privacy(monkeypatch):
    monkeypatch.setattr(P, "parse_sessions", lambda _d: _corpus())
    rep = P.build_next_moves("x")
    assert rep["kind"] == "next_moves" and rep["nudges"]
    with pytest.raises(SystemExit):
        P._audit({"nudges": [{"body": "/Users/someone/proj"}]})


def test_thin_data_graceful(monkeypatch):
    monkeypatch.setattr(P, "parse_sessions", lambda _d: [sess([tool("Edit")])])
    rep = P.build_next_moves("x")
    assert isinstance(rep["nudges"], list) and rep["footer"]
    assert "CORTEX" in P.render_html(rep)


def test_determinism(monkeypatch):
    monkeypatch.setattr(P, "parse_sessions", lambda _d: _corpus())
    a = json.dumps(P.build_next_moves("x"), sort_keys=True, ensure_ascii=False)
    b = json.dumps(P.build_next_moves("x"), sort_keys=True, ensure_ascii=False)
    assert a == b


def test_render_self_contained(monkeypatch):
    monkeypatch.setattr(P, "parse_sessions", lambda _d: _corpus())
    html = P.render_html(P.build_next_moves("x"))
    assert "CORTEX" in html and "<script src" not in html and "<link " not in html
    assert "cdn" not in html.lower() and html.count("</script>") == 1


# ------------------------------------------------------------------ held-out evaluation
def _dated(events, started):
    s = P._session_stats(events, started)
    return s


def test_evaluation_is_measured_out_of_sample_on_a_chronological_split():
    """The gates elsewhere control noise in the FITTED estimate, which is not the
    same property as predicting. This is the only thing in the repo that scores
    the model on sessions it was never fitted to."""
    corpus = [
        _dated([tool("Plan"), tool("Grep"), tool("Edit"), tool("Bash")] * 6,
               "2026-01-%02dT00:00:00Z" % (d + 1))
        for d in range(10)
    ]
    ev = P.evaluate_next_move(corpus)
    assert ev is not None
    assert ev["n_train_sessions"] + ev["n_test_sessions"] == 10
    assert ev["n_test_sessions"] >= 1 and ev["n_train_sessions"] >= 1
    assert ev["n_predictions"] > 0
    # a perfectly cyclic corpus is exactly what a context model should nail
    assert ev["top1_accuracy"] > ev["baseline_accuracy"]
    assert ev["beats_baseline"] is True
    assert P.predictive_claim_allowed(ev) is True


def test_sessions_are_ordered_by_time_not_by_filename(tmp_path):
    """Session filenames are UUIDs, so directory order is arbitrary. A split on
    arbitrary order is not a held-out evaluation, it is a shuffle."""
    def write(name, ts, tools):
        lines = []
        for t in tools:
            lines.append(json.dumps({
                "timestamp": ts,
                "message": {"role": "assistant",
                            "content": [{"type": "tool_use", "name": t}]},
            }))
        (tmp_path / name).write_text("\n".join(lines), encoding="utf-8")

    write("zzz.jsonl", "2026-01-01T00:00:00Z", ["Read", "Edit"])
    write("aaa.jsonl", "2026-06-01T00:00:00Z", ["Bash", "Grep"])
    got = [s["started"] for s in P.parse_sessions(str(tmp_path))]
    assert got == ["2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z"], (
        "sessions must be ordered chronologically, got %r" % got
    )


def test_a_model_that_cannot_beat_base_rate_is_reported_as_such():
    """The outcome the review cares about most: an unflattering number must still
    be produced, and must still be the one the wording is gated on."""
    losing = {
        "n_train_sessions": 8,
        "n_test_sessions": 2,
        "n_predictions": 400,
        "context_coverage": 0.9,
        "top1_accuracy": 0.31,
        "baseline_accuracy": 0.44,
        "accuracy_minus_baseline": -0.13,
        "beats_baseline": False,
    }
    assert P.predictive_claim_allowed(losing) is False
    # unmeasured is not the same as good
    assert P.predictive_claim_allowed(None) is False


def test_predictive_wording_is_suppressed_when_the_model_loses_to_base_rate():
    """Acceptance for 3.1: the word is gated on the measured figure, on the card
    itself and not only in the README."""
    nudges = [{"family": "f", "title": "After Grep, Edit.", "body": "b",
               "confidence": "high"}]
    losing = {
        "n_train_sessions": 8, "n_test_sessions": 2, "n_predictions": 400,
        "context_coverage": 0.9, "top1_accuracy": 0.31, "baseline_accuracy": 0.44,
        "accuracy_minus_baseline": -0.13, "beats_baseline": False,
    }
    html = P.render_html({
        "kind": "next_moves", "nudges": nudges, "evaluation": losing,
        "predictive": P.predictive_claim_allowed(losing), "footer": P.FOOTER,
    })
    assert "most predictable moment" not in html
    assert "DOES NOT BEAT BASE RATE" in html
    assert "31.0%" in html and "44.0%" in html, "the real numbers must be on the card"
    assert "nothing below is a prediction" in html

    winning = dict(losing, top1_accuracy=0.62, baseline_accuracy=0.44,
                   accuracy_minus_baseline=0.18, beats_baseline=True)
    html2 = P.render_html({
        "kind": "next_moves", "nudges": nudges, "evaluation": winning,
        "predictive": P.predictive_claim_allowed(winning), "footer": P.FOOTER,
    })
    assert "most predictable moment" in html2
    assert "MEASURED OUT OF SAMPLE" in html2
    assert "62.0%" in html2


def test_an_unmeasured_report_never_claims_prediction():
    html = P.render_html({
        "kind": "next_moves",
        "nudges": [{"family": "f", "title": "t.", "body": "b", "confidence": "low"}],
        "evaluation": None, "predictive": False, "footer": P.FOOTER,
    })
    assert "most predictable moment" not in html
    assert "NOT MEASURED" in html
    assert "nothing here is claimed to predict anything" in html


def test_evaluation_declines_rather_than_guessing_on_thin_history():
    assert P.evaluate_next_move([sess([tool("Bash")]) for _ in range(3)]) is None
