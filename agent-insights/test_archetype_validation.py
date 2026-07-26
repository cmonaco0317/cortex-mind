#!/usr/bin/env python3
"""
Validating the archetype construct (§3.2).

Wilson intervals, lift mining and leave-one-out sign checks are real machinery,
but they all sit downstream of a construct nobody validated: there is no ground
truth for "The Pouncer", no second rater, and nothing in the suite would have
failed if labels were assigned at random within their gates.

Two things are tested here, on synthetic sessions with no personal data:

  - STABILITY. Leave one session out and see whether the label survives. A label
    that flips when one session is dropped is describing that session.
  - NEGATIVE CONTROL. Shuffle event order within sessions. A claim about ORDER
    must collapse; whatever survives was never reading order, and no card may
    say it was.

The control is also what caught `reread_pct` and `reversal_count` being labelled
order-dependent when they are not — a set-membership question and a keyword
count respectively, neither of which cares about sequence.
"""

import json
import os

import extract as E
import taxonomy as T


def _line(ts, role, content, model="claude-opus-4-8", usage=True):
    msg = {"role": role, "content": content}
    if role == "assistant":
        msg["model"] = model
        if usage:
            msg["usage"] = {"input_tokens": 10, "output_tokens": 10}
    return json.dumps({"timestamp": ts, "message": msg})


def _tool(name, **inp):
    return {"type": "tool_use", "name": name, "input": inp}


def _write_session(d, name, *, prompts, corrections, tool_name="Bash", hour=14):
    """One synthetic session.

    Each correction is a user message beginning with a CORRECTION keyword that
    arrives shortly AFTER an assistant turn — which is what makes it a
    correction, and precisely what shuffling destroys.
    """
    lines = []
    t = 0
    for i in range(prompts):
        t += 1
        lines.append(
            _line(
                "2026-07-%02dT%02d:%02d:%02dZ" % (1 + (t // 3600) % 27, hour, (t // 60) % 60, t % 60),
                "assistant",
                [_tool(tool_name, command="echo %d" % i)],
            )
        )
        t += 2  # a fast follow-up: the "pounce"
        # The filler must be genuinely neutral: "actually" and "go back" are
        # BOTH in the CORRECTION list, so an innocent-looking filler like
        # "actually go back" silently made every prompt a correction.
        # The Pouncer also needs REVERSALS, so half the corrections use a word
        # ("actually") that is in both lists.
        if i < corrections:
            text = "no, not like that" if i % 2 else "actually, revert that"
        else:
            text = "please continue"
        lines.append(
            _line(
                "2026-07-%02dT%02d:%02d:%02dZ" % (1 + (t // 3600) % 27, hour, (t // 60) % 60, t % 60),
                "user",
                text,
            )
        )
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _corpus(root, n_sessions=8, prompts=14, corrections=8):
    os.makedirs(root, exist_ok=True)
    for k in range(n_sessions):
        _write_session(root, "s%02d.jsonl" % k, prompts=prompts, corrections=corrections)
    return sorted(f for f in os.listdir(root) if f.endswith(".jsonl"))


# ----------------------------------------------------------------- stability ---
def test_a_uniform_corpus_yields_a_stable_label(tmp_path):
    d = str(tmp_path / "uniform")
    files = _corpus(d)
    st = T.archetype_stability(d, files)
    assert st is not None
    assert st["folds"] == len(files)
    # every session looks the same, so dropping any one must change nothing
    assert st["agreement"] == 1.0, st
    assert st["flip_rate"] == 0.0
    assert st["stable"] is True


def test_stability_declines_to_answer_on_too_few_sessions(tmp_path):
    """Fewer than five sessions cannot support a leave-one-out estimate. None is
    a real answer and must not be read as 'stable'."""
    d = str(tmp_path / "thin")
    files = _corpus(d, n_sessions=3)
    assert T.archetype_stability(d, files) is None


def test_the_stability_measure_can_actually_detect_instability(tmp_path):
    """The check that keeps the check honest: a measure that returns 1.0 for
    every corpus proves nothing. Here one session carries all the corrections,
    so dropping THAT session changes the label."""
    d = str(tmp_path / "lopsided")
    os.makedirs(d, exist_ok=True)
    # five bland sessions with no corrections at all...
    for k in range(5):
        _write_session(d, "b%02d.jsonl" % k, prompts=14, corrections=0)
    # ...and one that is nothing but corrections
    _write_session(d, "z_hot.jsonl", prompts=40, corrections=40)
    files = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))

    labels = []
    for drop in files:
        subset = [f for f in files if f != drop]
        a = T.compute_archetype(E.aggregate(d, subset))
        labels.append(a["name"] if a else None)
    assert len(set(labels)) > 1, (
        "dropping any single session left the label unchanged, so this fixture "
        "cannot demonstrate that the measure detects instability: %r" % labels
    )


# ---------------------------------------------------------- negative control ---
def test_shuffling_event_order_collapses_the_order_dependent_signals(tmp_path):
    """The control §3.2 asks for. A correction is a user message that FOLLOWS an
    assistant turn; destroy the order and the signal must go with it."""
    d = str(tmp_path / "ctl")
    files = _corpus(d)
    real = E.aggregate(d, files)
    shuf = E.aggregate(d, files, shuffle_seed=1)

    assert real["corrections_caught"] > 0, "fixture produced no corrections to lose"
    assert shuf["corrections_caught"] != real["corrections_caught"], (
        "shuffling within-session order did not change the correction count, so "
        "corrections are not measuring order at all"
    )


def test_the_shuffle_is_deterministic(tmp_path):
    """A control you cannot re-run is not a control. No hash(), which is salted
    per process and would make this differ run to run."""
    d = str(tmp_path / "det")
    files = _corpus(d)
    a = E.aggregate(d, files, shuffle_seed=7)
    b = E.aggregate(d, files, shuffle_seed=7)
    assert a == b


def test_shuffling_leaves_order_independent_metrics_untouched(tmp_path):
    """The other half of the control, and the half that caught a mistake: a
    metric that does NOT move under shuffling was never an order signal, whatever
    a card might imply. reread_pct is set membership; reversal_count is a keyword
    tally."""
    d = str(tmp_path / "indep")
    files = _corpus(d)
    real = E.aggregate(d, files)
    shuf = E.aggregate(d, files, shuffle_seed=3)
    for k in ("reread_pct", "reversal_count", "tool_calls", "sessions"):
        assert real[k] == shuf[k], "%s moved under shuffling: %r -> %r" % (
            k,
            real[k],
            shuf[k],
        )


def test_control_reports_both_categories(tmp_path):
    d = str(tmp_path / "rep")
    files = _corpus(d)
    c = T.shuffled_control(d, files, seed=1)
    assert c["order_signal_collapsed"] is True
    assert "corrections_caught" in c["order_dependent"]
    assert "reread_pct" in c["order_independent"]


# --------------------------------------------------------------------- gate ---
def test_an_unstable_label_is_withheld_rather_than_shown():
    arch = {
        "kind": "archetype",
        "name": "The Pouncer",
        "tagline": "t",
        "definition": "d",
        "traits": ["a"],
        "score": 95,
    }
    unstable = {
        "label": "The Pouncer",
        "folds": 10,
        "sessions": 11,
        "sampled_folds": False,
        "agreement": 0.4,
        "flip_rate": 0.6,
        "runners_up": {"The Director": 6},
        "stable": False,
    }
    gated = T.gate_archetype(arch, unstable)
    assert gated["withheld"] is True
    assert gated["name"] == "No stable archetype"
    assert "60%" in gated["definition"]
    assert "The Director" in gated["definition"]


def test_a_stable_label_is_shown_with_its_measured_figure():
    arch = {
        "kind": "archetype",
        "name": "The Pouncer",
        "tagline": "t",
        "definition": "d",
        "traits": ["a"],
        "score": 95,
    }
    stable = {
        "label": "The Pouncer",
        "folds": 40,
        "sessions": 56,
        "sampled_folds": True,
        "agreement": 1.0,
        "flip_rate": 0.0,
        "runners_up": {},
        "stable": True,
    }
    gated = T.gate_archetype(arch, stable)
    assert gated.get("withheld") is not True
    assert gated["name"] == "The Pouncer"
    assert gated["stability"]["flip_rate"] == 0.0
    assert "100%" in gated["stability_note"]


def test_an_unmeasured_label_says_so_rather_than_implying_stability():
    arch = {"kind": "archetype", "name": "X", "tagline": "t", "definition": "d",
            "traits": [], "score": 1}
    gated = T.gate_archetype(arch, None)
    assert gated["stability"] is None
    assert "not an identity" in gated["stability_note"].lower()
