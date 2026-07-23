#!/usr/bin/env python3
"""
End-to-end pipeline test: extract -> taxonomy -> render -> report.

Every other test in this repo exercises one function in isolation. The one-gear
bug could not be seen that way: _taxes() built the tax correctly and the filter
in build_report() was correct in isolation, and the defect only existed in the
interaction between a template branch and a downstream filter, on realistic
data. So this runs the four stages as the README documents them — as separate
processes over .jsonl session files — and asserts on the artifacts that ship.

The fixture is a synthetic operator, assembled at runtime rather than committed
as a file. Same reason the credential fixtures are assembled at runtime: the
sessions contain path-, username- and prompt-shaped strings on purpose (that is
the point — they are what must NOT come out the other end), and a scanner
matches on shape.

Profile: 3 sessions, 270 assistant turns, 100% Opus with zero smaller-model
turns — the strongest one-gear signal, and the exact shape that used to ship an
all-compliments report.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))

# Canaries. Each is something the session logs contain and no artifact may.
USER = "zaphod"
PROMPT_CANARY = "PROMPTBODYCANARY"
INJECTED_CANARY = "INJECTEDSYSTEMCANARY"
BUILD = "claude-opus-4-8"
PROJECT = f"/Users/{USER}/projects/alpha"
HOT_FILE = f"{PROJECT}/service_layer.py"

MODEL_VERSION = re.compile(
    r"\b(claude|opus|sonnet|haiku|gpt|gemini)[\s\-_.]*v?\d", re.I
)
PATH_SHAPE = re.compile(r"/Users/|[A-Za-z]:\\\\|\.py\b|\.jsonl\b")

SESSIONS = 3
PROMPTS_PER_SESSION = 30
CORRECTIONS_PER_SESSION = 12


def _tool(name, path=None):
    inp = {"file_path": path} if path else {"command": "pytest -q"}
    return {"type": "tool_use", "name": name, "input": inp}


def _tool_script(s):
    """The session's tool calls, in order. Deterministic — no randomness, so a
    failure reproduces exactly."""
    calls = []
    for i in range(26):  # 26 distinct reads ...
        calls.append(_tool("Read", f"{PROJECT}/read_{i}.py"))
    for i in range(14):  # ... then 14 re-reads of files already seen
        calls.append(_tool("Read", f"{PROJECT}/read_{i}.py"))
    for _ in range(15):  # one file that keeps pulling edits
        calls.append(_tool("Edit", HOT_FILE))
    for i in range(15):
        for _ in range(3):
            calls.append(_tool("Edit", f"{PROJECT}/s{s}_mod{i}.py"))
    for i in range(13):
        calls.append(_tool("Write", f"{PROJECT}/s{s}_mod{i}.py"))
    calls += [_tool("Bash") for _ in range(40)]
    calls += [_tool("Workflow") for _ in range(7)]
    calls += [_tool("Agent") for _ in range(10)]
    calls += [_tool("TodoWrite") for _ in range(7)]
    calls += [_tool("mcp__github__search_code") for _ in range(3)]
    return calls


def _write_sessions(dirpath):
    """Three realistic .jsonl transcripts. Returns the number of records."""
    total = 0
    for s in range(SESSIONS):
        clock = datetime(2026, 3, 2 + s, 9, 0, 0, tzinfo=timezone.utc)
        script = _tool_script(s)
        per_turn = len(script) // (PROMPTS_PER_SESSION * 3)
        lines = []

        def rec(message):
            nonlocal clock
            clock += timedelta(seconds=5)
            lines.append(
                json.dumps(
                    {
                        "timestamp": clock.isoformat().replace("+00:00", "Z"),
                        "gitBranch": f"feature/{USER}-alpha",
                        "sessionId": f"{USER}-session-{s}",
                        "message": message,
                    }
                )
            )

        # Injected system context: stripped before behavioral tells are counted,
        # and never emitted.
        rec(
            {
                "role": "user",
                "content": f"<system-reminder>{INJECTED_CANARY}</system-reminder>",
            }
        )

        cur = 0
        for p in range(PROMPTS_PER_SESSION):
            corrective = p % 2 == 1 and p // 2 < CORRECTIONS_PER_SESSION
            text = (
                f"no, stop — go back to the earlier approach. {PROMPT_CANARY} {p}"
                if corrective
                else f"add the retry wrapper in {HOT_FILE}. {PROMPT_CANARY} {p}"
            )
            rec({"role": "user", "content": text})
            for _ in range(3):
                blocks = script[cur : cur + per_turn]
                cur += per_turn
                rec(
                    {
                        "role": "assistant",
                        "model": BUILD,
                        "usage": {
                            "input_tokens": 1200,
                            "output_tokens": 400,
                            "cache_read_input_tokens": 90000,
                            "cache_creation_input_tokens": 2000,
                        },
                        "content": [{"type": "thinking", "thinking": "..."}] + blocks,
                    }
                )
                rec(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": f"ok: {HOT_FILE}",
                            }
                        ],
                    }
                )

        with open(os.path.join(dirpath, f"{USER}-session-{s}.jsonl"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        total += len(lines)
    return total


def _run(*argv):
    r = subprocess.run(
        [sys.executable, *argv], cwd=REPO, capture_output=True, text=True
    )
    assert r.returncode == 0, f"{argv[0]} failed:\n{r.stdout}\n{r.stderr}"
    return r


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the documented pipeline end to end, once, over synthetic sessions."""
    root = tmp_path_factory.mktemp("agent-insights-e2e")
    logs = root / "sessions"
    logs.mkdir()
    _write_sessions(str(logs))

    metrics = root / "metrics.json"
    cards = root / "cards.json"
    cards_html = root / "cards.html"
    report_json = root / "report.json"
    report_html = root / "report.html"

    _run(f"{REPO}/extract.py", str(logs), "--out", str(metrics))
    _run(f"{REPO}/taxonomy.py", str(metrics), "--out", str(cards))
    _run(f"{REPO}/render.py", str(cards), "--out", str(cards_html))
    _run(
        f"{REPO}/report.py",
        str(metrics),
        "--out-json",
        str(report_json),
        "--out-html",
        str(report_html),
    )

    return {
        "metrics": json.loads(metrics.read_text()),
        "cards": json.loads(cards.read_text()),
        "report": json.loads(report_json.read_text()),
        "artifacts": {
            "metrics.json": metrics.read_text(),
            "cards.json": cards.read_text(),
            "cards.html": cards_html.read_text(),
            "report.json": report_json.read_text(),
            "report.html": report_html.read_text(),
        },
    }


# ------------------------------------------------------------------ completes
def test_pipeline_completes_and_emits_a_report(pipeline):
    m = pipeline["metrics"]
    assert m["sessions"] == SESSIONS
    assert m["assistant_turns"] == SESSIONS * PROMPTS_PER_SESSION * 3
    assert m["real_user_prompts"] == SESSIONS * PROMPTS_PER_SESSION  # reminder stripped

    rep = pipeline["report"]
    assert rep["kind"] == "operator_report"
    assert rep["archetype"] is not None
    assert rep["edges"], "an operator with this much signal must earn edges"
    assert rep["footer"]

    html = pipeline["artifacts"]["report.html"]
    assert "OPERATOR REPORT" in html
    assert "NOT ENOUGH YET" not in html
    assert "<script src" not in html and "cdn" not in html.lower()


# ------------------------------------------------- regression on the inverted tax
def test_strong_tax_signal_produces_a_tax_and_a_move(pipeline):
    """This operator ran 100% Opus and never once used a smaller model — the
    strongest form of the one-gear signal. Before the fix the tax was dropped by
    a numeral scan over its rendered body, and the moves cascaded to zero with
    it, so this profile shipped as pure compliments."""
    m, rep = pipeline["metrics"], pipeline["report"]
    assert m["opus_pct"] == 100.0
    assert m["models"] == {"opus": m["assistant_turns"]}

    keys = {t["key"] for t in rep["taxes"]}
    assert "onegear" in keys, rep["taxes"]
    assert {mv["key"] for mv in rep["moves"]} == keys

    # and it reaches the shipped page, not just the JSON
    html = pipeline["artifacts"]["report.html"]
    onegear = next(t for t in rep["taxes"] if t["key"] == "onegear")
    assert onegear["title"] in html
    assert "COSTS YOU'RE PAYING" in html


# ------------------------------------------------------------------ privacy
@pytest.mark.parametrize(
    "canary",
    [USER, PROMPT_CANARY, INJECTED_CANARY, BUILD, "service_layer", "feature/"],
)
def test_no_artifact_leaks_a_canary(pipeline, canary):
    """End-to-end privacy: username, raw paths, prompt text, injected system
    context, and the dated model build string all appear in the session logs and
    must appear in none of the five artifacts."""
    for name, blob in pipeline["artifacts"].items():
        assert canary not in blob, f"{canary!r} leaked into {name}"


def test_shipped_prose_has_no_paths_or_model_builds(pipeline):
    prose = [
        s
        for c in pipeline["cards"]
        for s in (
            c.get("hero"),
            c.get("title"),
            c.get("sub"),
            c.get("name"),
            c.get("tagline"),
            c.get("definition"),
            *(c.get("traits") or []),
        )
        if isinstance(s, str)
    ]
    rep = pipeline["report"]
    prose += [it["title"] for it in rep["edges"] + rep["taxes"] + rep["moves"]]
    prose += [it["body"] for it in rep["edges"] + rep["taxes"] + rep["moves"]]
    prose.append(rep["footer"])
    assert prose

    for s in prose:
        assert not PATH_SHAPE.search(s), s
        assert not MODEL_VERSION.search(s), s


def test_metrics_carry_only_hashed_file_identity(pipeline):
    m = pipeline["metrics"]
    # 46 distinct files were edited (1 hot file + 15 per session x 3); the hot
    # one pulled 45 edits. The counts survive; the paths do not.
    assert m["distinct_files_edited"] == 46
    assert m["most_churned_file_edits"] == 45
    assert "top_files" not in m and "files" not in m
