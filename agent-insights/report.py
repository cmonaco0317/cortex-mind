#!/usr/bin/env python3
"""
Cortex · agent-insights — the Operator Report (Phase 1 paid tier).

Sits AFTER taxonomy.py. Where the free cards are *descriptive* and shareable,
this stage is the *paid* layer: a structured **Edge / Tax / Move** read on how
you operate your agent fleet.

Design (locked in OPERATOR-REPORT.md — do not drift):
  - **Leverage, not advice.** EDGE = strengths to lean into; TAX = a tradeoff
    you're choosing, never a scold; MOVE = a one-week self-experiment.
  - **Personalized or nothing.** Every shipped body cites a number computed from
    *your* metrics. A template that can't cite a number does not ship.
  - **Taxes fire on signal PAIRS, not lone stats** — an earned tradeoff, not a nag.
  - **Inference-honest.** The data knows what you *did*, never whether it *worked*.
    Every judgment is attributed to YOUR action ("a turn you flagged"), never to
    reality ("a wrong turn"). No outcome-quality verdicts. No cross-user
    percentiles (self-vs-self only — there is no backend). Mandatory honesty footer.
  - **Honest attribution.** Never charge the operator for the model's or harness's
    behavior (default model, how the transcript chunks tool calls) or for a
    one-time tool install. Only operator-controllable signals become taxes.
  - **One tension per signal-family**, reconciled ACROSS edges and taxes so no
    signal-family is both praised and taxed. (A single metric may inform an edge and
    a tax in DIFFERENT families — e.g. corrections as reflex-speed vs read-order — by
    design, mirroring the hand-authored gold example.)

100% local. Reads metrics.json (extract.py) → reuses compute_archetype (taxonomy.py)
→ emits report.json + a self-contained, watermarked report.html with a shareable
"my operator edge" teaser card. No backend, no CDN, no payment flow.

Usage:
  report.py metrics.json [--out-html report.html] [--out-json report.json]
"""

import argparse
import json
import re
import sys
from typing import Any

from extract import model_family
from taxonomy import compute_archetype, fmt

# One tension per signal-family; a cap keeps the report tight (a hero read, not a
# dashboard). Highest-priority survivors win.
MAX_EDGES = 4
MAX_TAXES = 3

# Minimum evidence before we headline an identity at all — a report built on a
# near-empty corpus must not assert a confident archetype.
MIN_TURNS_FOR_ARCHETYPE = 50

FOOTER = (
    "This report reads what you did — tools, timing, corrections, models — never "
    "whether the work was any good. So treat every tax and move as a hypothesis "
    "about you, testable in a week, not a verdict. You're the judge. Most tools in "
    "this space will confidently tell you you're doing it wrong; this one tells you "
    "what's true and lets you decide."
)

# Scoped privacy tripwire (see _audit): real path/email/secret shapes only, so
# ordinary prose ("read/write", "CI/CD", an "@mention") can't hard-crash a report.
_PATH = re.compile(r"(^|[\s(\"'])[A-Za-z]:\\|(^|[\s(\"'])/[\w.\-]+/[\w.\-]+")
_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
_SECRET = re.compile(
    r"sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|pk_live_|AIza[0-9A-Za-z_\-]{20,}|xox[bapr]-"
)


def _leaky(s: str) -> bool:
    return bool(_PATH.search(s) or _EMAIL.search(s) or _SECRET.search(s))


def _audit(obj: Any, path: str = "report") -> None:
    """Recursively hard-fail if any emitted string looks like a real path, email,
    or secret. Scoped to actual leak shapes, not any bare '/' or '@'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _leaky(k):
                raise SystemExit(f"privacy tripwire: key at {path}.{k!r} looks unsafe")
            _audit(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _audit(v, f"{path}[{i}]")
    elif isinstance(obj, str) and _leaky(obj):
        raise SystemExit(f"privacy tripwire: value at {path} looks unsafe: {obj!r}")


# --------------------------------------------------------------------------- #
# derived signals (shared by the rules)
# --------------------------------------------------------------------------- #
def _dispatches(m: dict) -> int:
    return int(m.get("workflow_calls", 0)) + int(m.get("agent_calls", 0))


def _bash(m: dict) -> int:
    top = m.get("top_tools", {}) or {}
    return int(m.get("bash", 0)) or int(top.get("Bash", 0))


def _smaller_models(m: dict) -> int:
    """Turns on a genuinely smaller model family (haiku/sonnet). Excludes the
    <synthetic>/other placeholder buckets and unknown codenames so the 'smaller
    model' claim is literally true, not inflated by non-model turns."""
    models = m.get("models") or {}
    if models:
        return sum(
            int(v) for k, v in models.items() if model_family(k) in ("haiku", "sonnet")
        )
    # No family breakdown: approximate from the percentage (can't distinguish
    # a genuine smaller model from a placeholder, so this is a loose fallback).
    assistant = int(m.get("assistant_turns", 0))
    return max(0, round(assistant * (100 - float(m.get("opus_pct", 0))) / 100))


def _epf(m: dict) -> float:
    # Edits-per-file must count the SAME population that most_churned_file_edits and
    # distinct_files_edited count (extract.py increments both on Edit OR Write), or
    # the churn tax compares an Edit+Write max against an Edit-only average.
    files = int(m.get("distinct_files_edited", 0))
    touches = int(m.get("edit", 0)) + int(m.get("write", 0))
    return round(touches / files, 1) if files else 0.0


# Genuine task-TRACKING tools only. Explicitly excludes the bare `Task` tool
# (that's subagent DISPATCH, already counted as agent_calls — a startswith("Task")
# match would falsely read delegation as planning and double-count it against the
# dispatch edge).
_TRACKER_TOOLS = ("TaskCreate", "TaskUpdate", "TaskList")


def _planning_calls(m: dict) -> int:
    """Task-tracking / todo tooling = planning on paper. Counts TodoWrite (via
    todo_calls) plus the explicit tracker tools, NOT subagent dispatch."""
    top = m.get("top_tools", {}) or {}
    tracked = sum(int(top.get(k, 0)) for k in _TRACKER_TOOLS)
    return int(m.get("todo_calls", 0)) + tracked


def _has_number(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


# --------------------------------------------------------------------------- #
# the rule library — EDGE (strengths to lean into; behavior only, never outcome)
# --------------------------------------------------------------------------- #
def _edges(m: dict) -> list[dict]:
    out: list[dict] = []

    def edge(family: str, priority: int, title: str, body: str) -> None:
        out.append(
            {"family": family, "priority": priority, "title": title, "body": body}
        )

    pounce = m.get("pounce_median_sec")
    corrections = int(m.get("corrections_caught", 0))
    dispatches = _dispatches(m)
    wf = int(m.get("workflow_calls", 0))
    ag = int(m.get("agent_calls", 0))
    max_turns = int(m.get("max_turns_in_session", 0))
    tool_calls = int(m.get("tool_calls", 0))
    bash = _bash(m)
    reread = float(m.get("reread_pct", 0))
    files = int(m.get("distinct_files_edited", 0))
    epf = _epf(m)

    if pounce is not None and pounce <= 12 and corrections >= 15:
        edge(
            "reflex",
            95,
            "You interrupt fast.",
            f"{fmt(corrections)} times you cut into a turn you'd flagged — a median "
            f"{pounce} seconds from the agent's last action to your next message. You "
            f"don't sit and watch a turn head somewhere you didn't want; you jump in "
            f"and redirect. Lean in — a fast interrupt is leverage on a loose leash.",
        )

    if dispatches >= 40:
        edge(
            "dispatch",
            90,
            "You direct work, you don't type all of it.",
            f"{fmt(dispatches)} tasks handed off to workflows and subagents "
            f"({wf} workflows and {ag} subagents). You've crossed from chatting with an "
            f"agent to running a fleet — you hand off whole tasks and redirect them, "
            f"rather than driving every step by hand.",
        )

    if max_turns >= 300:
        edge(
            "endurance",
            85,
            "You keep one long thread.",
            f"Your longest single project transcript reached {fmt(max_turns)} turns in "
            f"one continuous thread. You keep the build in a single running context "
            f"rather than restarting into fresh chats to clear your head.",
        )

    if tool_calls and bash / tool_calls >= 0.30:
        pct = round(100 * bash / tool_calls)
        edge(
            "bash",
            80,
            "The terminal is your first reach.",
            f"{fmt(bash)} shell commands — {pct}% of every tool call routed through "
            f"Bash. When you want something done you drop to the terminal, not a wrapper "
            f"around it.",
        )

    if files >= 40 and 0 < epf <= 2.0 and reread < 20:
        edge(
            "churn",
            79,
            "You touch a file and move on.",
            f"{epf} edits per file across {fmt(files)} files, with {reread}% re-reads — "
            f"low-rework passes with little doubling back.",
        )

    if reread >= 25:
        edge(
            "reread",
            78,
            "You look again before you commit.",
            f"{reread}% of your reads re-open a file already seen that session — you send "
            f"the agent back to the source instead of acting on a stale read.",
        )

    planning = _planning_calls(m)
    if planning >= 15:
        edge(
            "planning",
            76,
            "You map before you move.",
            f"{fmt(planning)} times you wrote or updated a task list — you lay the plan "
            f"out on paper and track it as you go, rather than holding the whole build in "
            f"your head.",
        )

    return out


# --------------------------------------------------------------------------- #
# the rule library — TAX (each fires on a signal PAIR; operator-controllable only)
# --------------------------------------------------------------------------- #
def _taxes(m: dict) -> list[dict]:
    out: list[dict] = []

    def tax(key: str, family: str, priority: int, title: str, body: str) -> None:
        out.append(
            {
                "key": key,
                "family": family,
                "priority": priority,
                "title": title,
                "body": body,
            }
        )

    corrections = int(m.get("corrections_caught", 0))
    r2e = float(m.get("read_to_edit_ratio", 9))
    opus_pct = float(m.get("opus_pct", 0))
    smaller = _smaller_models(m)
    assistant = int(m.get("assistant_turns", 0))
    churn = int(m.get("most_churned_file_edits", 0))
    files = int(m.get("distinct_files_edited", 0))
    epf = _epf(m)

    # Steer-live tax — the pair: edit-before-read AND a stream of live cut-ins.
    if r2e < 1.0 and corrections >= 20:
        tax(
            "steer",
            "planning",
            90,
            "The steer-live tax.",
            f"Your agent edits before it reads (a {r2e}× read-to-edit ratio), and you "
            f"cut in {fmt(corrections)} times as it moves — one habit read two ways: you "
            f"point and steer live rather than mapping the ground first. The data can't "
            f"tell you how many of those {fmt(corrections)} cut-ins one line of intent up "
            f"front would have prevented — that's a tradeoff worth making on purpose, not "
            f"by default.",
        )

    # One-gear tax — the pair: a dominant model AND a near-zero smaller-model sample.
    if opus_pct >= 98 and smaller <= max(20, round(assistant * 0.01)):
        cnt = (
            "none of your turns have"
            if smaller == 0
            else f"exactly {fmt(smaller)} of your turns have"
        )
        tax(
            "onegear",
            "model",
            85,
            "The one-gear tax.",
            f"All-Opus is the platform default — a fine default, not a gear you actively "
            f"pick. But {cnt} ever run on a smaller model, so you've never had the "
            f"side-by-side that would show where a lighter one is indistinguishable. "
            f"Whether that's costing you anything is exactly what you can't know until "
            f"you look.",
        )

    # One-file gravity tax — the pair: one file far above your own per-file average.
    if churn >= 40 and files >= 20 and churn >= 3 * epf:
        tax(
            "churn",
            "churn",
            72,
            "The one-file gravity tax.",
            f"One file pulled {fmt(churn)} edits while your average across {fmt(files)} "
            f"files is {epf}. You'll stay on one thing until it's right rather than move "
            f"on — a real preference, and the cost of it lives in that one file.",
        )

    return out


# --------------------------------------------------------------------------- #
# the rule library — MOVE (one per tax key; rendered only if its tax fired)
# --------------------------------------------------------------------------- #
def _moves(m: dict) -> list[dict]:
    corrections = int(m.get("corrections_caught", 0))
    rev = float(m.get("reversal_rate_per_100", 0))
    one_in = round(100 / rev) if rev else 0
    smaller = _smaller_models(m)
    churn = int(m.get("most_churned_file_edits", 0))

    reversal_clause = (
        f", and your 1-in-{one_in} self-reversals ease" if one_in >= 2 else ""
    )
    smaller_cite = (
        "none of your turns have"
        if smaller == 0
        else f"only {fmt(smaller)} of your turns have"
    )

    return [
        {
            "key": "steer",
            "title": "Write one line of intent before you dispatch — not a todo list.",
            "body": (
                f"One sentence: goal, and done-when. No plan doc, no ceremony. "
                f"Hypothesis to test on yourself for a week: fewer turns get flagged in "
                f"the first place{reversal_clause}, because the agent starts with your "
                f"intent instead of discovering it live — the same {fmt(corrections)} "
                f"cut-ins, moved earlier. If nothing shifts, you've spent one sentence "
                f"and you keep the reflex."
            ),
        },
        {
            "key": "onegear",
            "title": "Run one routing probe.",
            "body": (
                f"Force one recurring, low-stakes task type — a mechanical edit, a status "
                f"check — to a smaller model for one week. You're buying the side-by-side "
                f"you don't have; {smaller_cite} ever run on a lighter model. Keep it "
                f"where the output is indistinguishable; revert it where it isn't."
            ),
        },
        {
            "key": "churn",
            "title": "Timebox the sticky file.",
            "body": (
                f"Give the file that pulled {fmt(churn)} edits a hard three-pass ceiling "
                f"for a week. If it isn't right by pass three, write down why and move on "
                f"— then see whether that ceiling ever actually cost you anything."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# assembly + guards
# --------------------------------------------------------------------------- #
def _diversify(edges: list[dict], taxes: list[dict]) -> tuple[list[dict], list[dict]]:
    """One tension per signal-family, reconciled ACROSS edges and taxes: the same
    signal can never be both praised (edge) and taxed. Highest priority keeps the
    family; the loser is dropped. This is the load-bearing diversity guard."""
    combined = [("edge", e) for e in edges] + [("tax", t) for t in taxes]
    combined.sort(key=lambda st: -st[1]["priority"])
    seen: set[str] = set()
    kept_e: list[dict] = []
    kept_t: list[dict] = []
    for side, it in combined:
        if it["family"] in seen:
            continue
        seen.add(it["family"])
        (kept_e if side == "edge" else kept_t).append(it)
    kept_e.sort(key=lambda x: -x["priority"])
    kept_t.sort(key=lambda x: -x["priority"])
    return kept_e[:MAX_EDGES], kept_t[:MAX_TAXES]


def _guard_archetype(arch: dict | None, m: dict) -> dict | None:
    """The archetype is the report's headline — never ship one asserted from the
    ABSENCE of data or from a hardcoded template that would read false. Defends the
    shipping boundary without touching the free-card taxonomy."""
    if not arch:
        return None
    if (
        int(m.get("assistant_turns", 0)) < MIN_TURNS_FOR_ARCHETYPE
        and int(m.get("sessions", 0)) < 2
    ):
        return None
    # "The Night Builder" fires on before_1pm==0 alone in taxonomy — true for an
    # afternoon operator and for empty input. Require genuine nocturnal volume,
    # mirroring the hardened card.
    if arch.get("name") == "The Night Builder":
        hh = m.get("hour_histogram", {}) or {}
        night = sum(int(hh.get(str(h), 0)) for h in range(2, 5))
        if night <= 200:
            return None
    return arch


def build_report(m: dict) -> dict[str, Any]:
    arch = _guard_archetype(compute_archetype(m), m)

    edges, taxes = _diversify(_edges(m), _taxes(m))

    # Personalized-or-nothing: a body that can't cite a number never ships.
    edges = [e for e in edges if _has_number(e["body"])]
    taxes = [t for t in taxes if _has_number(t["body"])]

    fired = {t["key"] for t in taxes}
    moves = [mv for mv in _moves(m) if mv["key"] in fired and _has_number(mv["body"])]

    report: dict[str, Any] = {
        "kind": "operator_report",
        "archetype": arch,
        "edges": [
            {"family": e["family"], "title": e["title"], "body": e["body"]}
            for e in edges
        ],
        "taxes": [
            {
                "key": t["key"],
                "family": t["family"],
                "title": t["title"],
                "body": t["body"],
            }
            for t in taxes
        ],
        "moves": moves,
        "footer": FOOTER,
        "meta": {"sessions": int(m.get("sessions", 0))},
    }

    # Enforce the privacy guarantee on everything prose-bearing before it ships.
    _audit({k: v for k, v in report.items() if k != "meta"})
    return report


# --------------------------------------------------------------------------- #
# render — a single self-contained artifact + a shareable teaser card
# --------------------------------------------------------------------------- #
def _teaser_line(report: dict) -> str:
    arch = report.get("archetype")
    name = arch["name"] if arch else "The Operator"
    edge = (
        report["edges"][0]["title"].rstrip(".")
        if report["edges"]
        else "a fleet you run"
    )
    return f"{name} — my edge: {edge}."


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html(report: dict, m: dict) -> str:
    arch = report.get("archetype")
    teaser = _teaser_line(report)

    arch_block = ""
    if arch:
        arch_block = f"""
      <div class="arch">
        <div class="kicker">YOUR CLAUDE CODE ARCHETYPE</div>
        <h1>{_esc(arch['name'])}</h1>
        <div class="tag">{_esc(arch['tagline'])}</div>
        <p class="def">{_esc(arch['definition'])}</p>
      </div>"""

    def section(
        title: str, kicker: str, items: list[dict], empty_note: str = ""
    ) -> str:
        if not items:
            if not empty_note:
                return ""
            return (
                f'<section><div class="kicker">{kicker}</div>'
                f'<h2>{_esc(title)}</h2><div class="empty">{_esc(empty_note)}</div></section>'
            )
        rows = []
        for i, it in enumerate(items, 1):
            rows.append(
                f'<div class="item"><span class="n">{i}</span><div>'
                f'<h3>{_esc(it["title"])}</h3><p>{_esc(it["body"])}</p></div></div>'
            )
        return (
            f'<section><div class="kicker">{kicker}</div>'
            f"<h2>{_esc(title)}</h2>{''.join(rows)}</section>"
        )

    # A report built on too little signal must say so, not fake an identity.
    if not arch and not report["edges"]:
        edges_html = (
            '<section><div class="kicker">NOT ENOUGH YET</div>'
            "<h2>The read is still forming.</h2>"
            '<div class="empty">There aren\'t enough sessions here to name a pattern '
            "honestly. Come back after a week or two of real work and re-run — this "
            "report only says what the data actually shows, and right now it doesn't "
            "show much.</div></section>"
        )
        taxes_html = moves_html = ""
    else:
        edges_html = section(
            "Your edge — lean in", "STRENGTHS TO PRESS", report["edges"]
        )
        taxes_html = section(
            "Your tax — the tradeoff you're choosing",
            "COSTS YOU'RE PAYING",
            report["taxes"],
            empty_note=(
                "None of the tradeoffs this report looks for showed up this period — "
                "your signals read balanced on the things it measures. (That's not a "
                "clean bill of health, just the absence of the specific patterns above.)"
            ),
        )
        moves_html = section(
            "The move — one week, one experiment each",
            "TESTABLE ON YOURSELF",
            report["moves"],
        )

    # Embed JSON in <script>: escape '<' so a "</script>" in any string can't break out.
    teaser_json = json.dumps({"line": teaser}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cortex — Operator Report</title>
<style>
  :root{{color-scheme:dark}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:#05070d;color:#cfe6f2;font:16px/1.6 -apple-system,system-ui,sans-serif}}
  .wrap{{max-width:860px;margin:0 auto;padding:40px 24px 90px}}
  .brand{{font:600 15px/1 Menlo,monospace;color:#4fd6f5;letter-spacing:.04em;margin-bottom:6px}}
  .badge{{display:inline-block;font:600 12px/1 Menlo,monospace;color:#7fdcf5;border:1px solid #2a6f8c;border-radius:999px;padding:6px 12px;margin-bottom:26px}}
  .arch{{background:linear-gradient(135deg,#06182a,#02060d);border:1px solid rgba(80,160,200,.22);border-radius:16px;padding:30px 28px;margin-bottom:34px;box-shadow:0 10px 50px rgba(0,0,0,.5)}}
  .kicker{{font:600 12px/1 Menlo,monospace;color:#4fd6f5;letter-spacing:.10em;margin-bottom:8px}}
  h1{{font:800 46px/1.05 -apple-system,system-ui;color:#eafaff;margin:0 0 6px}}
  .tag{{font:italic 19px/1.3 -apple-system,system-ui;color:#7fdcf5;margin-bottom:14px}}
  .def{{color:#bfe4f5;margin:0}}
  section{{margin:0 0 30px}}
  h2{{font:700 24px/1.2 -apple-system,system-ui;color:#bdf3ff;margin:2px 0 16px}}
  .item{{display:flex;gap:14px;padding:16px 0;border-top:1px solid rgba(80,160,200,.13)}}
  .item .n{{flex:0 0 30px;height:30px;border-radius:8px;background:#123543;color:#bdf3ff;font:700 15px/30px Menlo,monospace;text-align:center}}
  h3{{font:700 18px/1.3 -apple-system,system-ui;color:#eafaff;margin:2px 0 6px}}
  .item p{{margin:0;color:#a8cadd}}
  .empty{{color:#7fa0b4;font-size:15px;border-top:1px solid rgba(80,160,200,.13);padding-top:14px}}
  .footer{{background:#0b1a24;border-left:3px solid #4fd6f5;border-radius:8px;padding:18px 20px;margin-top:34px;color:#9fc4d8;font-size:15px}}
  .teaser{{margin:6px 0 30px}}
  canvas{{width:100%;height:auto;border-radius:14px;border:1px solid rgba(80,160,200,.2);box-shadow:0 10px 50px rgba(0,0,0,.5)}}
  .btns{{display:flex;gap:9px;margin-top:10px}}
  button{{background:#123543;color:#bdf3ff;border:1px solid #3aa6cf;border-radius:8px;padding:9px 15px;font:inherit;font-size:13px;cursor:pointer}}
  button:hover{{background:#1a4a5e}}
  .x{{background:#0b2836;border-color:#2a7fa0;color:#cbeeff}}
</style></head><body>
<div class="wrap">
  <div class="brand">◧ CORTEX · AGENT INSIGHTS</div>
  <div class="badge">OPERATOR REPORT · sample</div>
  {arch_block}
  <div class="teaser">
    <div class="kicker">SHARE YOUR EDGE</div>
    <canvas id="teaser"></canvas>
    <div class="btns">
      <button id="dl">⤓ download PNG</button>
      <button class="x" id="post">𝕏 post my edge</button>
    </div>
  </div>
  {edges_html}
  {taxes_html}
  {moves_html}
  <div class="footer"><strong>What this report can't know.</strong> {_esc(report['footer'])}</div>
</div>
<script>
const T = {teaser_json};
const W=1200,H=630;
function wrap(ctx,text,x,y,maxW,lh,maxLines){{
  const words=text.split(/\\s+/); let line='',lines=0;
  for(const w of words){{
    const t=line?line+' '+w:w;
    if(ctx.measureText(t).width>maxW && line){{ctx.fillText(line,x,y);line=w;y+=lh;if(++lines>=maxLines-1)break;}}
    else line=t;
  }}
  ctx.fillText(line,x,y);return y+lh;
}}
function draw(cv){{
  const ctx=cv.getContext('2d');cv.width=W;cv.height=H;
  const g=ctx.createLinearGradient(0,0,W,H);g.addColorStop(0,'#06182a');g.addColorStop(1,'#02060d');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  const glow=ctx.createRadialGradient(W*0.26,H*0.4,0,W*0.26,H*0.4,W*0.7);
  glow.addColorStop(0,'rgba(50,150,255,0.22)');glow.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=glow;ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(90,200,240,0.07)';ctx.lineWidth=1;
  for(let i=0;i<30;i++){{ctx.beginPath();ctx.moveTo((i*97%W),(i*151%H));ctx.lineTo((i*233%W),(i*71%H));ctx.stroke();}}
  ctx.textBaseline='alphabetic';ctx.textAlign='left';
  ctx.font='600 20px Menlo,monospace';ctx.fillStyle='#5fd6f5';
  ctx.fillText('◧ CORTEX · MY OPERATOR EDGE',56,92);
  ctx.font='800 60px -apple-system,system-ui,sans-serif';ctx.fillStyle='#eafaff';
  wrap(ctx,T.line,56,200,W-112,70,5);
  ctx.font='600 21px Menlo,monospace';ctx.fillStyle='#4fd6f5';
  ctx.fillText('◧ CORTEX',56,H-40);
  ctx.font='16px -apple-system,system-ui,sans-serif';ctx.fillStyle='#41627a';ctx.textAlign='right';
  ctx.fillText('decoded locally from my own Claude Code sessions',W-56,H-40);
  ctx.textAlign='left';
}}
const cv=document.getElementById('teaser');draw(cv);
function dl(){{const a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download='cortex-operator-edge.png';document.body.appendChild(a);a.click();a.remove();}}
document.getElementById('dl').onclick=dl;
document.getElementById('post').onclick=()=>{{dl();window.open('https://twitter.com/intent/tweet?text='+encodeURIComponent(T.line+'\\n\\nDecoded locally from my own sessions with Cortex 🧠'),'_blank');}};
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out-html", default="report.html")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    m = json.load(open(args.metrics))
    report = build_report(m)

    if args.out_json:
        open(args.out_json, "w").write(json.dumps(report, indent=2, ensure_ascii=False))
    html = render_html(report, m)
    open(args.out_html, "w").write(html)

    arch = report.get("archetype")
    if arch:
        print(f"\n★ {arch['name']} — {arch['tagline']}")
    print(f"\n  EDGE ({len(report['edges'])}):")
    for e in report["edges"]:
        print(f"   ▸ {e['title']}")
    print(f"\n  TAX ({len(report['taxes'])}):")
    for t in report["taxes"]:
        print(f"   ▸ {t['title']}")
    print(f"\n  MOVE ({len(report['moves'])}):")
    for mv in report["moves"]:
        print(f"   ▸ {mv['title']}")
    print(f"\nwrote {args.out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
