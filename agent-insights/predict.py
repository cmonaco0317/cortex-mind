#!/usr/bin/env python3
"""
Cortex · agent-insights — the Predictive "Next-Move" engine (Operator Report v2).

A local, deterministic **next-move recommender** for your Claude Code workflow. It's
shaped like a real recommender — candidate moves from a context-conditioned model →
ranked by LIFT over your own baseline → significance-gated recommendations → one
reserved explore/exploit slot — but implemented as plain-Python sequence-mining. That's
the TikTok/Monolith *architecture*, ownable and local, NOT the enterprise stack
(PREDICTIVE-ENGINE.md §3: no Kafka/Flink, no neural nets, no vector DB, no backend).

Two honesty axes, both load-bearing (an adversarial review caught the engine failing
each before this rewrite):

  1. WHOSE move is it?  In Claude Code the ASSISTANT emits every tool call — you don't
     pick Edit vs Bash. So tool-level patterns are your AGENT's execution, framed as
     "your sessions / your agent," never "your habit." "You" is reserved for what you
     actually type: corrections and reversals, and the session-level choices you drive
     (planning early, how often you steer). Experiments target levers YOU hold.

  2. Is the signal real?  Every rate carries a Wilson 95% interval; transitions are
     ranked by lift over your base rate (so your most-used tool can't win by default);
     a recommendation ships ONLY when the alternative's interval clears the default's
     (no winner's-curse); the "smoother" proxy (a follow-up correction/error/reversal
     in the next WINDOW moves, full-window only) is explicitly a noisy proxy, and its
     direction (early steering = rockier) is disclosed as an assumption.

No true outcome label exists, so it predicts what you USUALLY do, never what's optimal.
100% local, aggregate-only (tool-name level), privacy-tripwired, no payment flow.

Usage:
  predict.py <project_dir> [--out-json next_moves.json] [--out-html whats_next.html]
  predict.py <project_dir> --context "Edit,Bash"   # live: predict the next move
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

from extract import CORRECTION, REVERSAL, blocks, is_tool_result, user_text
from report import _audit, _esc, _has_number
from taxonomy import fmt

# Tools that hand a whole task to the machine.
DISPATCH = ("Workflow", "Agent", "Task")
# Events the OPERATOR authors directly (typed), vs everything else (agent execution).
OPERATOR_TRIGGERS = ("correction", "reversal")
# Read-only probes that can't be "corrected" and aren't workflow levers — excluded
# from the novelty slot so it can't reward a trivial status call by reverse causation.
_PROBE = re.compile(
    r"^(list|get|read|search|find|fetch|show|quote|screenshot)_|_(list|get|status)$",
    re.I,
)

MIN_CHAIN = 8
MIN_TRIGGER = 8
MIN_CTX = 12  # a context must recur this often to condition on
MIN_ALT = 15  # an alternative branch needs this many before we'll name it
WINDOW = 5
MIN_LIFT = 1.4
MAX_NUDGES = 5


def _pretty(name: str) -> str:
    if isinstance(name, str) and name.startswith("mcp__"):
        parts = name.split("__")
        return parts[-1] if parts[-1] else name
    return name or "?"


def _wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate k/n, as percentages. Closed form
    — the honest way to show a small-sample rate without pretending 0/12 means 0%."""
    if n <= 0:
        return (0.0, 100.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (
        round(100 * max(0.0, centre - half), 1),
        round(100 * min(1.0, centre + half), 1),
    )


def _confidence(n: int) -> str:
    return "high" if n >= 40 else "medium" if n >= 15 else "low"


# --------------------------------------------------------------------------- #
# 1. parse ordered events (re-reads the raw .jsonl — it needs ORDER)
# --------------------------------------------------------------------------- #
def _trigger_kind(txt: str) -> str | None:
    low = txt.lower()
    if any(k in low[:70] for k in CORRECTION):
        return "correction"
    if any(k in low for k in REVERSAL):
        return "reversal"
    return None


def parse_sessions(project_dir: str) -> list[dict]:
    sessions: list[dict] = []
    for fn in sorted(f for f in os.listdir(project_dir) if f.endswith(".jsonl")):
        events: list[tuple[str, str | None]] = []
        try:
            fh = open(os.path.join(project_dir, fn), "r", errors="replace")
        except OSError:
            continue
        with fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    o = json.loads(ln)
                except (ValueError, TypeError):
                    continue
                m = o.get("message")
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "assistant":
                    for b in blocks(m):
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            events.append(("tool", b.get("name") or "?"))
                elif m.get("role") == "user":
                    c = m.get("content")
                    if is_tool_result(c):
                        if any(
                            isinstance(b, dict)
                            and b.get("type") == "tool_result"
                            and b.get("is_error")
                            for b in c
                        ):
                            events.append(("error", None))
                    else:
                        raw = user_text(c)
                        if (
                            "SYSTEM NOTIFICATION - NOT USER INPUT" in raw
                            or "<task-notification>" in raw
                        ):
                            continue
                        txt = re.sub(
                            r"<system-reminder>.*?</system-reminder>",
                            " ",
                            raw,
                            flags=re.S,
                        ).strip()
                        if txt and (k := _trigger_kind(txt)):
                            events.append((k, None))
        if events:
            sessions.append(_session_stats(events))
    return sessions


def _session_stats(events: list[tuple[str, str | None]]) -> dict:
    tools = [e for e in events if e[0] == "tool"]
    corrections = sum(1 for e in events if e[0] == "correction")
    return {
        "events": events,
        "n_tool_events": len(tools),
        "corrections": corrections,
        "errors": sum(1 for e in events if e[0] == "error"),
        "correction_rate": round(100 * corrections / len(tools), 1) if tools else 0.0,
        "tools_used": {e[1] for e in tools},
    }


# --------------------------------------------------------------------------- #
# 2. mine
# --------------------------------------------------------------------------- #
def _tool_seq(session: dict) -> list[str]:
    return [e[1] for e in session["events"] if e[0] == "tool"]


def base_tool_freq(sessions: list[dict]) -> dict[str, float]:
    c: Counter = Counter()
    for s in sessions:
        c.update(_tool_seq(s))
    total = sum(c.values()) or 1
    return {t: n / total for t, n in c.items()}


def conditional_next(sessions: list[dict], order: int = 2) -> dict[tuple, Counter]:
    """Variable-order Markov model: {context tuple of last 1..order tools: Counter(next)}.
    This is the actual predictive core — P(next | your current tail)."""
    model: dict[tuple, Counter] = defaultdict(Counter)
    for s in sessions:
        seq = _tool_seq(s)
        for i in range(1, len(seq)):
            for k in range(1, order + 1):
                if i - k >= 0:
                    model[tuple(seq[i - k : i])][seq[i]] += 1
    return model


def high_lift_moves(sessions: list[dict]) -> list[dict]:
    """The most DETERMINED points in your flow: contexts where the next move is far more
    likely than its base rate (lift), with enough samples and a Wilson floor above base.
    Ranked by lift — so a high-base-rate tool (Bash) can't win by popularity alone."""
    base = base_tool_freq(sessions)
    model = conditional_next(sessions, order=2)
    out: list[dict] = []
    for ctx, nxt in model.items():
        total = sum(nxt.values())
        if total < MIN_CTX:
            continue
        tool, k = nxt.most_common(1)[0]
        p_cond = k / total
        p_base = base.get(tool, 1e-9)
        lift = p_cond / p_base if p_base else 0
        lo, _ = _wilson(k, total)
        if (
            lift >= MIN_LIFT and lo > 100 * p_base
        ):  # conditional CI clears the base rate
            out.append(
                {
                    "context": tuple(ctx),
                    "next": tool,
                    "k": k,  # times that context led to `next`
                    "pct": round(100 * p_cond),
                    "lift": round(lift, 1),
                    "n": total,  # times the context occurred
                    "wilson": _wilson(k, total),
                }
            )
    # rank by lift, but prefer patterns with a real absolute tilt and more samples,
    # so the rarest-corner (n at the floor, 50% modal) doesn't top "most predictable".
    out.sort(
        key=lambda d: (d["lift"] * min(d["pct"], 100) * math.log(d["n"]),), reverse=True
    )
    return out


def signature_chain(sessions: list[dict]) -> dict | None:
    """Your agent's most OVER-REPRESENTED chain — ranked by lift vs a same-length
    random-order chain, and required to contain a distinctive tool (not just the two
    most-used tools), so a universal 'edit→run' pair can't masquerade as a fingerprint.
    """
    base = base_tool_freq(sessions)
    top2 = {t for t, _ in Counter(base).most_common(2)}
    counts: Counter = Counter()
    for s in sessions:
        seq = _tool_seq(s)
        for i in range(len(seq) - 2):
            counts[tuple(seq[i : i + 3])] += 1
    best = None
    for chain, c in counts.items():
        if c < MIN_CHAIN or len(set(chain)) < 2:
            continue
        if set(chain) <= top2:  # universal (only your two most-used tools) — skip
            continue
        expected = 1.0
        for t in chain:
            expected *= base.get(t, 1e-9)
        lift = (c / sum(counts.values())) / expected if expected else 0
        if best is None or lift > best["lift"]:
            best = {"chain": tuple(chain), "count": c, "lift": round(lift, 1)}
    return best


def post_trigger(sessions: list[dict]) -> dict[str, dict]:
    """For each trigger, the next-tool distribution + a full-window 'rocky' proxy: did
    a correction/error/reversal land within WINDOW tool-steps AFTER the branch. Only
    occurrences with a full observable window are counted (no right-censoring bias)."""
    triggers = ("error", "correction", "reversal", "dispatch")
    nxt: dict[str, Counter] = {t: Counter() for t in triggers}
    rocky: dict[str, dict[str, list[int]]] = {t: defaultdict(list) for t in triggers}
    for s in sessions:
        ev = s["events"]
        for i, (kind, name) in enumerate(ev):
            trig = (
                kind
                if kind in ("error", "correction", "reversal")
                else ("dispatch" if kind == "tool" and name in DISPATCH else None)
            )
            if not trig:
                continue
            j = i + 1
            while j < len(ev) and ev[j][0] != "tool":
                j += 1
            if j >= len(ev):
                continue
            nxt[trig][ev[j][1]] += 1
            # walk a full WINDOW of tool-steps; a "rocky" follow-up = any steer/error
            steps, hit, full = 0, 0, False
            k = j + 1
            while k < len(ev):
                if ev[k][0] in ("correction", "reversal", "error"):
                    hit = 1
                    full = True
                    break
                if ev[k][0] == "tool":
                    steps += 1
                    if steps >= WINDOW:
                        full = True
                        break
                k += 1
            if full:  # only score occurrences we could actually observe to WINDOW
                rocky[trig][ev[j][1]].append(hit)
    out: dict[str, dict] = {}
    for t in triggers:
        total = sum(nxt[t].values())
        if total < MIN_TRIGGER:
            continue
        dist = []
        for tool, c in nxt[t].most_common():
            samples = rocky[t][tool]
            k = sum(samples)
            dist.append(
                {
                    "tool": tool,
                    "display": _pretty(tool),
                    "count": c,
                    "pct": round(100 * c / total),
                    "rocky_n": len(samples),
                    "rocky_pct": round(100 * k / len(samples)) if samples else None,
                    "rocky_ci": _wilson(k, len(samples)) if samples else None,
                }
            )
        out[t] = {"total": total, "dist": dist}
    return out


def session_shape(sessions: list[dict]) -> list[dict]:
    """Self-vs-self, OPERATOR-owned: does an early choice YOU make (planning, dispatch)
    separate your smoother sessions from your rockier ones?"""
    if len(sessions) < 6:
        return []
    out: list[dict] = []

    def early_has(s: dict, pred) -> bool:
        head = s["events"][: max(3, len(s["events"]) // 5)]
        return any(pred(e) for e in head)

    signals = {
        "an early plan (TodoWrite/task list)": lambda e: e[0] == "tool"
        and (
            e[1] == "TodoWrite" or (isinstance(e[1], str) and e[1].startswith("Task"))
        ),
        "an early dispatch": lambda e: e[0] == "tool" and e[1] in DISPATCH,
    }
    for label, pred in signals.items():
        did = [s["correction_rate"] for s in sessions if early_has(s, pred)]
        didnt = [s["correction_rate"] for s in sessions if not early_has(s, pred)]
        if len(did) >= 3 and len(didnt) >= 3:
            a, b = round(sum(did) / len(did), 1), round(sum(didnt) / len(didnt), 1)
            out.append(
                {
                    "signal": label,
                    "with": a,
                    "without": b,
                    "n_with": len(did),
                    "n_without": len(didnt),
                    "delta": round(b - a, 1),
                }
            )
    return out


def novelty(sessions: list[dict]) -> dict | None:
    """Explore/exploit — a WORKFLOW-shaping tool (not a read-only probe) you rarely use,
    whose rare sessions looked smoother. Sign-checked under leave-one-out WHEN the sample
    allows (>3 uses); at the 3-use floor it's a raw delta. Always surfaced as a low-
    confidence curiosity with the selection effect disclosed in the copy."""
    if len(sessions) < 8:
        return None
    sess_with: Counter = Counter()
    for s in sessions:
        sess_with.update(s["tools_used"])
    n = len(sessions)
    cands = [
        t
        for t, c in sess_with.items()
        if 3 <= c <= max(3, n // 4) and t != "?" and not _PROBE.search(_pretty(t))
    ]
    best = None
    for t in cands:
        used = [s["correction_rate"] for s in sessions if t in s["tools_used"]]
        rest = [s["correction_rate"] for s in sessions if t not in s["tools_used"]]
        if len(used) < 3 or len(rest) < 3:
            continue
        delta = round(sum(rest) / len(rest) - sum(used) / len(used), 1)
        # leave-one-out sign stability: drop the best 'used' session, still smoother?
        loo = sorted(used)[1:] if len(used) > 3 else used
        loo_delta = sum(rest) / len(rest) - (sum(loo) / len(loo) if loo else 0)
        if delta > 0 and loo_delta > 0 and (best is None or delta > best["delta"]):
            best = {
                "tool": t,
                "sessions": sess_with[t],
                "delta": delta,
                "with": round(sum(used) / len(used), 1),
            }
    return best


def mine(sessions: list[dict]) -> dict:
    return {
        "n_sessions": len(sessions),
        "signature": signature_chain(sessions),
        "high_lift": high_lift_moves(sessions),
        "transitions": post_trigger(sessions),
        "shape": session_shape(sessions),
        "novelty": novelty(sessions),
    }


def predict_next(
    sessions: list[dict], context: list[str], order: int = 2
) -> list[dict]:
    """LIVE prediction (the CLI --context mode): given your current tail of tool moves,
    the empirical distribution of what your agent does next. Backs off long→short ctx.
    """
    model = conditional_next(sessions, order=order)
    for k in range(min(order, len(context)), 0, -1):
        ctx = tuple(context[-k:])
        nxt = model.get(ctx)
        if nxt and sum(nxt.values()) >= MIN_TRIGGER:
            total = sum(nxt.values())
            base = base_tool_freq(sessions)
            return [
                {
                    "next": _pretty(t),
                    "pct": round(100 * c / total),
                    "lift": (
                        round((c / total) / base.get(t, 1e-9), 1)
                        if base.get(t)
                        else None
                    ),
                    "n": total,
                }
                for t, c in nxt.most_common(5)
            ]
    return []


# --------------------------------------------------------------------------- #
# 3. build nudges (honest actor framing + significance-gated recommendations)
# --------------------------------------------------------------------------- #
_TRIGGER = {
    "error": ("an error", "agent"),  # both sides agent
    "correction": ("you cut in", "operator"),  # trigger is you; response is agent
    "reversal": ("you reverse course", "operator"),
    "dispatch": ("your agent dispatches a task", "agent"),
}


def build_nudges(mined: dict) -> list[dict]:
    nudges: list[dict] = []

    def add(family, priority, title, body, confidence):
        nudges.append(
            {
                "family": family,
                "priority": priority,
                "title": title,
                "body": body,
                "confidence": confidence,
            }
        )

    # (A) signature chain — your AGENT's over-represented fingerprint (descriptive)
    sig = mined["signature"]
    if sig:
        arrow = " → ".join(_pretty(t) for t in sig["chain"])
        add(
            "signature",
            88,
            "Your agent's signature chain.",
            f"Your sessions run {arrow} {fmt(sig['count'])} times — about {sig['lift']}× more "
            f"than random tool order would give. That's your agent's execution fingerprint "
            f"(the model picks the tools, not you) — worth recognizing as a default pattern.",
            _confidence(sig["count"]),
        )

    # (B) the predictive core — your most DETERMINED moment (highest lift, CI clears base)
    if mined["high_lift"]:
        hm = mined["high_lift"][0]
        ctx = " → ".join(_pretty(t) for t in hm["context"])
        tail = (
            "your agent is nearly on rails here"
            if hm["pct"] >= 75
            else "the strongest tilt in your flow — not a lock, but far above its usual rate"
        )
        add(
            "determined",
            90,
            "Your most predictable moment.",
            f"When your session reaches {ctx} (which happened {fmt(hm['n'])} times), the next move "
            f"is {_pretty(hm['next'])} in {hm['k']} of them — {hm['pct']}%, or {hm['lift']}× its usual "
            f"rate — {tail}. (The model picks the tool, not you — this is your agent's conditional habit.)",
            _confidence(hm["n"]),
        )

    # (C) post-trigger — operator triggers get an operator-runnable experiment;
    #     recommendation ships ONLY on a Wilson-CI-clearing alternative (no winner's curse)
    for trig in ("correction", "error", "dispatch"):
        info = mined["transitions"].get(trig)
        if not info or not info["dist"]:
            continue
        label, actor = _TRIGGER[trig]
        top = info["dist"][0]
        subj = (
            "your session's next move is"
            if actor == "agent"
            else "your sessions then go"
        )
        body = f"After {label}, {subj} {top['display']} {top['pct']}% of the time ({fmt(top['count'])} of {fmt(info['total'])})."
        # significance-gated alternative: alt's rocky-CI upper < top's rocky-CI lower
        rec = None
        if top["rocky_ci"] and top["rocky_pct"] is not None:
            for d in info["dist"]:
                if (
                    d["tool"] == top["tool"]
                    or d["count"] < MIN_ALT
                    or not d["rocky_ci"]
                ):
                    continue
                if d["rocky_ci"][1] < top["rocky_ci"][0]:  # CI non-overlap, alt lower
                    rec = d
                    break
        conf = info["total"]
        if rec:
            conf = rec["rocky_n"]
            exp = (
                f"steer toward a {rec['display']}-style move there"
                if actor == "operator"
                else f"a {rec['display']}-first prompt in that spot"
            )
            body += (
                f" The {rec['pct']}% that went {rec['display']} instead (n={rec['count']}) saw a "
                f"follow-up steer only {rec['rocky_pct']}% of the time vs {top['rocky_pct']}% — a gap "
                f"that clears the noise at these samples, though it's still correlation in your own "
                f"data, not proof. One experiment: {exp}, and watch."
            )
        else:
            body += " No alternative branch beat it beyond sampling noise — that's just your default there."
        add(
            f"trans_{trig}",
            80 - list(_TRIGGER).index(trig),
            f"After {label}…",
            body,
            _confidence(conf),
        )

    # (D) session-shape — OPERATOR-owned, operator-runnable
    for sh in sorted(mined["shape"], key=lambda x: -abs(x["delta"])):
        if abs(sh["delta"]) < 2:
            continue
        add(
            "shape",
            72,
            "How you open a session.",
            f"Sessions you opened with {sh['signal']} ran at ~{sh['with']} corrections per 100 "
            f"tool-calls ({sh['n_with']} sessions) vs ~{sh['without']} without ({sh['n_without']}) "
            f"— {'fewer' if sh['delta'] > 0 else 'more'} of your own cut-ins when you did. That's a "
            f"lever you actually hold; worth a week of doing it on purpose. (Correlation in your own "
            f"data, not cause.)",
            _confidence(sh["n_with"] + sh["n_without"]),
        )
        break

    # (E) explore/exploit — the reserved anti-rut slot (workflow tool, selection disclosed)
    nov = mined.get("novelty")
    if nov:
        td = _pretty(nov["tool"])
        add(
            "novelty",
            50,
            "The road not taken.",
            f"Your sessions that used {td} — only {nov['sessions']} of them — ran ~{nov['delta']} "
            f"corrections-per-100 lower than the rest. Big caveat: {td} was the best-looking of "
            f"several rarely-used tools, so expect that gap to shrink. Still, it's the one worth a "
            f"single deliberate session leaning that way, precisely because it's outside your usual lane.",
            "low",
        )

    nudges = [n for n in nudges if _has_number(n["body"])]
    nudges.sort(key=lambda n: -n["priority"])
    seen, kept = set(), []
    for n in nudges:
        if n["family"] in seen:
            continue
        seen.add(n["family"])
        kept.append(n)
    nov_slot = [n for n in kept if n["family"] == "novelty"][:1]
    rest = [n for n in kept if n["family"] != "novelty"][: MAX_NUDGES - len(nov_slot)]
    return rest + nov_slot


FOOTER = (
    "Two things this can't know. WHOSE move: in Claude Code the assistant emits every "
    "tool call, so the tool-by-tool patterns are your AGENT's execution, not your hand — "
    "what YOU drive is your corrections, reversals, and what you set up (planning, "
    "dispatch). WHETHER it worked: there's no outcome label, so 'rockier' here just means "
    "a follow-up correction/error landed soon — a noisy proxy, and treating early steering "
    "as bad is an assumption (it can be healthy fast iteration). Every nudge is a "
    "hypothesis testable in a week; the last is deliberately outside your lane, because a "
    "predictor that only echoes your ruts would entrench them. You decide."
)


def build_next_moves(project_dir: str) -> dict[str, Any]:
    mined = mine(parse_sessions(project_dir))
    report = {
        "kind": "next_moves",
        "nudges": [
            {k: n[k] for k in ("family", "title", "body", "confidence")}
            for n in build_nudges(mined)
        ],
        "footer": FOOTER,
        "meta": {"sessions": mined["n_sessions"]},
    }
    _audit({k: v for k, v in report.items() if k != "meta"})
    return report


# --------------------------------------------------------------------------- #
# 4. render
# --------------------------------------------------------------------------- #
def render_html(report: dict) -> str:
    nudges = report["nudges"]
    teaser = (
        f"My most predictable moment: {nudges[0]['title'].rstrip('.')}"
        if nudges
        else "My Claude Code next-move read"
    )
    teaser_json = json.dumps({"line": teaser}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    if nudges:
        rows = "".join(
            f'<div class="item"><span class="n">{i}</span><div>'
            f'<h3>{_esc(x["title"])} <em class="conf">{_esc(x["confidence"])} confidence</em></h3>'
            f'<p>{_esc(x["body"])}</p></div></div>'
            for i, x in enumerate(nudges, 1)
        )
        body_html = f'<section><div class="kicker">WHAT TENDS TO COME NEXT</div><h2>Your next moves</h2>{rows}</section>'
    else:
        body_html = (
            '<section><div class="kicker">NOT ENOUGH YET</div><h2>No pattern to read yet.</h2>'
            '<div class="empty">There aren\'t enough ordered sessions to mine a reliable next-move '
            "pattern. Come back after more real work.</div></section>"
        )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Cortex — What's Next</title>
<style>
  :root{{color-scheme:dark}}*{{box-sizing:border-box}}
  body{{margin:0;background:#05070d;color:#cfe6f2;font:16px/1.6 -apple-system,system-ui,sans-serif}}
  .wrap{{max-width:860px;margin:0 auto;padding:40px 24px 90px}}
  .brand{{font:600 15px/1 Menlo,monospace;color:#4fd6f5;letter-spacing:.04em;margin-bottom:6px}}
  .badge{{display:inline-block;font:600 12px/1 Menlo,monospace;color:#7fdcf5;border:1px solid #2a6f8c;border-radius:999px;padding:6px 12px;margin-bottom:26px}}
  .kicker{{font:600 12px/1 Menlo,monospace;color:#4fd6f5;letter-spacing:.10em;margin-bottom:8px}}
  h2{{font:700 24px/1.2 -apple-system,system-ui;color:#bdf3ff;margin:2px 0 16px}}
  .item{{display:flex;gap:14px;padding:16px 0;border-top:1px solid rgba(80,160,200,.13)}}
  .item .n{{flex:0 0 30px;height:30px;border-radius:8px;background:#123543;color:#bdf3ff;font:700 15px/30px Menlo,monospace;text-align:center}}
  h3{{font:700 18px/1.3 -apple-system,system-ui;color:#eafaff;margin:2px 0 6px}}
  .conf{{font:600 11px/1 Menlo,monospace;color:#5f8296;font-style:normal;letter-spacing:.04em}}
  .item p{{margin:0;color:#a8cadd}}.empty{{color:#7fa0b4;font-size:15px;border-top:1px solid rgba(80,160,200,.13);padding-top:14px}}
  .footer{{background:#0b1a24;border-left:3px solid #4fd6f5;border-radius:8px;padding:18px 20px;margin-top:34px;color:#9fc4d8;font-size:15px}}
  canvas{{width:100%;height:auto;border-radius:14px;border:1px solid rgba(80,160,200,.2);box-shadow:0 10px 50px rgba(0,0,0,.5);margin-bottom:8px}}
  .btns{{display:flex;gap:9px;margin-bottom:24px}}button{{background:#123543;color:#bdf3ff;border:1px solid #3aa6cf;border-radius:8px;padding:9px 15px;font:inherit;font-size:13px;cursor:pointer}}
  button:hover{{background:#1a4a5e}}.x{{background:#0b2836;border-color:#2a7fa0;color:#cbeeff}}
</style></head><body>
<div class="wrap">
  <div class="brand">◧ CORTEX · AGENT INSIGHTS</div><div class="badge">WHAT'S NEXT · sample</div>
  <canvas id="teaser"></canvas>
  <div class="btns"><button id="dl">⤓ download PNG</button><button class="x" id="post">𝕏 post</button></div>
  {body_html}
  <div class="footer"><strong>What this can't know.</strong> {_esc(report['footer'])}</div>
</div>
<script>
const T={teaser_json};const W=1200,H=630;
function wrap(ctx,t,x,y,mw,lh,ml){{const ws=t.split(/\\s+/);let l='',n=0;for(const w of ws){{const tt=l?l+' '+w:w;if(ctx.measureText(tt).width>mw&&l){{ctx.fillText(l,x,y);l=w;y+=lh;if(++n>=ml-1)break;}}else l=tt;}}ctx.fillText(l,x,y);return y+lh;}}
function draw(cv){{const ctx=cv.getContext('2d');cv.width=W;cv.height=H;
  const g=ctx.createLinearGradient(0,0,W,H);g.addColorStop(0,'#06182a');g.addColorStop(1,'#02060d');ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  const gl=ctx.createRadialGradient(W*0.26,H*0.4,0,W*0.26,H*0.4,W*0.7);gl.addColorStop(0,'rgba(50,150,255,0.22)');gl.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=gl;ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(90,200,240,0.07)';ctx.lineWidth=1;for(let i=0;i<30;i++){{ctx.beginPath();ctx.moveTo(i*97%W,i*151%H);ctx.lineTo(i*233%W,i*71%H);ctx.stroke();}}
  ctx.textBaseline='alphabetic';ctx.textAlign='left';ctx.font='600 20px Menlo,monospace';ctx.fillStyle='#5fd6f5';ctx.fillText('◧ CORTEX · WHAT COMES NEXT',56,92);
  ctx.font='800 56px -apple-system,system-ui,sans-serif';ctx.fillStyle='#eafaff';wrap(ctx,T.line,56,190,W-112,66,5);
  ctx.font='600 21px Menlo,monospace';ctx.fillStyle='#4fd6f5';ctx.fillText('◧ CORTEX',56,H-40);
  ctx.font='16px -apple-system,system-ui,sans-serif';ctx.fillStyle='#41627a';ctx.textAlign='right';ctx.fillText('mined locally from my own Claude Code sessions',W-56,H-40);ctx.textAlign='left';}}
const cv=document.getElementById('teaser');draw(cv);
function dl(){{const a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download='cortex-next-move.png';document.body.appendChild(a);a.click();a.remove();}}
document.getElementById('dl').onclick=dl;
document.getElementById('post').onclick=()=>{{dl();window.open('https://twitter.com/intent/tweet?text='+encodeURIComponent(T.line+'\\n\\nMined locally from my own sessions with Cortex 🧠'),'_blank');}};
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument(
        "--context",
        default="",
        help="comma-separated recent tools → predict the next move",
    )
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-html", default="whats_next.html")
    args = ap.parse_args()

    if args.context:
        preds = predict_next(
            parse_sessions(args.project_dir),
            [t.strip() for t in args.context.split(",") if t.strip()],
        )
        if not preds:
            print("no confident prediction for that context (too few examples).")
            return 0
        print(f"\n  After {args.context} → your agent usually:")
        for p in preds:
            lift = f"  ({p['lift']}× base)" if p.get("lift") else ""
            print(f"   {p['pct']:>3}%  {p['next']}{lift}   [n={p['n']}]")
        return 0

    report = build_next_moves(args.project_dir)
    if args.out_json:
        open(args.out_json, "w").write(json.dumps(report, indent=2, ensure_ascii=False))
    open(args.out_html, "w").write(render_html(report))
    print(
        f"\n  WHAT'S NEXT ({len(report['nudges'])} nudges, {report['meta']['sessions']} sessions):"
    )
    for n in report["nudges"]:
        print(f"   ▸ [{n['confidence']}] {n['title']}")
    print(f"\nwrote {args.out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
