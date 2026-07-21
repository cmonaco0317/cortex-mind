#!/usr/bin/env python3
"""
Cortex · agent-insights — extract behavioral signals from Claude Code sessions.

Reads a project's .jsonl session transcripts and computes deterministic,
content-agnostic metrics about how you (and your agent) actually work. The
output feeds the insight taxonomy (taxonomy.py) which turns numbers into ranked,
non-obvious "how did it know that about me" cards.

100% local. Emits **aggregate stats only** — counts, ratios, timing — and NEVER
file contents, file paths, project names, message text, or model build strings.
File paths are one-way hashed at ingest; model names are collapsed to a coarse
family (opus/sonnet/haiku/other); no raw prompt token is ever emitted. A
privacy tripwire (see _audit) hard-fails the run if any emitted string still
looks like a path, email, or secret — so the guarantee is enforced, not promised.

Usage:
  extract.py <project_dir> [--out metrics.json]
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Keyword-distinctive behavioral tells. These match on your OWN prompt text but
# only ever increment counters — no token is stored or emitted.
CORRECTION = (
    "no ",
    "no,",
    "wait",
    "stop",
    "actually",
    "don't",
    "dont",
    "not ",
    "instead",
    "revert",
    "go back",
    "undo",
    "nvm",
    "never mind",
    "hold on",
    "that's wrong",
    "thats wrong",
)
REVERSAL = (
    "actually",
    "revert",
    "go back",
    "nvm",
    "never mind",
    "undo",
    "on second thought",
    "scratch that",
)

# Privacy tripwire: anything that looks like a path / email / secret must never
# reach the emitted metrics. If it does, the run fails loudly instead of shipping.
_SECRET = re.compile(
    r"sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|pk_live_|AIza[0-9A-Za-z_-]{20,}|xox[bapr]-"
)


def _leaky(s):
    return ("/" in s) or ("\\" in s) or ("@" in s) or bool(_SECRET.search(s))


def _audit(obj, path="metrics"):
    """Recursively fail if any emitted key/value looks like a path/email/secret."""
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


def model_family(name):
    """Collapse an exact model build string to a coarse family so no dated build
    identifier / internal codename (e.g. 'claude-fable-5', '<synthetic>') leaks."""
    n = (name or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in n:
            return fam
    return "other"


def hpath(fp):
    """One-way hash of a file path — preserves identity for counting (churn,
    distinct files, re-reads) while making the raw path unrecoverable."""
    return hashlib.sha1(fp.encode("utf-8", "replace")).hexdigest()


def parse_ts(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def blocks(msg):
    c = msg.get("content") if isinstance(msg, dict) else None
    return c if isinstance(c, list) else []


def mcp_server(name):
    # mcp__<server>__<tool>  ->  <server>
    if isinstance(name, str) and name.startswith("mcp__"):
        parts = name.split("__")
        return parts[1] if len(parts) > 1 else "mcp"
    return None


def user_text(content):
    """Text of a *real* user prompt (not a tool_result carrier)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "text"
            and isinstance(b.get("text"), str)
        )
    return ""


def is_tool_result(content):
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.project_dir) if f.endswith(".jsonl"))
    if not files:
        print("no .jsonl files", file=sys.stderr)
        return 1

    tools = Counter()
    mcp_servers = Counter()
    models = Counter()
    tok = Counter()  # input/output/cache_read/cache_creation
    tok_by_model = defaultdict(Counter)
    hour_hist = Counter()
    weekday_hist = Counter()
    branches = set()
    edited_files = Counter()  # keyed by HASHED path

    n_sessions = 0
    user_turns = assistant_turns = 0
    real_user_prompts = 0
    correction_latencies = []  # seconds: assistant msg -> your corrective reply
    reversal_count = 0  # you reversed your own approved direction
    thinking_blocks = 0
    thinking_chars = 0
    tool_calls = 0
    tool_errors = 0
    bash_interrupted = 0
    reread_events = 0
    total_reads = 0
    multi_tool_turns = 0
    max_parallel = 0
    session_durations = []  # minutes
    session_turns = []
    latest_night = None  # hour of the latest-started assistant activity (0-5 = night)

    for fn in files:
        path = os.path.join(args.project_dir, fn)
        n_sessions += 1
        first_ts = last_ts = None
        seen_reads = set()  # HASHED paths seen this session
        s_turns = 0
        last_assistant_ts = None
        try:
            fh = open(path, "r", errors="replace")
        except Exception:
            continue
        with fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    o = json.loads(ln)
                except Exception:
                    continue
                ts = parse_ts(o.get("timestamp"))
                if ts:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
                    hour_hist[ts.hour] += 1
                    weekday_hist[ts.weekday()] += 1
                    if ts.hour <= 4:
                        latest_night = (
                            ts.hour
                            if latest_night is None
                            else min(latest_night, ts.hour)
                        )
                gb = o.get("gitBranch")
                if gb:
                    branches.add(gb)  # NOTE: not emitted, only counted
                m = o.get("message")
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                if role == "user":
                    user_turns += 1
                    c = m.get("content")
                    if is_tool_result(c):
                        for b in c:
                            if (
                                isinstance(b, dict)
                                and b.get("type") == "tool_result"
                                and b.get("is_error")
                            ):
                                tool_errors += 1
                    else:
                        raw = user_text(c)
                        # Strip injected system context so behavioral tells reflect
                        # YOUR words, not task-notifications / system-reminders.
                        if (
                            "SYSTEM NOTIFICATION - NOT USER INPUT" in raw
                            or "<task-notification>" in raw
                        ):
                            txt = ""
                        else:
                            txt = re.sub(
                                r"<system-reminder>.*?</system-reminder>",
                                " ",
                                raw,
                                flags=re.S,
                            ).strip()
                        if txt:
                            real_user_prompts += 1
                            low = txt.lower()
                            # The Pounce — how fast you jump on a wrong turn.
                            if (
                                last_assistant_ts
                                and ts
                                and any(k in low[:70] for k in CORRECTION)
                            ):
                                dt = (ts - last_assistant_ts).total_seconds()
                                if 0 < dt < 3600:
                                    correction_latencies.append(dt)
                            # The Second-Guesser — reversing your own direction.
                            if any(k in low for k in REVERSAL):
                                reversal_count += 1
                elif role == "assistant":
                    if ts:
                        last_assistant_ts = ts
                    assistant_turns += 1
                    s_turns += 1
                    if m.get("model"):
                        models[m["model"]] += 1
                    u = m.get("usage")
                    if isinstance(u, dict):
                        for k, key in (
                            ("input_tokens", "input"),
                            ("output_tokens", "output"),
                            ("cache_read_input_tokens", "cache_read"),
                            ("cache_creation_input_tokens", "cache_creation"),
                        ):
                            v = u.get(k) or 0
                            tok[key] += v
                            if m.get("model"):
                                tok_by_model[m["model"]][key] += v
                    tuse = 0
                    for b in blocks(m):
                        if not isinstance(b, dict):
                            continue
                        bt = b.get("type")
                        if bt == "thinking":
                            thinking_blocks += 1
                            thinking_chars += len(b.get("thinking") or "")
                        elif bt == "tool_use":
                            tuse += 1
                            tool_calls += 1
                            name = b.get("name") or "?"
                            tools[name] += 1
                            srv = mcp_server(name)
                            if srv:
                                mcp_servers[srv] += 1
                            inp = b.get("input") or {}
                            if name == "Read":
                                total_reads += 1
                                fp = inp.get("file_path")
                                if fp:
                                    h = hpath(fp)
                                    if h in seen_reads:
                                        reread_events += 1
                                    seen_reads.add(h)
                            elif name in ("Edit", "Write"):
                                fp = inp.get("file_path")
                                if fp:
                                    edited_files[hpath(fp)] += 1
                    if tuse > 1:
                        multi_tool_turns += 1
                    max_parallel = max(max_parallel, tuse)
                # bash interruption from toolUseResult
                tr = o.get("toolUseResult")
                if isinstance(tr, dict) and tr.get("interrupted"):
                    bash_interrupted += 1
        if first_ts and last_ts:
            session_durations.append((last_ts - first_ts).total_seconds() / 60.0)
        session_turns.append(s_turns)

    reads = total_reads or 1
    edits = tools.get("Edit", 0)
    writes = tools.get("Write", 0)
    reads_c = tools.get("Read", 0)
    opus = sum(v for k, v in models.items() if "opus" in k)
    total_model_turns = sum(models.values()) or 1
    pounce = (
        round(statistics.median(correction_latencies), 1)
        if correction_latencies
        else None
    )

    # Collapse model keys to a coarse family before emitting (no build strings).
    fam_models = Counter()
    for k, v in models.items():
        fam_models[model_family(k)] += v
    fam_tok = defaultdict(Counter)
    for k, v in tok_by_model.items():
        f = model_family(k)
        for kk, vv in v.items():
            fam_tok[f][kk] += vv

    metrics = {
        "sessions": n_sessions,
        "user_turns": user_turns,
        "real_user_prompts": real_user_prompts,
        "assistant_turns": assistant_turns,
        "pounce_median_sec": pounce,
        "corrections_caught": len(correction_latencies),
        "reversal_count": reversal_count,
        "reversal_rate_per_100": round(
            100 * reversal_count / (real_user_prompts or 1), 1
        ),
        "tool_calls": tool_calls,
        "top_tools": dict(tools.most_common(15)),
        "read": reads_c,
        "edit": edits,
        "write": writes,
        "read_to_edit_ratio": round(reads_c / (edits or 1), 1),
        "read_to_write_ratio": round(reads_c / (writes or 1), 1),
        "reread_events": reread_events,
        "reread_pct": round(100 * reread_events / reads, 1),
        "bash": tools.get("Bash", 0),
        "bash_interrupted": bash_interrupted,
        "tool_errors": tool_errors,
        "tool_error_pct": round(100 * tool_errors / (tool_calls or 1), 1),
        "thinking_blocks": thinking_blocks,
        "thinking_chars": thinking_chars,
        "avg_thinking_chars_per_turn": round(thinking_chars / (assistant_turns or 1)),
        "multi_tool_turns": multi_tool_turns,
        "max_parallel_tools": max_parallel,
        "models": dict(fam_models),
        "opus_pct": round(100 * opus / total_model_turns, 1),
        "tokens": dict(tok),
        "tokens_by_model": {k: dict(v) for k, v in fam_tok.items()},
        "mcp_calls": sum(mcp_servers.values()),
        "mcp_servers_used": len(mcp_servers),
        "top_mcp_servers": dict(mcp_servers.most_common(10)),
        "distinct_files_edited": len(edited_files),
        "most_churned_file_edits": max(edited_files.values()) if edited_files else 0,
        "git_branches": len(branches),
        "hour_histogram": {str(h): hour_hist.get(h, 0) for h in range(24)},
        "peak_hour": max(hour_hist, key=hour_hist.get) if hour_hist else None,
        "weekday_histogram": {str(d): weekday_hist.get(d, 0) for d in range(7)},
        "night_owl_hour": latest_night,
        "longest_session_min": (
            round(max(session_durations), 1) if session_durations else 0
        ),
        "avg_session_min": (
            round(sum(session_durations) / len(session_durations), 1)
            if session_durations
            else 0
        ),
        "total_active_hours": (
            round(sum(session_durations) / 60.0, 1) if session_durations else 0
        ),
        "max_turns_in_session": max(session_turns) if session_turns else 0,
        "workflow_calls": tools.get("Workflow", 0),
        "agent_calls": tools.get("Agent", 0) + tools.get("Task", 0),
        "todo_calls": tools.get("TodoWrite", 0),
    }

    # Enforce the privacy guarantee before anything is written or printed.
    _audit(metrics)

    out = json.dumps(metrics, indent=2)
    if args.out:
        open(args.out, "w").write(out)
        print(f"wrote {args.out} ({n_sessions} sessions)")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
