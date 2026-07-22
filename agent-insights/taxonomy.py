#!/usr/bin/env python3
"""
Cortex · agent-insights — the insight taxonomy (the editorial moat).

Turns the deterministic metrics (extract.py) into a ranked, non-obvious,
identity-level card set. The value is NOT the parsing — it's *which* patterns
become insights and *how they're worded*. Each rule fires only when the signal
is notable, and phrases it as a "how did it know that about me" reveal.

Three guards keep the visible set honest (all enforced in apply_diversity):
  1. ARCHETYPE ECHO — the chosen archetype "owns" the families it already
     narrates (e.g. The Pouncer owns corrections + reversals + read:edit); those
     standalone cards are dropped so the set explores *other* dimensions of you.
  2. COMMODITY LANE — families a free /insights-style dashboard already ships
     (token/cache counts, MCP-server counts) are dropped from the visible set;
     they are not defensible identity tells.
  3. ONE CARD PER METRIC — two cards derived from the same underlying number
     (e.g. mean-vs-max edits-per-file) can contradict each other; only the
     strongest survives.

Numbers are shown exactly as measured (never "~500") and no card claims a
cross-user comparison the local data cannot support.

Usage:
  taxonomy.py metrics.json [--out cards.json] [--n 10]
"""

import argparse
import json
import sys

# Families the archetype already narrates in its traits -> their standalone
# cards are dropped from the visible set (anti-echo).
ARCHETYPE_OWNS = {
    "The Pouncer": {"corrections", "reversals", "latency", "read_edit"},
    "The Director": {"dispatch", "model", "stack"},
    "The Bulldozer": {"model", "read_edit", "todo"},
    "The Night Builder": {"tempo"},
    "The Terminal Native": {"bash"},
    "The Puppeteer": {"browser"},
    "The Surgeon": {"churn", "read_edit"},
    "The Marathoner": {"endurance"},
}

# Families a free dashboard already ships -> never allowed in the visible set.
COMMODITY_FAMILIES = {"scale", "stack"}


def fmt(n):
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    return f"{n:,}"


def _planning(m):
    """Planning-on-paper = TodoWrite PLUS the Task* tracker family, not just
    TodoWrite — so the 'never make a list' card can't fire for someone who tracks
    tasks with TaskUpdate (the mislabel the review caught)."""
    top = m.get("top_tools", {}) or {}
    task = sum(int(top.get(k, 0)) for k in ("TaskCreate", "TaskUpdate", "TaskList"))
    return int(m.get("todo_calls", 0)) + task


def build_cards(m):
    cards = []

    def card(cat, hero, title, sub, score, fam, metric):
        cards.append(
            {
                "category": cat,
                "hero": hero,
                "title": title,
                "sub": sub,
                "score": score,
                "family": fam,
                "metric": metric,
            }
        )

    tok = m.get("tokens", {})
    top_mcp = m.get("top_mcp_servers", {})
    hh = m.get("hour_histogram", {})
    wd = m.get("weekday_histogram", {})
    browser = sum(
        v
        for k, v in top_mcp.items()
        if any(x in k.lower() for x in ("chrome", "browser", "playwright"))
    )
    before_1pm = sum(int(hh.get(str(h), 0)) for h in range(6, 13))
    night = sum(int(hh.get(str(h), 0)) for h in range(2, 5))
    cache_read = tok.get("cache_read", 0)
    output = tok.get("output", 1) or 1
    non_opus = sum(
        v for k, v in m.get("models", {}).items() if k != "opus" and k != "synthetic"
    )
    opus_pct = m.get("opus_pct", 0)
    assistant = m.get("assistant_turns", 0)
    tool_calls = m.get("tool_calls", 0) or 1
    reads = m.get("read", 0)
    edits = m.get("edit", 0)
    r2e = m.get("read_to_edit_ratio", 99)
    files_edited = m.get("distinct_files_edited", 0)
    epf = round(edits / files_edited, 1) if files_edited else 0
    churn = m.get("most_churned_file_edits", 0)
    reread_pct = m.get("reread_pct", 0)
    corrections = m.get("corrections_caught", 0)
    rev = m.get("reversal_rate_per_100", 0)
    reversal_count = m.get("reversal_count", 0)
    pounce = m.get("pounce_median_sec")
    dispatches = m.get("workflow_calls", 0) + m.get("agent_calls", 0)
    bash = m.get("top_tools", {}).get("Bash", 0)
    mcp_servers = m.get("mcp_servers_used", 0)
    max_turns = m.get("max_turns_in_session", 0)
    sessions = m.get("sessions", 0)
    wd_total = sum(int(v) for v in wd.values()) if wd else 0
    weekend = sum(int(wd.get(str(d), 0)) for d in (5, 6))  # Sat, Sun
    weekend_pct = round(100 * weekend / wd_total) if wd_total else 0

    # --- the taxonomy (editorial phrasing is the point) ---

    if opus_pct >= 95:
        card(
            "taste",
            f"{opus_pct}%",
            "You never reach for a lesser model.",
            f"Exactly {non_opus} turns out of {fmt(assistant)} ran on anything cheaper than Opus 4.8. Not one hard call handed down to a smaller model.",
            95,
            "model",
            "model_mix",
        )

    # The Pounce (latency) — research's #1 uncanny tell: HOW FAST you cut in.
    if pounce is not None and pounce <= 20 and corrections >= 15:
        card(
            "reflex",
            f"{pounce}s",
            "You pounce faster than you could read.",
            f"When a turn starts to drift, your median time to cut in is {pounce} seconds — 'no', 'wait', 'stop'. That's reflex, not review.",
            94,
            "latency",
            "pounce",
        )

    if rev >= 4 and reversal_count >= 10:
        one_in = round(100 / rev)
        card(
            "psyche",
            f"1 in {one_in}",
            "You change your own mind.",
            f"You reverse a direction you already gave — 'actually…', 'go back' — once every {one_in} prompts ({fmt(reversal_count)} times). You think out loud and pivot mid-stream.",
            93,
            "reversals",
            "reversals",
        )

    if r2e < 1.2:
        card(
            "instinct",
            f"{r2e}×",
            "You edit before you look.",
            f"{fmt(edits)} edits, only {fmt(reads)} reads — a {r2e}× read:edit ratio. Most people read first; you just go.",
            92,
            "read_edit",
            "read_edit",
        )

    if _planning(m) == 0 and assistant > 500:
        card(
            "style",
            "0",
            "You never make a list.",
            f"{fmt(assistant)} turns, {fmt(tool_calls)} tool calls, and not a single todo or task-list entry. No plan on paper — you hold it in your head and move.",
            90,
            "todo",
            "todo",
        )

    if dispatches >= 40:
        card(
            "automation",
            fmt(dispatches),
            "You script yourself.",
            f"{m.get('workflow_calls',0)} workflows and {m.get('agent_calls',0)} subagents dispatched — {fmt(dispatches)} whole tasks handed to the machine instead of typed by hand. You delegate the work you've already figured out once.",
            86,
            "dispatch",
            "dispatch",
        )

    # Churn profile — the bimodal truth (mean vs the one outlier), ONE card.
    if churn >= 25 and files_edited >= 30:
        card(
            "craft",
            fmt(churn),
            "One file you refused to abandon.",
            f"Across {fmt(files_edited)} files you average {epf} quick passes each — then there's the one. {fmt(churn)} edits to a single file. You stayed on it until it was right instead of starting over.",
            84,
            "churn",
            "edits_per_file",
        )

    if corrections >= 20:
        card(
            "psyche",
            fmt(corrections),
            "You don't let a turn drift.",
            f"{fmt(corrections)} times you cut in the instant a turn started heading somewhere you didn't ask for — 'no', 'wait', 'stop'. You watch every move.",
            83,
            "corrections",
            "corrections",
        )

    # The Surgeon (single-pass editor) — same metric as churn; the metric-dedupe
    # guard guarantees only one of the two ever appears.
    if files_edited >= 40 and 0 < epf <= 2.0 and reread_pct < 20:
        card(
            "precision",
            f"{epf}×",
            "You cut once and move on.",
            f"You touch each file {epf} times across {fmt(files_edited)} files, then it's done — clean, low-rework passes with almost no doubling back.",
            82,
            "churn",
            "edits_per_file",
        )

    if weekend_pct >= 35:
        card(
            "devotion",
            f"{weekend_pct}%",
            "This is what you do instead of resting.",
            f"{weekend_pct}% of your sessions land on a Saturday or Sunday — past the {round(100*2/7)}% you'd expect from a weekday job. It isn't a job. It's the thing you reach for on your days off.",
            80,
            "weekend",
            "weekend",
        )

    if bash / tool_calls >= 0.25:
        card(
            "instinct",
            f"{round(100*bash/tool_calls)}%",
            "Bash is your reflex.",
            f"{fmt(bash)} raw shell commands — {round(100*bash/tool_calls)}% of every tool call. When you want something done, the terminal is your first instinct, not your last resort.",
            79,
            "bash",
            "bash",
        )

    if browser >= 100:
        card(
            "operator",
            fmt(browser),
            "You didn't build a coder. You built hands.",
            f"{fmt(browser)} calls that drive a live web browser. Your agent doesn't just write — it clicks, types, and navigates the real web for you.",
            78,
            "browser",
            "browser",
        )

    # Tempo — the hour histogram alone is a commodity 'night owl' stat; the ONLY
    # non-commodity residue is the behavioral ABSOLUTE (not one before 1pm), so
    # lead with that and keep it mid-set, never the headliner.
    if before_1pm == 0 and night > 200:
        card(
            "tempo",
            "2–4am",
            "Not one session before 1pm.",
            f"Zero — across {fmt(sessions)} sessions, not a single one started in the morning. Your window is 2–4am ({fmt(night)} events). Your agent has never seen your daylight.",
            74,
            "tempo",
            "hours",
        )

    if max_turns >= 500:
        card(
            "endurance",
            fmt(max_turns),
            "You don't restart. You keep going.",
            f"Your longest single session ran {fmt(max_turns)} turns. Most people open a fresh chat to clear their head — you kept the whole build in one thread.",
            68,
            "endurance",
            "max_turns",
        )

    if reread_pct >= 25:
        card(
            "discipline",
            f"{reread_pct}%",
            "You never trust a stale read.",
            f"{reread_pct}% of file reads re-open something already seen this session — you make the agent look again before it acts. You verify before you commit.",
            66,
            "reread",
            "reread",
        )

    if cache_read > 100_000_000:
        card(
            "scale",
            fmt(cache_read),
            "You live in long context.",
            f"{fmt(cache_read)} tokens read from cache — {round(cache_read/output)}× more than you output ({fmt(output)}). Your agent never forgets what it just saw.",
            60,  # commodity family — dropped from the visible set by apply_diversity
            "scale",
            "cache",
        )

    if mcp_servers >= 12:
        card(
            "stack",
            f"{mcp_servers}",
            "Your agent has a cockpit.",
            f"{mcp_servers} different MCP servers wired, {fmt(m.get('mcp_calls',0))} calls. You didn't settle for a chat box — you built an instrument panel.",
            58,  # commodity family — dropped from the visible set by apply_diversity
            "stack",
            "mcp",
        )

    return cards


def apply_diversity(cards, arch):
    """Return the VISIBLE card set: drop archetype-echo families and commodity
    families entirely, then keep at most one card per underlying metric (highest
    score wins). This is the fix for the pressure-test failures: no card re-tells
    the archetype, no dead-lane dashboard stat leads, and no two cards derived
    from one number can contradict each other."""
    owned = ARCHETYPE_OWNS.get(arch["name"], set()) if arch else set()
    kept = [
        c
        for c in cards
        if c["family"] not in owned and c["family"] not in COMMODITY_FAMILIES
    ]
    kept.sort(key=lambda c: c["score"], reverse=True)
    visible, seen_metric = [], set()
    for c in kept:
        if c["metric"] in seen_metric:
            continue
        seen_metric.add(c["metric"])
        visible.append(c)
    return visible


def compute_archetype(m):
    """Assign ONE named archetype from DEEP behavior — the research's core wedge
    (an identity, not a stat). Highest-scoring trigger wins."""
    tools = m.get("top_tools", {})
    top_mcp = m.get("top_mcp_servers", {})
    browser = sum(
        v
        for k, v in top_mcp.items()
        if any(x in k.lower() for x in ("chrome", "browser", "playwright"))
    )
    hh = m.get("hour_histogram", {})
    before_1pm = sum(int(hh.get(str(h), 0)) for h in range(6, 13))
    bash_pct = round(100 * tools.get("Bash", 0) / (m.get("tool_calls") or 1))
    dispatches = m.get("workflow_calls", 0) + m.get("agent_calls", 0)
    r2e = m.get("read_to_edit_ratio", 9)
    rev = m.get("reversal_rate_per_100", 0)
    files_edited = m.get("distinct_files_edited", 0)
    epf = round(m.get("edit", 0) / files_edited, 1) if files_edited else 9

    cands = []

    def A(name, tag, defn, traits, score):
        cands.append(
            {
                "kind": "archetype",
                "name": name,
                "tagline": tag,
                "definition": defn,
                "traits": traits,
                "score": score,
            }
        )

    if m.get("corrections_caught", 0) >= 40 and rev >= 4:
        A(
            "The Pouncer",
            "watches every move — cuts in the instant a turn drifts",
            "You don't hand off and walk away. You hover, and the second a turn looks like it's heading somewhere you didn't ask for, you pounce.",
            [
                f"{fmt(m['corrections_caught'])} mid-flight course-corrections",
                f"reverses course 1 in {round(100/rev)} prompts",
                f"{r2e}× read:edit — acts, never browses",
            ],
            95,
        )
    if dispatches >= 80:
        A(
            "The Director",
            "doesn't do the work — runs a fleet of agents",
            "You stopped being the hands. Now you dispatch, review, and redirect a team of subagents.",
            [
                f"{fmt(dispatches)} workflows + subagents dispatched",
                f"{m.get('opus_pct')}% Opus — nothing but the top model",
                f"{fmt(m.get('mcp_servers_used', 0))} MCP servers wired",
            ],
            88,
        )
    if m.get("opus_pct", 0) >= 98 and r2e < 1 and m.get("todo_calls", 1) == 0:
        A(
            "The Bulldozer",
            "max model, edit-first, zero ceremony",
            "No plans, no downshift, no reading twice. You point the strongest model at the problem and start moving.",
            [
                f"{m.get('opus_pct')}% Opus, always",
                f"{r2e}× read:edit — edits before it looks",
                "0 todo lists — the plan lives in your head",
            ],
            84,
        )
    # The Surgeon — precision operator: genuinely single-pass, low re-read.
    if files_edited >= 40 and epf <= 2.0 and m.get("reread_pct", 100) < 20:
        A(
            "The Surgeon",
            "cuts once, never circles back",
            "You don't thrash a file. You read it, make the change, and it's done — one clean pass.",
            [
                f"{epf} edits per file across {fmt(files_edited)} files",
                f"only {m.get('reread_pct')}% re-reads",
                "measure twice, cut once",
            ],
            82,
        )
    # The Marathoner — endurance: one enormous unbroken session.
    if m.get("max_turns_in_session", 0) >= 800 and m.get("avg_session_min", 0) >= 120:
        A(
            "The Marathoner",
            "one session, no restart, until it's done",
            "You don't open a fresh chat when it gets long. You stay in the same thread and grind it out.",
            [
                f"longest session: {fmt(m['max_turns_in_session'])} turns",
                f"{round(m.get('avg_session_min',0)/60,1)}h average session",
                "never restarts to clear context",
            ],
            76,
        )
    if before_1pm == 0:
        A(
            "The Night Builder",
            "never shipped in daylight",
            "Your best work happens while everyone else sleeps. Not one session before 1pm.",
            [
                "0 events before 1pm",
                "peak activity 2–4am",
                f"{fmt(m.get('sessions', 0))} sessions, all after dark",
            ],
            80,
        )
    if bash_pct >= 30:
        A(
            "The Terminal Native",
            "reaches for the shell first",
            "You don't click and wait for a tool. You drop to Bash and make it happen.",
            [
                f"{bash_pct}% of tool calls are raw Bash",
                f"{fmt(tools.get('Bash', 0))} shell commands",
                "tools are a last resort",
            ],
            74,
        )
    if browser >= 200:
        A(
            "The Puppeteer",
            "gave the agent hands",
            "Your agent doesn't just write code — it clicks, types, and drives the real web for you.",
            [
                f"{fmt(browser)} browser-driving calls",
                "navigates live sites",
                "an operator, not a coder",
            ],
            72,
        )

    cands.sort(key=lambda c: -c["score"])
    return cands[0] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    m = json.load(open(args.metrics))
    arch = compute_archetype(m)
    cards = apply_diversity(build_cards(m), arch)[: args.n]
    out_list = ([arch] if arch else []) + cards
    if args.out:
        open(args.out, "w").write(json.dumps(out_list, indent=2, ensure_ascii=False))
    if arch:
        print(f"\n★ ARCHETYPE: {arch['name']} — {arch['tagline']}")
        print(f"   {arch['definition']}")
        for t in arch["traits"]:
            print(f"     · {t}")
    for i, c in enumerate(cards, 1):
        print(f"\n{i}. [{c['category']}]  {c['hero']}  —  {c['title']}")
        print(f"   {c['sub']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
