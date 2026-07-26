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
import os
from collections import Counter
import sys

from extract import model_family

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
    # Turns on a genuinely smaller model FAMILY. Two things this must not do:
    # name a build ("Opus 4.8" — false for anyone who ran 4.6, and stale every
    # release), and count the placeholder/unknown-codename buckets as "smaller"
    # (extract collapses <synthetic> and unrecognised codenames into `other`, so
    # the old `k != "synthetic"` test never matched the collapsed key and the
    # card overstated how often a lighter model was actually used).
    smaller = sum(
        v
        for k, v in m.get("models", {}).items()
        if model_family(k) in ("haiku", "sonnet")
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
            f"{fmt(smaller)} turns out of {fmt(assistant)} ran on a smaller model. When the call is hard, you don't hand it down.",
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


# --------------------------------------------------------------------------- #
# Validating the archetype (3.2)
# --------------------------------------------------------------------------- #
# Wilson intervals, lift mining and leave-one-out sign checks are real machinery,
# but they all sit DOWNSTREAM of a construct nobody validated. There is no ground
# truth for "The Pouncer", no second rater, and nothing that would fail if labels
# were assigned at random within their gates. Rigour downstream of an unvalidated
# construct produces confident output, not evidence.
#
# Two checks, and the label is gated on the first:
#   - STABILITY: leave one session out and see whether the label survives. A
#     label that flips when a single session is removed is describing that
#     session, not you.
#   - NEGATIVE CONTROL: shuffle event order within sessions. Anything that claims
#     to read ORDER must collapse. Whatever survives was never reading order.

# A label that cannot survive this fraction of leave-one-session-out folds is not
# reported as an identity. It is not a p-value and is not presented as one: it is
# the share of folds that agreed, and the threshold is an editorial line drawn
# before the number was measured.
STABILITY_MIN = 0.70

# Leave-one-out costs one full aggregation per fold. Above this many sessions a
# deterministic spread is sampled instead, and the fold count is reported so the
# figure is never mistaken for all-of-them.
MAX_FOLDS = 40


def _fold_files(files, max_folds):
    if len(files) <= max_folds:
        return list(files)
    step = len(files) / float(max_folds)
    return [files[int(i * step)] for i in range(max_folds)]


def archetype_stability(project_dir, files=None, max_folds=MAX_FOLDS):
    """How often the archetype survives dropping one session.

    Returns None when there are too few sessions to say anything -- which is a
    real answer, and callers must not read it as "stable".
    """
    import extract as E

    if files is None:
        files = sorted(f for f in os.listdir(project_dir) if f.endswith(".jsonl"))
    if len(files) < 5:
        return None

    full = compute_archetype(E.aggregate(project_dir, files))
    full_name = full["name"] if full else None

    folds = _fold_files(files, max_folds)
    names = []
    for drop in folds:
        subset = [f for f in files if f != drop]
        a = compute_archetype(E.aggregate(project_dir, subset))
        names.append(a["name"] if a else None)

    agree = sum(1 for n in names if n == full_name)
    others = Counter(n for n in names if n != full_name)
    return {
        "label": full_name,
        "folds": len(names),
        "sessions": len(files),
        "sampled_folds": len(folds) < len(files),
        "agreement": round(agree / len(names), 3),
        "flip_rate": round(1 - agree / len(names), 3),
        "runners_up": dict(others.most_common(3)),
        "stable": (agree / len(names)) >= STABILITY_MIN,
    }


# Metrics a reader would reasonably assume depend on the ORDER of events within a
# session. The control CLASSIFIES them empirically rather than trusting this list
# -- which is the point, because the list was wrong when first written:
#
#   corrections_caught  98 -> 31       collapses (a correction needs an assistant
#   pounce_median_sec   2.7s -> 1411s  turn to follow, so order is the signal)
#   reread_pct          41.1 -> 41.1   UNCHANGED: it counts whether a file was read
#                                      more than once, which is a set membership
#                                      question, not an ordering one
#   reversal_count      29 -> 29       UNCHANGED: keyword matches in your prompts
#
# So two of the four are order-independent, and no card may claim otherwise.
_ORDER_CANDIDATES = (
    "corrections_caught",
    "pounce_median_sec",
    "reread_pct",
    "reversal_count",
)


def shuffled_control(project_dir, files=None, seed=1):
    """Negative control: the same pipeline over shuffled within-session order.

    An archetype that survives shuffling is not reading order, whatever its card
    says. This reports both the label and the order-dependent metrics either
    side, so "it survived" is a measurement rather than an impression.
    """
    import extract as E

    if files is None:
        files = sorted(f for f in os.listdir(project_dir) if f.endswith(".jsonl"))
    real = E.aggregate(project_dir, files)
    shuf = E.aggregate(project_dir, files, shuffle_seed=seed)
    a_real = compute_archetype(real)
    a_shuf = compute_archetype(shuf)
    moved = {k: (real.get(k), shuf.get(k)) for k in _ORDER_CANDIDATES if real.get(k) != shuf.get(k)}
    unmoved = {k: real.get(k) for k in _ORDER_CANDIDATES if real.get(k) == shuf.get(k)}
    return {
        "label_real": a_real["name"] if a_real else None,
        "label_shuffled": a_shuf["name"] if a_shuf else None,
        "label_survived_shuffle": bool(
            a_real and a_shuf and a_real["name"] == a_shuf["name"]
        ),
        # measured, not assumed: what actually moved when order was destroyed
        "order_dependent": {k: {"real": v[0], "shuffled": v[1]} for k, v in moved.items()},
        "order_independent": unmoved,
        "order_signal_collapsed": bool(moved),
    }


def compute_archetype(m):
    """Assign ONE named archetype from DEEP behavior — an identity, not a stat.

    Archetypes describe *style*, so every discriminating condition is a RATE.
    They used to be absolute counts (corrections >= 40, dispatches >= 80,
    files_edited >= 40), which made the label a proxy for how heavily you use
    Claude Code rather than how you use it: run enough sessions and you become
    "The Pouncer" purely by accumulating corrections, without correcting any
    more often than anyone else.

    Absolute counts survive only as MINIMUM SAMPLE SIZE gates — you can't
    characterise someone's style from three files — never as the style signal.

    The winner is the best-FITTING archetype: each candidate's score is its
    editorial weight scaled by how far its rates clear their thresholds, so a
    profile that barely trips two triggers doesn't beat one that strongly
    matches a single archetype. It used to be a fixed hardcoded priority.
    """
    tools = m.get("top_tools", {})
    top_mcp = m.get("top_mcp_servers", {})
    browser = sum(
        v
        for k, v in top_mcp.items()
        if any(x in k.lower() for x in ("chrome", "browser", "playwright"))
    )
    hh = m.get("hour_histogram", {})
    before_1pm = sum(int(hh.get(str(h), 0)) for h in range(6, 13))
    tool_calls = m.get("tool_calls") or 1
    assistant = m.get("assistant_turns") or 1
    bash_pct = round(100 * tools.get("Bash", 0) / tool_calls)
    dispatches = m.get("workflow_calls", 0) + m.get("agent_calls", 0)
    r2e = m.get("read_to_edit_ratio", 9)
    rev = m.get("reversal_rate_per_100", 0)
    files_edited = m.get("distinct_files_edited", 0)
    epf = round(m.get("edit", 0) / files_edited, 1) if files_edited else 9

    # --- rates: the actual style signals -----------------------------------
    # Denominator matters. A correction is something the USER does, so it
    # normalises by user prompts -- the same denominator reversal_rate_per_100
    # already uses. Dividing by assistant turns instead would understate a
    # hands-on operator whose every prompt steers several assistant turns.
    prompts = m.get("real_user_prompts") or assistant
    corrections = m.get("corrections_caught", 0)
    corrections_per_100 = 100 * corrections / prompts
    dispatch_pct = 100 * dispatches / tool_calls
    browser_pct = 100 * browser / tool_calls

    # --- sample-size floors: enough evidence to say anything at all ---------
    ENOUGH_PROMPTS = prompts >= 50
    ENOUGH_CALLS = tool_calls >= 300
    ENOUGH_FILES = files_edited >= 15

    cands = []

    def A(name, tag, defn, traits, score, fit=1.0):
        # `fit` (>=1) is how decisively the rates clear their thresholds. It
        # breaks ties on strength of match instead of a fixed pecking order.
        cands.append(
            {
                "kind": "archetype",
                "name": name,
                "tagline": tag,
                "definition": defn,
                "traits": traits,
                "score": round(score * min(fit, 1.6), 2),
            }
        )

    # Style = corrects OFTEN (per turn), not corrects A LOT (in total).
    if ENOUGH_PROMPTS and corrections_per_100 >= 8 and rev >= 4:
        A(
            "The Pouncer",
            "watches every move — cuts in the instant a turn drifts",
            "You don't hand off and walk away. You hover, and the second a turn looks like it's heading somewhere you didn't ask for, you pounce.",
            [
                f"{corrections_per_100:.1f} course-corrections per 100 prompts ({fmt(corrections)} total)",
                f"reverses course 1 in {round(100/rev)} prompts",
                f"{r2e}× read:edit — acts, never browses",
            ],
            95,
            fit=corrections_per_100 / 8,
        )
    if ENOUGH_CALLS and dispatch_pct >= 2:
        A(
            "The Director",
            "doesn't do the work — runs a fleet of agents",
            "You stopped being the hands. Now you dispatch, review, and redirect a team of subagents.",
            [
                f"{dispatch_pct:.1f}% of tool calls are dispatches ({fmt(dispatches)} total)",
                f"{m.get('opus_pct')}% Opus — nothing but the top model",
                f"{fmt(m.get('mcp_servers_used', 0))} MCP servers wired",
            ],
            88,
            fit=dispatch_pct / 2,
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
    # files_edited is a sample-size floor here, not the trait; epf and
    # reread_pct (both rates) are what actually decide it.
    if ENOUGH_FILES and epf <= 2.0 and m.get("reread_pct", 100) < 20:
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
    if ENOUGH_CALLS and browser_pct >= 5:
        A(
            "The Puppeteer",
            "gave the agent hands",
            "Your agent doesn't just write code — it clicks, types, and drives the real web for you.",
            [
                f"{browser_pct:.1f}% of tool calls drive a browser ({fmt(browser)} total)",
                "navigates live sites",
                "an operator, not a coder",
            ],
            72,
            fit=browser_pct / 5,
        )

    cands.sort(key=lambda c: -c["score"])
    return cands[0] if cands else None


def gate_archetype(arch, stability):
    """Attach the measured stability to the label, and withhold the label
    entirely when it is not stable enough to be an identity.

    Below STABILITY_MIN the name is not shown at all: a label that changes when
    one session is dropped is describing that session. The card says what was
    measured instead of quietly presenting a coin flip as a personality.
    """
    if arch is None:
        return None
    out = dict(arch)
    if stability is None:
        out["stability"] = None
        out["stability_note"] = (
            "Not enough sessions to test whether this label survives dropping one. "
            "Read it as a description of the data so far, not an identity."
        )
        return out
    out["stability"] = {
        "agreement": stability["agreement"],
        "flip_rate": stability["flip_rate"],
        "folds": stability["folds"],
        "stable": stability["stable"],
    }
    if not stability["stable"]:
        alts = ", ".join(stability["runners_up"]) or "a different label"
        out["withheld"] = True
        out["name"] = "No stable archetype"
        out["tagline"] = "your sessions do not agree on one"
        out["definition"] = (
            "Dropping a single session changes this label %.0f%% of the time "
            "(it becomes %s). That is not an identity, so it is not being shown "
            "as one." % (100 * stability["flip_rate"], alts)
        )
        out["traits"] = [
            "%d of %d leave-one-out folds agreed" % (
                round(stability["agreement"] * stability["folds"]), stability["folds"]
            ),
            "threshold for showing a label: %.0f%% agreement" % (100 * STABILITY_MIN),
            "the ranked cards below are unaffected — they are per-signal, not a label",
        ]
    else:
        out["stability_note"] = (
            "Survived %.0f%% of leave-one-session-out folds (%d tested)."
            % (100 * stability["agreement"], stability["folds"])
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", default="")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument(
        "--sessions",
        default="",
        help="session dir; enables the leave-one-out stability check and the "
        "shuffled-order negative control (3.2)",
    )
    args = ap.parse_args()
    m = json.load(open(args.metrics))
    arch = compute_archetype(m)
    stability = control = None
    if args.sessions:
        stability = archetype_stability(args.sessions)
        control = shuffled_control(args.sessions)
    arch = gate_archetype(arch, stability) if args.sessions else arch
    cards = apply_diversity(build_cards(m), arch)[: args.n]
    out_list = ([arch] if arch else []) + cards
    if args.out:
        open(args.out, "w").write(json.dumps(out_list, indent=2, ensure_ascii=False))
    if arch:
        print(f"\n★ ARCHETYPE: {arch['name']} — {arch['tagline']}")
        print(f"   {arch['definition']}")
        for t in arch["traits"]:
            print(f"     · {t}")
        if arch.get("stability_note"):
            print(f"   {arch['stability_note']}")
    # Printed whether flattering or not: a measurement that only appears when it
    # is favourable is not a measurement.
    if stability:
        print(
            "\n  stability: %.0f%% agreement across %d leave-one-session-out folds "
            "(flip rate %.0f%%)%s"
            % (
                100 * stability["agreement"],
                stability["folds"],
                100 * stability["flip_rate"],
                "" if stability["stable"] else "  — BELOW THRESHOLD, label withheld",
            )
        )
    if control:
        print(
            "  negative control (shuffled event order): label %s"
            % (
                "SURVIVED — it is not reading order"
                if control["label_survived_shuffle"]
                else "collapsed %s -> %s, as an order-dependent label should"
                % (control["label_real"], control["label_shuffled"])
            )
        )
        for k, v in control["order_dependent"].items():
            print("    %-20s %s -> %s  (order-dependent)" % (k, v["real"], v["shuffled"]))
        for k, v in control["order_independent"].items():
            print("    %-20s %s (unchanged — NOT an order signal)" % (k, v))
    for i, c in enumerate(cards, 1):
        print(f"\n{i}. [{c['category']}]  {c['hero']}  —  {c['title']}")
        print(f"   {c['sub']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
