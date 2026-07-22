# Cortex · agent-insights

**Read your own Claude Code sessions and find out how you actually work with your AI agent** —
computed 100% locally, rendered as shareable cards.

Point it at your Claude Code project directory. It reads the `.jsonl` session
transcripts, computes deterministic behavioral signals, assigns you one named
**archetype** ("The Pouncer," "The Director," "The Surgeon"…), and produces a set
of ranked, non-obvious insight cards — then renders them as 1200×630 images you can
post.

## Pipeline

```bash
# 1) extract deterministic behavioral signals from your sessions
python3 extract.py ~/.claude/projects/<your-project-dir> --out metrics.json

# 2) turn signals into a ranked archetype + insight cards
python3 taxonomy.py metrics.json --out cards.json --n 10

# 3) render each card to a shareable PNG (opens in your browser)
python3 render.py cards.json --out cards.html
python3 -m http.server 8899   # then open http://localhost:8899/cards.html
```

No dependencies beyond the Python 3 standard library.

## Operator Report (Edge / Tax / Move)

The cards are the *descriptive*, shareable layer. `report.py` is the deeper read:
it turns the same signals into a structured **Edge / Tax / Move** report about how
you operate — strengths to lean into, tradeoffs you're choosing, and one one-week
self-experiment per tradeoff — plus a self-contained `report.html` with a shareable
"my operator edge" card.

```bash
python3 report.py metrics.json                       # writes report.html
python3 report.py metrics.json --out-json report.json # also dump the structured data
```

It is built to be **honest, not flattering**:

- **Attributed to what you did, never to whether it was good.** The data can't know
  if your work was any good, so the report never claims it. Every judgment is tied to
  *your* action (a turn you flagged), not to reality (a "wrong" turn).
- **No misattribution.** It won't charge you for the model's or harness's behavior —
  e.g. the default model, or how the transcript happens to chunk tool calls.
- **Personalized or nothing.** Every line cites a number from *your* sessions; a claim
  that can't cite one doesn't ship.
- **A mandatory "what this can't know" footer**, and no cross-user comparisons (there
  is no backend — it's self-vs-self only).

Thin data gets an honest thin report; it never invents an identity from the absence
of signal. Tested with a property-based suite (`test_report.py`): `python3 -m pytest`.

## Privacy

Everything runs on your machine. The extractor emits **aggregate statistics only** —
counts, ratios, timing, tool/model mix. It never reads or emits file *contents*,
file *paths*, project names, or your prompt text, so a card you share cannot leak
anything about what you were working on.

`metrics.json`, `cards.json`, `cards.html`, `report.json`, and `report.html` are your
personal output and are **git-ignored** — they are regenerable and never committed.
`report.py` runs the same privacy tripwire before it emits anything: the run hard-fails
if any string it's about to write looks like a real path, email, or secret.

## What's solid vs. soft (honest data notes)

Claude Code transcripts interleave injected system context
(`<system-reminder>`, `<task-notification>`, quoted tool output) *into* the user
turns. So **text-based signals are unreliable** and are deliberately dropped
(a naive "most-used word" returns system noise like `task`/`type`; average prompt
length is inflated by injected context).

**Reliable signals** (what the cards use): tool mix, read:edit ratio, re-reads,
model/token spend, thinking, hour-of-day timing, MCP/orchestration counts, todo
usage, file-churn, and two keyword-distinctive behavioral tells — reversals
("actually…/go back") and mid-flight course-corrections ("no/wait/stop" right
after an agent turn).

## Design

The editorial layer (`taxonomy.py`) is the point: *which* patterns become insights
and *how they're worded*. Two rules it enforces:

1. Each card belongs to a signal **family**. The chosen archetype "owns" the
   families it already narrates, and cards in an owned family are demoted — so the
   visible set explores *different* dimensions of you instead of restating the
   archetype three ways.
2. Numbers are shown exactly as measured. Nothing is rounded into something that
   reads as fabricated.
