// Stable, dependency-free text primitives shared by the ingest core and the
// (may-break) agent-trace adapter. Kept in its own module so the adapter never
// has to import back from ingest.ts (no import cycle) and so the security-
// sensitive redaction lives in one place.

export interface Concept {
  id: string;
  label: string;
  domain: string;
  text: string;
  /** Document this passage came from, when the concept is one slice of a longer
   *  file. Lets the graph tell "two ideas in the same note" apart from "two
   *  ideas that had no reason to meet". Absent for whole-document concepts. */
  source?: string;
}

/** A human-readable diagnostic surfaced to the user ("couldn't parse, here's why"). */
export interface ParseNote {
  file: string;
  level: "info" | "warn" | "error";
  message: string;
}

export function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "c";
}

export function firstWords(s: string, n: number): string {
  return s.split(/\s+/).slice(0, n).join(" ");
}

// Scrub common secrets from ingested text BEFORE it becomes a neuron/insight/
// shareable card. Agent traces (.jsonl) routinely carry API keys, tokens, and
// private keys; without this, the watermarked share card is a leak vector.
export function redactSecrets(s: string): string {
  return s
    .replace(/-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----/g, "[REDACTED KEY]")
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}/g, "[REDACTED JWT]")
    .replace(/\b(?:sk|pk|rk)[-_](?:live|test|proj|ant)?[-_]?[A-Za-z0-9]{16,}\b/gi, "[REDACTED]")
    .replace(/\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g, "[REDACTED]")
    .replace(/\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g, "[REDACTED]")
    .replace(/\bAKIA[0-9A-Z]{16}\b/g, "[REDACTED]")
    .replace(/\bAIza[0-9A-Za-z_-]{20,}\b/g, "[REDACTED]")
    .replace(/\bBearer\s+[A-Za-z0-9._-]{16,}/gi, "Bearer [REDACTED]")
    .replace(/\b(api[_-]?key|secret|token|password|passwd|authorization)\b(["']?\s*[:=]\s*["']?)[A-Za-z0-9._-]{12,}/gi, "$1$2[REDACTED]");
}

export function redactConcept(c: Concept): Concept {
  return { ...c, label: redactSecrets(c.label), text: redactSecrets(c.text) };
}

// --------------------------------------------------------------------------- //
//  Passage splitting
//
//  A whole document used to become ONE neuron: truncated to 600 characters and
//  mean-pooled into a single vector. That threw away most of a long file and
//  made the graph document x document, so "concept x concept insight" oversold
//  what was really coarse document similarity — and a note covering three ideas
//  collapsed into one blurred point sitting between all three.
//
//  Splitting on structure first (headings, then blank lines) keeps each vector
//  about ONE thing, which is the unit the whole surprise metric assumes.
// --------------------------------------------------------------------------- //

/** A paragraph longer than this gets cut on sentence boundaries, so one wall of
 *  text doesn't mean-pool into a single blurred vector. */
const PASSAGE_TARGET = 900;
/** Below this a fragment isn't a concept — it gets glued to its neighbour.
 *  Deliberately small: a blank line is the AUTHOR saying "new idea", and that
 *  signal beats any size heuristic. Passages are not greedily packed to the
 *  target across paragraph breaks; doing that merged three distinct pasted
 *  notes into one neuron. */
const PASSAGE_MIN = 30;

/** Strip markdown to plain prose, preserving nothing but the words. */
export function stripMarkdown(raw: string): string {
  return raw
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, " ")
    .replace(/[#>*_~|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export interface Passage {
  heading: string;
  text: string;
}

/** Break a document into heading-delimited sections, then into paragraph groups
 *  of roughly PASSAGE_TARGET characters. Never splits mid-sentence when it can
 *  avoid it, and folds runt fragments into their neighbour rather than emitting
 *  a neuron with no content. */
export function splitPassages(raw: string, docTitle = ""): Passage[] {
  const lines = raw.split(/\r?\n/);
  const sections: Array<{ heading: string; body: string[] }> = [];
  let current = { heading: docTitle, body: [] as string[] };
  for (const ln of lines) {
    const h = ln.trim().match(/^#{1,6}\s+(.*)/);
    if (h) {
      if (current.body.join(" ").trim()) sections.push(current);
      current = { heading: h[1].trim(), body: [] };
    } else {
      current.body.push(ln);
    }
  }
  if (current.body.join(" ").trim() || sections.length === 0) sections.push(current);

  const out: Passage[] = [];
  for (const sec of sections) {
    const paras = sec.body
      .join("\n")
      .split(/\n\s*\n/)
      .map((p) => stripMarkdown(p))
      .filter((p) => p.length > 0);
    let buf = "";
    const flush = (): void => {
      const t = buf.trim();
      buf = "";
      if (!t) return;
      // A runt tail belongs with what came before, not as its own "concept".
      const prev = out[out.length - 1];
      if (t.length < PASSAGE_MIN && prev && prev.heading === sec.heading) {
        prev.text = `${prev.text} ${t}`.trim();
        return;
      }
      out.push({ heading: sec.heading, text: t });
    };
    for (const p of paras) {
      // A single paragraph longer than the target gets cut on sentence ends.
      if (p.length > PASSAGE_TARGET) {
        flush();
        for (const piece of chunkSentences(p, PASSAGE_TARGET)) out.push({ heading: sec.heading, text: piece });
        continue;
      }
      buf = buf ? `${buf} ${p}` : p;
      // Respect the author's paragraph break as soon as we have enough text to
      // stand alone. Only genuine fragments keep accumulating.
      if (buf.length >= PASSAGE_MIN) flush();
    }
    flush();
  }
  return out.filter((p) => p.text.length >= PASSAGE_MIN);
}

/** Split a long run of prose on sentence boundaries into ~size chunks. */
export function chunkSentences(text: string, size: number): string[] {
  const sentences = text.match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g) || [text];
  const out: string[] = [];
  let buf = "";
  for (const s of sentences) {
    if (buf && buf.length + s.length > size) {
      out.push(buf.trim());
      buf = "";
    }
    buf += s;
  }
  if (buf.trim()) out.push(buf.trim());
  return out.filter(Boolean);
}
