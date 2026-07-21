// ┌─────────────────────────────────────────────────────────────────────────┐
// │  ⚠️  VERSIONED, BEST-EFFORT ADAPTER — THIS MODULE MAY BREAK.               │
// │                                                                           │
// │  It parses third-party agent/session logs whose schemas Cortex does NOT   │
// │  own (Claude Code / Anthropic `.jsonl` sessions, and generic JSON arrays  │
// │  of records). Those upstream formats can change without notice; when they │
// │  do, THIS is the file that breaks — and it is designed to fail SOFT:      │
// │  it never throws, it returns whatever it could parse plus plain-English   │
// │  `notes` explaining what it couldn't. The rest of Cortex (markdown /      │
// │  plaintext ingest, brain math, rendering) does not depend on these        │
// │  schemas, so a break here degrades one input path, not the whole app.     │
// │                                                                           │
// │  If you're a non-maintainer and traces stop importing: the corpus/notes/  │
// │  paste inputs still work. This module can be updated in isolation.        │
// └─────────────────────────────────────────────────────────────────────────┘

import { slug, firstWords, type Concept, type ParseNote } from "../text";

/** Bump when the assumptions below change. Shown in ingest diagnostics. */
export const AGENT_TRACE_ADAPTER_VERSION = "1.0.0";

// Supported as of v1.0.0:
//  • Claude Code / Anthropic session `.jsonl` / `.ndjson`: one JSON object per
//    line. Text is pulled from `text` | `summary` | `body`, from `content`
//    (string or an array of {type:"text"|"tool_use"|"tool_result", …} blocks),
//    or from `message.content`. Role comes from `role` | `message.role` | `type`.
//  • Generic `.json`: a TOP-LEVEL ARRAY of records shaped like
//    {label|name, text|description|content, domain?}.

export interface ParsedRecords {
  concepts: Concept[];
  notes: ParseNote[];
}

interface TraceEntry {
  text?: unknown;
  summary?: unknown;
  body?: unknown;
  content?: unknown;
  type?: unknown;
  role?: unknown;
  name?: unknown;
  label?: unknown;
  message?: { content?: unknown; role?: unknown };
}

/** Best-effort text extraction from a single content block. */
export function blockText(bl: unknown): string {
  if (typeof bl === "string") return bl;
  if (bl && typeof bl === "object") {
    const b = bl as Record<string, unknown>;
    if (b.type === "text" && typeof b.text === "string") return b.text;
    if (b.type === "tool_use" && typeof b.name === "string")
      return `used tool ${b.name}: ${JSON.stringify(b.input ?? {}).slice(0, 160)}`;
    if (b.type === "tool_result")
      return typeof b.content === "string" ? b.content : JSON.stringify(b.content ?? "").slice(0, 200);
    if (typeof b.text === "string") return b.text;
  }
  return "";
}

/** Best-effort text extraction from a whole entry/record (many known shapes). */
export function entryText(e: TraceEntry): string {
  const parts: string[] = [];
  for (const v of [e.text, e.summary, e.body]) if (typeof v === "string") parts.push(v);
  const content = e.content ?? e.message?.content;
  if (typeof content === "string") parts.push(content);
  else if (Array.isArray(content)) for (const bl of content) parts.push(blockText(bl));
  return parts.filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
}

const MIN_TRACE_TEXT = 30; // an entry needs this many chars to be worth a neuron
const MIN_RECORD_TEXT = 20;

/** Parse a JSONL agent trace (e.g. a Claude Code session) into concepts. */
export function parseJsonlTrace(text: string, base: string): ParsedRecords {
  const out: Concept[] = [];
  const lines = text.split(/\r?\n/);
  let nonEmpty = 0;
  let badJson = 0;
  let tooShort = 0;
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i].trim();
    if (!ln) continue;
    nonEmpty++;
    let e: TraceEntry;
    try {
      e = JSON.parse(ln) as TraceEntry;
    } catch {
      badJson++;
      continue;
    }
    const t = entryText(e);
    if (t.length < MIN_TRACE_TEXT) {
      tooShort++;
      continue;
    }
    const role = typeof e.role === "string" ? e.role : typeof e.message?.role === "string" ? String(e.message.role) : typeof e.type === "string" ? e.type : "entry";
    out.push({
      id: `${slug(base)}-${i}`,
      label: firstWords(t, 6),
      domain: slug(role).slice(0, 20) || "trace",
      text: t.slice(0, 600),
    });
  }
  return { concepts: out, notes: [jsonlNote(base, nonEmpty, badJson, tooShort, out.length)] };
}

function jsonlNote(file: string, nonEmpty: number, badJson: number, tooShort: number, kept: number): ParseNote {
  if (nonEmpty === 0) return { file, level: "error", message: `${file}: no non-empty lines — an agent trace should be one JSON object per line (.jsonl).` };
  const skipped: string[] = [];
  if (badJson) skipped.push(`${badJson} line${badJson === 1 ? "" : "s"} failed to parse as JSON`);
  if (tooShort) skipped.push(`${tooShort} entr${tooShort === 1 ? "y was" : "ies were"} too short (<${MIN_TRACE_TEXT} chars)`);
  const tail = skipped.length ? ` (skipped ${skipped.join("; ")})` : "";
  if (kept === 0) {
    if (badJson === nonEmpty) return { file, level: "error", message: `${file}: none of the ${nonEmpty} lines were valid JSON — is this really a JSONL agent trace? (adapter v${AGENT_TRACE_ADAPTER_VERSION})` };
    return { file, level: "error", message: `${file}: parsed ${nonEmpty} lines but found no usable text${tail}. The trace shape may have changed (adapter v${AGENT_TRACE_ADAPTER_VERSION}).` };
  }
  return { file, level: skipped.length ? "warn" : "info", message: `${file}: ${kept} concept${kept === 1 ? "" : "s"} from ${nonEmpty} trace lines${tail}.` };
}

/** Parse a generic JSON array of records into concepts. */
export function parseJsonArrayRecords(text: string, base: string): ParsedRecords {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch (err) {
    return { concepts: [], notes: [{ file: base, level: "error", message: `${base}: not valid JSON — ${String(err instanceof Error ? err.message : err).slice(0, 120)}` }] };
  }
  if (!Array.isArray(data)) {
    const kind = data === null ? "null" : Array.isArray(data) ? "array" : typeof data;
    return { concepts: [], notes: [{ file: base, level: "error", message: `${base}: top-level JSON is a ${kind}, but this importer expects an ARRAY of records like [{ "label": "...", "text": "..." }]. (If this is an agent session, save it as .jsonl instead.)` }] };
  }
  const out: Concept[] = [];
  let tooShort = 0;
  data.forEach((item, i) => {
    if (item && typeof item === "object") {
      const o = item as Record<string, unknown>;
      const text2 = entryText(o as TraceEntry) || (typeof o.description === "string" ? o.description : "");
      const label = typeof o.label === "string" ? o.label : typeof o.name === "string" ? o.name : firstWords(text2, 6);
      if (text2.length >= MIN_RECORD_TEXT) {
        out.push({
          id: `${slug(base)}-${i}`,
          label: label.slice(0, 60),
          domain: typeof o.domain === "string" ? slug(o.domain).slice(0, 20) : "item",
          text: text2.slice(0, 600),
        });
      } else {
        tooShort++;
      }
    } else {
      tooShort++;
    }
  });
  return { concepts: out, notes: [jsonArrayNote(base, data.length, tooShort, out.length)] };
}

function jsonArrayNote(file: string, total: number, tooShort: number, kept: number): ParseNote {
  if (total === 0) return { file, level: "error", message: `${file}: the JSON array is empty.` };
  if (kept === 0) return { file, level: "error", message: `${file}: ${total} array items but none had enough text (need a "text"/"description"/"content" field ≥${MIN_RECORD_TEXT} chars).` };
  const tail = tooShort ? ` (skipped ${tooShort} without usable text)` : "";
  return { file, level: tooShort ? "warn" : "info", message: `${file}: ${kept} concept${kept === 1 ? "" : "s"} from ${total} array items${tail}.` };
}
