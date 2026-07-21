// In-browser corpus -> brain pipeline. Everything here runs client-side: text is
// embedded locally (transformers.js) and the brain map (PCA layout + synapse
// graph) is built in JS. Your data never leaves the browser.
//
// This module is the STABLE core: local embedding, the brain-build math, and the
// simple/robust text inputs (markdown, plaintext, pasted notes). The fragile,
// schema-coupled inputs (agent traces, generic JSON records) live in the
// explicitly-versioned, may-break ./adapters/agent-trace module so a break there
// can't take down the rest of the app.

import { pipeline, env } from "@xenova/transformers";
import type { BrainMap, BrainNeuron, BrainSynapse, BrainInsight } from "./types";
import { slug, firstWords, redactConcept, type Concept, type ParseNote } from "./text";
import { parseJsonlTrace, parseJsonArrayRecords } from "./adapters/agent-trace";

// Re-exported so callers can keep importing these from "./ingest".
export type { Concept, ParseNote } from "./text";
export { redactSecrets } from "./text";
export { AGENT_TRACE_ADAPTER_VERSION } from "./adapters/agent-trace";

const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2"; // 384-dim, ~23MB, fast in WASM

// Provable-local runtime: the embedding model AND the ONNX-runtime WebAssembly
// are both loaded from files bundled with the app (public/models + public/ort),
// never a CDN. This closes the two hidden network dependencies transformers.js
// has by default (huggingface.co for weights, cdn.jsdelivr.net for the wasm) —
// which is both the "survives with no maintainer" fix and the credibility proof
// that nothing but the page itself is ever fetched. Configured lazily so the
// mocked unit tests never touch the real runtime.
let runtimeConfigured = false;
function configureLocalRuntime(): void {
  if (runtimeConfigured) return;
  runtimeConfigured = true;
  const base = import.meta.env.BASE_URL || "/"; // subpath-safe (e.g. GitHub Pages)
  env.allowLocalModels = true;
  env.allowRemoteModels = false; // hard no-CDN: never reach out to huggingface.co
  env.localModelPath = `${base}models/`;
  const wasm = env.backends?.onnx?.wasm;
  if (wasm) {
    wasm.wasmPaths = `${base}ort/`;
    wasm.numThreads = 1; // non-threaded → only the vendored ort-wasm(-simd).wasm load
  }
}

export interface Progress {
  stage: "model" | "embed" | "build";
  loaded?: number; // 0..1 for model download
  i?: number;
  total?: number;
}

type Extractor = (text: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>;
let extractor: Extractor | null = null;

async function getExtractor(onProgress?: (p: Progress) => void): Promise<Extractor> {
  if (!extractor) {
    configureLocalRuntime();
    extractor = (await pipeline("feature-extraction", EMBED_MODEL, {
      quantized: true, // pin the vendored onnx/model_quantized.onnx explicitly
      progress_callback: (e: { status?: string; progress?: number }) => {
        if (e.status === "progress" && typeof e.progress === "number") {
          onProgress?.({ stage: "model", loaded: e.progress / 100 });
        }
      },
    })) as unknown as Extractor;
  }
  return extractor;
}

export async function embedTexts(
  texts: string[],
  onProgress?: (p: Progress) => void,
): Promise<number[][]> {
  const ex = await getExtractor(onProgress);
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    const res = await ex(texts[i] || " ", { pooling: "mean", normalize: true });
    out.push(Array.from(res.data));
    onProgress?.({ stage: "embed", i: i + 1, total: texts.length });
  }
  return out;
}

// ---- small numeric helpers (no numpy in the browser) ----

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normalize(v: number[]): number[] {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map((x) => x / n);
}

/** Top principal components via power iteration + deflation (avoids forming DxD). */
function topPCs(rows: number[][], num: number): number[][] {
  const N = rows.length;
  const D = rows[0].length;
  const data = rows.map((r) => r.slice());
  const rng = mulberry32(7);
  const comps: number[][] = [];
  for (let c = 0; c < num; c++) {
    let v = normalize(Array.from({ length: D }, () => rng() - 0.5));
    for (let it = 0; it < 40; it++) {
      const u = data.map((row) => dot(row, v)); // N
      const w = new Array(D).fill(0);
      for (let i = 0; i < N; i++) {
        const ui = u[i];
        const row = data[i];
        for (let d = 0; d < D; d++) w[d] += row[d] * ui;
      }
      v = normalize(w);
    }
    comps.push(v);
    for (let i = 0; i < N; i++) {
      const p = dot(data[i], v);
      const row = data[i];
      for (let d = 0; d < D; d++) row[d] -= p * v[d];
    }
  }
  return comps;
}

function templateInsight(a: Concept, b: Concept): { why: string; angle: string } {
  const bridge = a.domain !== b.domain ? `across ${a.domain} and ${b.domain}` : `within ${a.domain}`;
  return {
    why: `“${a.label}” and “${b.label}” rarely sit together, yet the link runs ${bridge}.`,
    angle: `What would “${a.label}” look like reframed through “${b.label}”?`,
  };
}

// ---- turn dropped files / pasted text / agent traces into concepts ----

const TEXT_EXTS = /\.(md|markdown|mdx|txt|text|rst)$/i;

export function isIngestable(name: string): boolean {
  return TEXT_EXTS.test(name) || /\.(jsonl|ndjson|json)$/i.test(name);
}

function cleanMarkdown(raw: string): { title: string; text: string } {
  let title = "";
  for (const ln of raw.split(/\r?\n/)) {
    const m = ln.trim().match(/^#{1,6}\s+(.*)/);
    if (m) {
      title = m[1].trim();
      break;
    }
  }
  const text = raw
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`[^`]*`/g, " ")
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[#>*_~|-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return { title, text };
}

const MIN_MD_TEXT = 30;

/** The result of turning raw inputs into concepts, plus plain-English diagnostics. */
export interface IngestResult {
  concepts: Concept[];
  notes: ParseNote[];
}

/**
 * Turn dropped/chosen files into concepts. Routes each file by extension to the
 * right parser and collects human-readable notes so the UI can tell the user
 * exactly what did and didn't parse ("couldn't parse X, here's why").
 */
export function ingestFiles(files: Array<{ name: string; text: string }>): IngestResult {
  const collected: Concept[] = [];
  const notes: ParseNote[] = [];
  for (const f of files) {
    if (/\.(jsonl|ndjson)$/i.test(f.name)) {
      const r = parseJsonlTrace(f.text, f.name);
      collected.push(...r.concepts);
      notes.push(...r.notes);
    } else if (/\.json$/i.test(f.name)) {
      const r = parseJsonArrayRecords(f.text, f.name);
      collected.push(...r.concepts);
      notes.push(...r.notes);
    } else {
      const { title, text } = cleanMarkdown(f.text);
      if (text.length < MIN_MD_TEXT) {
        notes.push({ file: f.name, level: "warn", message: `${f.name}: only ${text.length} chars of text after stripping formatting — too short to form a concept (need ≥${MIN_MD_TEXT}).` });
        continue;
      }
      const parts = f.name.split(/[/\\]/);
      const domain = parts.length > 1 ? slug(parts[parts.length - 2]).slice(0, 20) : "note";
      collected.push({
        id: slug(f.name.replace(/\.[^.]+$/, "")),
        label: (title || parts[parts.length - 1].replace(/\.[^.]+$/, "")).slice(0, 60),
        domain: domain || "note",
        text: `${title}. ${text}`.slice(0, 600),
      });
    }
  }
  // de-dup ids, then scrub secrets before anything becomes a neuron/card
  const seen = new Set<string>();
  const deduped = collected.filter((c) => (seen.has(c.id) ? false : (seen.add(c.id), true)));
  const dropped = collected.length - deduped.length;
  if (dropped > 0) notes.push({ file: "(all files)", level: "info", message: `Merged ${dropped} concept${dropped === 1 ? "" : "s"} with duplicate ids.` });
  return { concepts: deduped.map(redactConcept), notes };
}

/** Split pasted freeform text into concept-sized chunks (by headings / blank lines). */
export function ingestText(raw: string): IngestResult {
  const blocks = raw.split(/\n\s*\n|(?=^#{1,6}\s)/m).map((b) => b.trim()).filter((b) => b.length >= MIN_MD_TEXT);
  const concepts = blocks
    .map((b, i) => {
      const { title, text } = cleanMarkdown(b);
      return {
        id: `paste-${i}`,
        label: (title || firstWords(text, 6)).slice(0, 60),
        domain: "note",
        text: text.slice(0, 600),
      };
    })
    .map(redactConcept);
  const notes: ParseNote[] = concepts.length
    ? [{ file: "(pasted text)", level: "info", message: `${concepts.length} concept${concepts.length === 1 ? "" : "s"} from pasted text.` }]
    : [{ file: "(pasted text)", level: "error", message: `No chunks ≥${MIN_MD_TEXT} chars found. Paste a few paragraphs separated by blank lines (or Markdown headings).` }];
  return { concepts, notes };
}

// Back-compat, concept-only wrappers (kept so existing callers/tests are stable).
export function filesToConcepts(files: Array<{ name: string; text: string }>): Concept[] {
  return ingestFiles(files).concepts;
}

export function textToConcepts(raw: string): Concept[] {
  return ingestText(raw).concepts;
}

export function buildBrainMap(
  concepts: Concept[],
  vecs: number[][],
  name: string,
  k = 8,
): BrainMap {
  const n = concepts.length;
  const dim = vecs[0].length;
  const unit = vecs.map(normalize);

  // center + PCA -> 3D
  const mean = new Array(dim).fill(0);
  for (const v of unit) for (let d = 0; d < dim; d++) mean[d] += v[d] / n;
  const centered = unit.map((v) => v.map((x, d) => x - mean[d]));
  const comps = topPCs(centered, 3);
  let coords = centered.map((row) => comps.map((c) => dot(row, c)));

  // normalize axes -> ellipsoid + organic jitter
  const radii = [26, 18, 21];
  const rng = mulberry32(11);
  for (let axis = 0; axis < 3; axis++) {
    const vals = coords.map((c) => c[axis]);
    const m = vals.reduce((s, x) => s + x, 0) / n;
    const sd = Math.sqrt(vals.reduce((s, x) => s + (x - m) ** 2, 0) / n) || 1;
    for (let i = 0; i < n; i++) {
      coords[i][axis] = ((coords[i][axis] - m) / sd) * radii[axis] + (rng() - 0.5) * 2.8;
    }
  }

  // cosine similarity (unit vectors) -> kNN synapses + long bridges
  const sim: number[][] = unit.map((a) => unit.map((b) => dot(a, b)));
  for (let i = 0; i < n; i++) sim[i][i] = -1;
  const order = sim.map((row) =>
    row.map((_, j) => j).sort((x, y) => row[y] - row[x]),
  );

  const edgeKey = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const edges = new Map<string, BrainSynapse>();
  const kk = Math.min(k, n - 1);
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < kk; r++) {
      const j = order[i][r];
      const key = edgeKey(i, j);
      const w = Math.round(sim[i][j] * 1e4) / 1e4;
      const cur = edges.get(key);
      if (!cur || w > cur.w) edges.set(key, { s: Math.min(i, j), t: Math.max(i, j), w, long: false });
    }
  }

  const insights: BrainInsight[] = [];
  const nBridges = Math.max(8, Math.floor(n / 4));
  const seen = new Set<number>();
  for (let b = 0; b < nBridges; b++) {
    const i = Math.floor(rng() * n);
    if (seen.has(i)) continue;
    seen.add(i);
    // moderately-similar-but-not-nearest, scaled so small corpora still bridge
    const lo = Math.min(12, Math.max(2, Math.floor(n / 4)));
    const band = order[i].slice(lo, Math.min(60, n - 1));
    if (!band.length) continue;
    let far = band[0];
    let farD = -1;
    for (const j of band) {
      const d =
        (coords[i][0] - coords[j][0]) ** 2 +
        (coords[i][1] - coords[j][1]) ** 2 +
        (coords[i][2] - coords[j][2]) ** 2;
      if (d > farD) {
        farD = d;
        far = j;
      }
    }
    const key = edgeKey(i, far);
    edges.set(key, { s: Math.min(i, far), t: Math.max(i, far), w: Math.round(sim[i][far] * 1e4) / 1e4, long: true });
    const card = templateInsight(concepts[i], concepts[far]);
    insights.push({ s: Math.min(i, far), t: Math.max(i, far), why: card.why, angle: card.angle });
  }

  const neurons: BrainNeuron[] = concepts.map((c, i) => ({
    id: c.id,
    label: c.label,
    domain: c.domain,
    x: Math.round(coords[i][0] * 100) / 100,
    y: Math.round(coords[i][1] * 100) / 100,
    z: Math.round(coords[i][2] * 100) / 100,
    snippet: c.text.slice(0, 140),
  }));
  const synapses = [...edges.values()];
  return {
    meta: { name, count: n, dim, k: kk, synapses: synapses.length, insights: insights.length },
    neurons,
    synapses,
    insights,
  };
}
