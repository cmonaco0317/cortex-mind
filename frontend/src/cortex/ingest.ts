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
import { slug, firstWords, redactConcept, splitPassages, type Concept, type ParseNote } from "./text";
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

/**
 * The "why" line for a bridge.
 *
 * This replaces a fill-in-the-blank template that asserted every pair "rarely
 * sit together" without ever checking, and contradicted itself on same-domain
 * pairs ("rarely sit together, yet the link runs within ai"). Every clause here
 * is a statement about a number that was actually measured, and the numbers
 * ride along in `evidence` so a reader can check the claim instead of trusting
 * it. It is still a sentence built from measurements, not generated prose --
 * generated explanations need a model, which the offline `build_brain.py`
 * pipeline uses and the in-browser path deliberately does not.
 */
function bridgeInsight(
  a: Concept,
  b: Concept,
  sim: number,
  overlap: number,
  cross: boolean,
  sameDoc = false,
): { why: string; angle: string } {
  const relatedness =
    sim >= 0.75 ? "Strongly related" : sim >= 0.55 ? "Clearly related" : "Loosely related";
  const separation =
    overlap === 0
      ? "they share no near neighbours at all"
      : `they share only ${Math.round(overlap * 100)}% of their near neighbours`;
  const where = sameDoc
    ? `both from “${a.source}” — the same note, so you already wrote them side by side`
    : cross
      ? `bridging ${a.domain} and ${b.domain}`
      : `inside ${a.domain}, between two groups that otherwise don't touch`;
  return {
    why:
      `${relatedness} (cosine ${sim.toFixed(2)}), yet ${separation} — ` +
      `a link ${where}${sameDoc ? "." : " that neither one's own neighbourhood would surface."}`,
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
/** Ceiling on neurons from a single file, so one 400-page document can't drown
 *  out every other source in the graph. */
const MAX_PASSAGES_PER_DOC = 40;
/** Per-passage text cap. Larger than the old whole-document 600 because a
 *  passage is now one idea rather than a whole file, and mean-pooling a single
 *  idea over ~1000 chars still yields a sharp vector. */
const PASSAGE_TEXT_CAP = 1200;

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
      const base = f.name.replace(/\.[^.]+$/, "");
      const docLabel = (title || parts[parts.length - 1].replace(/\.[^.]+$/, "")).slice(0, 60);
      // One file used to become one neuron. Now it becomes one neuron per
      // passage, so a note covering three ideas stops collapsing into a single
      // blurred point sitting between all three.
      const passages = splitPassages(f.text, title).slice(0, MAX_PASSAGES_PER_DOC);
      if (passages.length > 1) {
        notes.push({ file: f.name, level: "info", message: `${f.name}: split into ${passages.length} passages.` });
      }
      passages.forEach((p, i) => {
        const label = p.heading && p.heading !== title
          ? p.heading
          : passages.length > 1
            ? `${docLabel} — ${firstWords(p.text, 5)}`
            : docLabel;
        collected.push({
          id: slug(passages.length > 1 ? `${base}-${i}` : base),
          label: label.slice(0, 60),
          domain: domain || "note",
          text: (p.heading ? `${p.heading}. ${p.text}` : p.text).slice(0, PASSAGE_TEXT_CAP),
          source: docLabel,
        });
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
  const concepts = splitPassages(raw)
    .slice(0, MAX_PASSAGES_PER_DOC)
    .map((p, i) => ({
      id: `paste-${i}`,
      label: (p.heading || firstWords(p.text, 6)).slice(0, 60),
      domain: "note",
      text: (p.heading ? `${p.heading}. ${p.text}` : p.text).slice(0, PASSAGE_TEXT_CAP),
    }))
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

  // ---- surprise scoring -------------------------------------------------
  //
  // What makes a connection non-obvious? Not distance: two unrelated concepts
  // are far apart and boring. The interesting case is a pair that is genuinely
  // RELATED yet sits in two neighbourhoods that never touch -- two clusters
  // meeting at a single point. So:
  //
  //     surprise = relatedness x (1 - neighbourhoodOverlap) x domainBonus
  //
  // All three terms are measured in the FULL embedding space. The previous
  // version ranked by distance in the 3D PCA layout, which is a projection
  // artifact: a pair lands far apart in 3D precisely when PCA discarded the
  // axis on which they agree, so "most surprising" partly meant "worst
  // projected". Nothing was sorted either, despite the claim of ranking.
  const NBR = Math.min(12, Math.max(2, n - 1));
  const nbrSet = order.map((row) => new Set(row.slice(0, NBR)));

  const overlapOf = (i: number, j: number): number => {
    const a = nbrSet[i];
    const b = nbrSet[j];
    let inter = 0;
    for (const x of a) if (b.has(x)) inter++;
    const union = a.size + b.size - inter;
    return union ? inter / union : 0;
  };

  const nBridges = Math.max(8, Math.floor(n / 4));
  const lo = Math.min(12, Math.max(2, Math.floor(n / 4)));
  const hi = Math.min(60, n - 1);

  type Cand = { i: number; j: number; score: number; sim: number; overlap: number; cross: boolean; sameDoc: boolean };
  const cands: Cand[] = [];
  const scored = new Set<string>();
  for (let i = 0; i < n; i++) {
    for (const j of order[i].slice(lo, hi)) {
      const key = edgeKey(i, j);
      if (scored.has(key)) continue; // each unordered pair scored once
      scored.add(key);
      const rel = sim[i][j];
      if (rel <= 0) continue; // unrelated isn't surprising, it's noise
      const ov = overlapOf(i, j);
      const cross = concepts[i].domain !== concepts[j].domain;
      // Now that one document yields many passages, two chunks of the SAME note
      // are the commonest high-similarity pair in the graph — and the least
      // interesting: the author already put those ideas side by side. Nothing
      // was discovered, so they're heavily discounted rather than allowed to
      // crowd out genuine cross-source bridges.
      const sameDoc =
        concepts[i].source !== undefined && concepts[i].source === concepts[j].source;
      cands.push({
        i,
        j,
        sim: rel,
        overlap: ov,
        cross,
        sameDoc,
        score: rel * (1 - ov) * (cross ? 1.15 : 1) * (sameDoc ? 0.35 : 1),
      });
    }
  }
  // Actually ranked, most-surprising-first -- the thing the README promised.
  cands.sort((a, b) => b.score - a.score);

  // Diversity guard: without it the single most "bridgeable" concept takes
  // every slot and the card deck is one node over and over.
  const MAX_PER_CONCEPT = 2;
  const usedCount = new Map<number, number>();
  const insights: BrainInsight[] = [];
  for (const c of cands) {
    if (insights.length >= nBridges) break;
    if ((usedCount.get(c.i) ?? 0) >= MAX_PER_CONCEPT) continue;
    if ((usedCount.get(c.j) ?? 0) >= MAX_PER_CONCEPT) continue;
    usedCount.set(c.i, (usedCount.get(c.i) ?? 0) + 1);
    usedCount.set(c.j, (usedCount.get(c.j) ?? 0) + 1);
    const s = Math.min(c.i, c.j);
    const t = Math.max(c.i, c.j);
    edges.set(edgeKey(c.i, c.j), {
      s,
      t,
      w: Math.round(c.sim * 1e4) / 1e4,
      long: true,
    });
    const card = bridgeInsight(concepts[c.i], concepts[c.j], c.sim, c.overlap, c.cross, c.sameDoc);
    insights.push({
      s,
      t,
      why: card.why,
      angle: card.angle,
      score: Math.round(c.score * 1e4) / 1e4,
      evidence: {
        sim: Math.round(c.sim * 1e4) / 1e4,
        overlap: Math.round(c.overlap * 1e4) / 1e4,
        crossDomain: c.cross,
        sameDocument: c.sameDoc,
      },
    });
  }

  const neurons: BrainNeuron[] = concepts.map((c, i) => ({
    id: c.id,
    label: c.label,
    domain: c.domain,
    x: Math.round(coords[i][0] * 100) / 100,
    y: Math.round(coords[i][1] * 100) / 100,
    z: Math.round(coords[i][2] * 100) / 100,
    snippet: c.text.slice(0, 140),
    ...(c.source ? { source: c.source } : {}),
  }));
  const synapses = [...edges.values()];
  return {
    meta: { name, count: n, dim, k: kk, synapses: synapses.length, insights: insights.length },
    neurons,
    synapses,
    insights,
  };
}
