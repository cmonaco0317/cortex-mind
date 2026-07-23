import { describe, it, expect, vi } from "vitest";

// Don't load the real transformers.js (heavy WASM) — we only test the pure
// parsing + brain-build math here, not the embedding call.
vi.mock("@xenova/transformers", () => ({
  pipeline: async () => async () => ({ data: new Float32Array(16) }),
  env: { backends: { onnx: { wasm: {} } } },
}));

import { env } from "@xenova/transformers";
import { filesToConcepts, textToConcepts, buildBrainMap, ingestFiles, ingestText, embedTexts } from "../src/cortex/ingest";

describe("ingest parsers", () => {
  it("splits pasted text into concepts on blank lines", () => {
    const c = textToConcepts("First idea about neurons firing.\n\nSecond idea about reward learning.\n\nThird about embeddings and vectors.");
    expect(c.length).toBe(3);
    expect(c[0].text).toContain("neurons");
  });

  it("turns a markdown file into a titled concept", () => {
    const c = filesToConcepts([{ name: "notes/plasticity.md", text: "# Synaptic Plasticity\n\nConnections strengthen with use." }]);
    expect(c).toHaveLength(1);
    expect(c[0].label).toBe("Synaptic Plasticity");
    expect(c[0].domain).toBe("notes");
  });

  it("parses a Claude Code agent trace (jsonl) into per-turn concepts", () => {
    const jsonl = [
      JSON.stringify({ type: "user", message: { role: "user", content: "Fix the auth race where sessions set before validation." } }),
      JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text: "Checking the token verification path in the login flow now." }, { type: "tool_use", name: "Read", input: { file: "login.ts" } }] } }),
    ].join("\n");
    const c = filesToConcepts([{ name: "session.jsonl", text: jsonl }]);
    expect(c.length).toBe(2);
    expect(c[0].domain).toBe("user");
    expect(c[1].text).toContain("token verification");
  });

  it("redacts secrets from ingested agent traces (leak protection)", () => {
    const jsonl = JSON.stringify({
      type: "assistant",
      message: { role: "assistant", content: "I exported OPENAI_API_KEY=sk-proj-ABCD1234EFGH5678IJKL9012MNOP and used ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 for the push." },
    });
    const c = filesToConcepts([{ name: "session.jsonl", text: jsonl }]);
    expect(c[0].text).not.toMatch(/sk-proj-ABCD/);
    expect(c[0].text).not.toMatch(/ghp_ABCDEFG/);
    expect(c[0].text).toContain("[REDACTED]");
  });

  it("parses a JSON array of items", () => {
    const json = JSON.stringify([
      { label: "Alpha", text: "A concept about attention mechanisms in transformers." },
      { name: "Beta", description: "A concept about the hippocampus and memory consolidation." },
    ]);
    const c = filesToConcepts([{ name: "data.json", text: json }]);
    expect(c.length).toBe(2);
    expect(c[0].label).toBe("Alpha");
  });
});

describe("ingest diagnostics (schema-tolerant — explains failures, never silent)", () => {
  it("explains why invalid JSON couldn't parse", () => {
    const r = ingestFiles([{ name: "broken.json", text: "{ not valid json" }]);
    expect(r.concepts).toHaveLength(0);
    expect(r.notes.some((n) => n.level === "error" && /not valid JSON/i.test(n.message))).toBe(true);
  });

  it("explains that a JSON object (not array) is the wrong shape, with guidance", () => {
    const r = ingestFiles([{ name: "data.json", text: JSON.stringify({ a: 1 }) }]);
    expect(r.concepts).toHaveLength(0);
    expect(r.notes.some((n) => n.level === "error" && /array/i.test(n.message))).toBe(true);
  });

  it("notes a markdown file too short to form a concept", () => {
    const r = ingestFiles([{ name: "tiny.md", text: "# Hi\n\nok" }]);
    expect(r.concepts).toHaveLength(0);
    expect(r.notes.some((n) => n.level === "warn" && /too short/i.test(n.message))).toBe(true);
  });

  it("surfaces that some jsonl lines failed to parse but keeps the good ones", () => {
    const jsonl = [
      JSON.stringify({ role: "user", content: "A sufficiently long line about neural plasticity and learning." }),
      "this is not json at all",
    ].join("\n");
    const r = ingestFiles([{ name: "s.jsonl", text: jsonl }]);
    expect(r.concepts).toHaveLength(1);
    expect(r.notes.some((n) => /failed to parse as JSON/i.test(n.message))).toBe(true);
  });

  it("tells the user when pasted text has no usable chunks", () => {
    const r = ingestText("too short");
    expect(r.concepts).toHaveLength(0);
    expect(r.notes.some((n) => n.level === "error")).toBe(true);
  });

  it("keeps filesToConcepts/textToConcepts back-compat (concept-only)", () => {
    expect(Array.isArray(filesToConcepts([{ name: "a.md", text: "# T\n\nEnough words here to pass the length gate easily." }]))).toBe(true);
    expect(Array.isArray(textToConcepts("Enough words here to pass the length gate easily and form a chunk."))).toBe(true);
  });
});

describe("provable-local runtime (no CDN) — the existential invariant", () => {
  it("forces local model + wasm and disables all remote fetches when embedding", async () => {
    // Mocked pipeline, so this exercises configureLocalRuntime without a real model.
    await embedTexts(["a sufficiently long sentence so the mock extractor runs"]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const e = env as any;
    expect(e.allowLocalModels).toBe(true);
    expect(e.allowRemoteModels).toBe(false); // hard no-CDN: never reach huggingface.co
    expect(e.localModelPath).toBe("/models/");
    expect(e.backends.onnx.wasm.wasmPaths).toBe("/ort/"); // never jsdelivr
    expect(e.backends.onnx.wasm.numThreads).toBe(1); // only the vendored non-threaded wasm
  });
});

describe("buildBrainMap", () => {
  it("builds a valid, spread brain from vectors", () => {
    const n = 14;
    const dim = 16;
    const concepts = Array.from({ length: n }, (_, i) => ({
      id: `c${i}`,
      label: `Concept ${i}`,
      domain: i % 2 ? "a" : "b",
      text: `text ${i}`,
    }));
    const vecs = concepts.map((_, i) =>
      Array.from({ length: dim }, (_, d) => Math.sin((i + 1) * (d + 1) * 0.37)),
    );

    const map = buildBrainMap(concepts, vecs, "test");
    expect(map.neurons).toHaveLength(n);
    expect(map.synapses.length).toBeGreaterThan(0);
    expect(map.insights?.length ?? 0).toBeGreaterThan(0);
    // every synapse references valid, ordered node indices
    for (const s of map.synapses) {
      expect(s.s).toBeLessThan(s.t);
      expect(s.t).toBeLessThan(n);
    }
    // positions are not all collapsed at the origin
    const spread = map.neurons.some((nu) => Math.abs(nu.x) + Math.abs(nu.y) + Math.abs(nu.z) > 1);
    expect(spread).toBe(true);
  });
});

// The README claims insights are "ranked most-surprising-first" and that each
// card explains itself. Both were false: there was no score and no sort, and the
// text was a fill-in-the-blank template that asserted "rarely sit together"
// without measuring anything. These pin the claims to the behaviour.
describe("insight ranking + evidence", () => {
  const build = (n = 20, dim = 24) => {
    const concepts = Array.from({ length: n }, (_, i) => ({
      id: `c${i}`,
      label: `Concept ${i}`,
      domain: i % 3 === 0 ? "alpha" : i % 3 === 1 ? "beta" : "gamma",
      text: `text ${i}`,
    }));
    const vecs = concepts.map((_, i) =>
      Array.from({ length: dim }, (_, d) => Math.sin((i + 1) * (d + 1) * 0.29) + Math.cos((i + 3) * d * 0.11)),
    );
    return buildBrainMap(concepts, vecs, "rank-test");
  };

  it("emits insights sorted by surprise, descending", () => {
    const insights = build().insights ?? [];
    expect(insights.length).toBeGreaterThan(1);
    const scores = insights.map((i) => i.score ?? -1);
    expect(scores.every((s) => s >= 0)).toBe(true);
    for (let i = 1; i < scores.length; i++) expect(scores[i]).toBeLessThanOrEqual(scores[i - 1]);
  });

  it("carries the measurements behind each claim", () => {
    for (const ins of build().insights ?? []) {
      expect(ins.evidence).toBeDefined();
      expect(ins.evidence!.sim).toBeGreaterThan(0);
      expect(ins.evidence!.overlap).toBeGreaterThanOrEqual(0);
      expect(ins.evidence!.overlap).toBeLessThanOrEqual(1);
      // score is relatedness x (1 - overlap) x cross-domain bonus
      const expected = ins.evidence!.sim * (1 - ins.evidence!.overlap) * (ins.evidence!.crossDomain ? 1.15 : 1);
      expect(Math.abs((ins.score ?? 0) - expected)).toBeLessThan(1e-3);
    }
  });

  it("never contradicts itself on same-domain pairs", () => {
    for (const ins of build().insights ?? []) {
      // the old template emitted "rarely sit together, yet the link runs within ai"
      expect(ins.why).not.toMatch(/rarely sit together/i);
      if (!ins.evidence!.crossDomain) expect(ins.why).not.toMatch(/bridging/i);
    }
  });

  it("states the measured neighbour overlap rather than asserting novelty", () => {
    for (const ins of build().insights ?? []) {
      const ov = ins.evidence!.overlap;
      if (ov === 0) expect(ins.why).toMatch(/share no near neighbours/i);
      else expect(ins.why).toMatch(new RegExp(`share only ${Math.round(ov * 100)}%`, "i"));
      expect(ins.why).toContain(ins.evidence!.sim.toFixed(2)); // the cosine is shown
    }
  });

  it("spreads bridges across concepts instead of fixating on one node", () => {
    const insights = build(24).insights ?? [];
    const uses = new Map<number, number>();
    for (const ins of insights) {
      uses.set(ins.s, (uses.get(ins.s) ?? 0) + 1);
      uses.set(ins.t, (uses.get(ins.t) ?? 0) + 1);
    }
    for (const [, count] of uses) expect(count).toBeLessThanOrEqual(2);
  });
});
