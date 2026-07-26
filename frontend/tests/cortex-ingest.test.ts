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
    // The fixtures are ASSEMBLED AT RUNTIME rather than written as literals.
    // A credential scanner matches on shape, so it cannot distinguish a real
    // token from a synthetic one of the same shape — and a fixture that doesn't
    // have the real shape wouldn't test the redactor properly. Concatenating
    // keeps the runtime string realistic while leaving no matchable literal in
    // the source, so gitleaks/trufflehog runs on this repo stay signal, not
    // noise. (A scanner that cries wolf is one that gets ignored.)
    const fakeOpenAI = "sk-" + "proj-" + "A".repeat(28);
    const fakeGitHub = "ghp_" + "B".repeat(36);
    const jsonl = JSON.stringify({
      type: "assistant",
      message: {
        role: "assistant",
        content: `I exported OPENAI_API_KEY=${fakeOpenAI} and used ${fakeGitHub} for the push.`,
      },
    });
    const c = filesToConcepts([{ name: "session.jsonl", text: jsonl }]);
    expect(c[0].text).not.toContain(fakeOpenAI);
    expect(c[0].text).not.toContain(fakeGitHub);
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
      // score is relatedness x (1 - overlap) x same-note discount. crossDomain
      // is an embedding-cluster label reported as context, not a multiplier.
      const expected = ins.evidence!.sim * (1 - ins.evidence!.overlap) * (ins.evidence!.sameDocument ? 0.35 : 1);
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

// §2.2 / §2.3: the surprise score was cosine in disguise — overlap was 0 for
// almost every pair (candidate band and overlap window never met) and the
// cross-domain term was a folder-name string compare, dead for pasted text.
// These pin the fix: overlap now discriminates, the score reorders relative to
// cosine, and domains come from the embeddings.
import { clusterDomains } from "../src/cortex/ingest";

describe("the score actually ranks (§2.2/§2.3)", () => {
  // A realistic-sized corpus: 10 clusters x 18 nodes in 40D with noise. Overlap
  // only discriminates when the graph is much larger than the candidate window
  // (n >> hi), which is the regime the shipped 237-node brain is in and where
  // the review measured overlap = 0 for 54/59. (Tiny pasted corpora, n < ~120,
  // still saturate — a separate limitation noted in the remediation report.)
  const clustered = (clusters = 10, perCluster = 18, dim = 40) => {
    const vecs: number[][] = [];
    const concepts: { id: string; label: string; domain: string; text: string }[] = [];
    let seed = 7;
    const jit = () => ((seed = (1103515245 * seed + 12345) & 0x7fffffff) / 0x7fffffff - 0.5);
    for (let k = 0; k < clusters; k++)
      for (let m = 0; m < perCluster; m++) {
        // one-hot cluster centre (mod dim) plus noise — irregular neighbourhoods
        vecs.push(Array.from({ length: dim }, (_, d) => (d % clusters === k ? 1 : 0) + 0.35 * jit()));
        concepts.push({ id: `c${k}-${m}`, label: `n${k}-${m}`, domain: "note", text: "t" });
      }
    return { concepts, vecs };
  };

  it("overlap is no longer degenerate — it carries real variance", () => {
    const { concepts, vecs } = clustered();
    const insights = buildBrainMap(concepts, vecs, "clustered").insights ?? [];
    expect(insights.length).toBeGreaterThan(3);
    const ovs = insights.map((i) => i.evidence!.overlap);
    // not every pair pinned to 0 (the old failure: 54/59 were exactly 0)
    expect(ovs.some((o) => o > 0)).toBe(true);
    // and they actually differ from one another
    expect(new Set(ovs.map((o) => Math.round(o * 1000))).size).toBeGreaterThan(1);
  });

  it("the score reorders relative to plain cosine", () => {
    const { concepts, vecs } = clustered();
    const insights = buildBrainMap(concepts, vecs, "clustered").insights ?? [];
    const byScore = [...insights];
    const byCosine = [...insights].sort((a, b) => b.evidence!.sim - a.evidence!.sim);
    // if the score were k×cosine, these orders would be identical
    const same = byScore.every((ins, i) => ins === byCosine[i]);
    expect(same).toBe(false);
  });

  it("clusterDomains is embedding-derived and non-constant on a flat corpus", () => {
    // Every concept has domain "note" (the pasted-text / folderless case), which
    // used to make crossDomain uniformly false. Clustering must recover >1 group.
    const { vecs } = clustered();
    const norm = (v: number[]) => {
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      return v.map((x) => x / n);
    };
    const dom = clusterDomains(vecs.map(norm));
    expect(new Set(dom).size).toBeGreaterThan(1);
  });

  it("clusterDomains is deterministic", () => {
    const { vecs } = clustered();
    const norm = (v: number[]) => {
      const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
      return v.map((x) => x / n);
    };
    const unit = vecs.map(norm);
    expect(clusterDomains(unit)).toEqual(clusterDomains(unit));
  });
});

// SECURITY.md: "Secrets are redacted on ingest — before text becomes a neuron,
// an insight, or a shareable card. Agent traces routinely contain API keys;
// without this the watermarked share image would be a leak vector."
//
// That claim had no test at all, and it was false for the single likeliest
// secret to appear in an agent trace. The Python half of this project already
// learned this exact lesson (extract.py's _SECRET carries a comment about it);
// the browser redactor had the same hole.
//
// Every sample below is obviously fake — a shape, never a credential.
import { redactSecrets } from "../src/cortex/text";

describe("secret redaction on ingest", () => {
  const leaks = (s: string) => redactSecrets(s).includes(s.slice(0, 24));

  it("redacts a real-shaped Anthropic key", () => {
    // sk-ant-api03-… : the alphanumeric run after "sk-" breaks on a hyphen
    // after three characters, which is what the old pattern could not survive.
    expect(leaks("sk-ant-api03-" + "A".repeat(40) + "-" + "B".repeat(50))).toBe(false);
  });

  it("redacts the other common key shapes", () => {
    for (const sample of [
      "sk-ant-" + "C".repeat(30),
      "sk-proj-" + "D".repeat(40),
      "sk-" + "E".repeat(48),
      "sk_live_" + "F".repeat(24),
      "ghp_" + "G".repeat(36),
      "github_pat_" + "H".repeat(30),
      "AKIA" + "I".repeat(16),
      "AIza" + "J".repeat(30),
      "xoxb-" + "K".repeat(20),
    ]) {
      expect(leaks(sample), sample.slice(0, 12)).toBe(false);
    }
  });

  it("redacts a key embedded in surrounding trace text", () => {
    const line = 'ANTHROPIC_API_KEY="sk-ant-api03-' + "Z".repeat(60) + '" was exported';
    const out = redactSecrets(line);
    expect(out).not.toContain("sk-ant-api03-ZZZZ");
    expect(out).toContain("was exported"); // ordinary text survives
  });

  it("leaves ordinary prose alone", () => {
    const prose = "The transformer architecture scales with sequence length.";
    expect(redactSecrets(prose)).toBe(prose);
  });
});
