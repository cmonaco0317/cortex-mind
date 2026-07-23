import { describe, it, expect, vi } from "vitest";

vi.mock("@xenova/transformers", () => ({
  pipeline: async () => async () => ({ data: new Float32Array(16) }),
  env: { backends: { onnx: { wasm: {} } } },
}));

import { splitPassages, chunkSentences } from "../src/cortex/text";
import { ingestFiles, ingestText } from "../src/cortex/ingest";

// A document used to become ONE neuron: truncated to 600 chars and mean-pooled
// into a single vector. Three ideas in one note collapsed into one blurred point
// sitting between all three, and everything past 600 characters was discarded —
// which made "concept × concept insight" oversell a document-similarity graph.
const MULTI_IDEA = `# Field Notes

## Spaced repetition
Recall is strengthened by retrieving information at increasing intervals rather
than by rereading it. The forgetting curve flattens each time the memory is
successfully reconstructed from scratch. Rereading produces a strong feeling of
familiarity that is easily mistaken for knowledge, which is why learners who
reread rate themselves as better prepared and then perform worse than learners
who quizzed themselves and felt uncertain the whole time.

## Sleep consolidation
During slow-wave sleep the hippocampus replays the day's episodes to the cortex.
This is when a fragile trace becomes a durable one, which is why an all-nighter
buys hours and costs retention. The replay is compressed, running many times
faster than the original experience, and the cortical trace that survives is a
generalisation rather than a recording — the gist is kept and the incidental
detail is dropped, which is both the strength and the failure mode.

## Interleaving
Mixing problem types within a study session feels worse and works better than
blocking them. The difficulty is the mechanism, not a side effect of it.
`;

describe("passage splitting (concept-level, not document-level)", () => {
  it("splits a multi-idea document on its headings", () => {
    const p = splitPassages(MULTI_IDEA, "Field Notes");
    expect(p.length).toBeGreaterThanOrEqual(3);
    const headings = p.map((x) => x.heading);
    expect(headings).toContain("Spaced repetition");
    expect(headings).toContain("Sleep consolidation");
    expect(headings).toContain("Interleaving");
  });

  it("keeps each passage about ONE thing", () => {
    for (const p of splitPassages(MULTI_IDEA, "Field Notes")) {
      const others = ["hippocampus", "Interleaving", "forgetting curve"].filter((t) =>
        p.text.toLowerCase().includes(t.toLowerCase()),
      );
      expect(others.length).toBeLessThanOrEqual(1); // no passage straddles two sections
    }
  });

  it("turns one file into several labelled, traceable neurons", () => {
    const r = ingestFiles([{ name: "notes/memory.md", text: MULTI_IDEA }]);
    expect(r.concepts.length).toBeGreaterThanOrEqual(3);
    for (const c of r.concepts) {
      expect(c.source).toBe("Field Notes"); // provenance back to the document
      expect(c.domain).toBe("notes");
    }
    expect(new Set(r.concepts.map((c) => c.id)).size).toBe(r.concepts.length); // unique ids
    expect(r.notes.some((n) => /split into \d+ passages/.test(n.message))).toBe(true);
  });

  it("no longer discards everything past the old 600-character cap", () => {
    // Reproduce exactly what the old path produced: strip markdown, then keep
    // the first 600 characters of the whole document as ONE concept.
    const oldCleaned = MULTI_IDEA.replace(/```[\s\S]*?```/g, " ")
      .replace(/`[^`]*`/g, " ")
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
      .replace(/[#>*_~|-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const oldKept = `Field Notes. ${oldCleaned}`.slice(0, 600);

    const r = ingestFiles([{ name: "n/long.md", text: MULTI_IDEA }]);
    const all = r.concepts.map((c) => c.text).join(" ");

    // The last section fell outside the old window and was silently lost.
    expect(oldKept).not.toMatch(/interleav/i);
    expect(all).toMatch(/interleav/i);
    expect(all).toMatch(/difficulty is the mechanism/i);
    // and the new path retains materially more of the document than 600 chars
    expect(all.length).toBeGreaterThan(oldKept.length);
  });

  it("respects the author's blank lines rather than repacking to a size target", () => {
    const c = ingestText("First idea about neurons firing.\n\nSecond idea about reward learning.\n\nThird about embeddings and vectors.");
    expect(c.concepts).toHaveLength(3);
  });

  it("glues true fragments onto their neighbour instead of emitting empty concepts", () => {
    const p = splitPassages("A real paragraph with enough substance to stand on its own here.\n\nok\n");
    expect(p).toHaveLength(1);
    expect(p[0].text).toMatch(/ok$/);
  });

  it("splits a wall of text on sentence boundaries", () => {
    const wall = Array.from({ length: 60 }, (_, i) => `Sentence number ${i} carries a little meaning.`).join(" ");
    const p = splitPassages(wall);
    expect(p.length).toBeGreaterThan(1);
    for (const x of p) expect(x.text.length).toBeLessThan(1400);
    expect(chunkSentences("One. Two. Three.", 8).length).toBeGreaterThan(1);
  });

  it("caps how many neurons a single document may contribute", () => {
    const huge = Array.from({ length: 200 }, (_, i) => `## Section ${i}\nA paragraph about topic number ${i} with enough text.`).join("\n\n");
    const r = ingestFiles([{ name: "big.md", text: huge }]);
    expect(r.concepts.length).toBeLessThanOrEqual(40);
  });
});
