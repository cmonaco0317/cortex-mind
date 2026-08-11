import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { buildBrainMap, clusterDomains, folderDomain, cleanMarkdown } from "../src/cortex/ingest";
import { splitPassages } from "../src/cortex/text";

/**
 * Cross-language conformance: the TypeScript half (§2.4).
 *
 * build_brain.py and frontend/src/cortex/ implement the same algorithm twice.
 * The only contract between them used to be a comment saying "mirrors
 * splitPassages()", and they had already drifted — the same nested corpus got
 * different domain labels depending on which pipeline ran it.
 *
 * These read the SAME conformance/fixture.json and conformance/golden.json that
 * test_conformance.py reads. Either language drifting turns CI red.
 */
const root = resolve(__dirname, "..", "..");
const fixture = JSON.parse(readFileSync(resolve(root, "conformance/fixture.json"), "utf8"));
const golden = JSON.parse(readFileSync(resolve(root, "conformance/golden.json"), "utf8"));

const norm = (v: number[]): number[] => {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
};

describe("cross-language conformance (§2.4)", () => {
  it("folderDomain matches the golden", () => {
    const got = fixture.paths.map((p: string[]) => folderDomain(p));
    expect(got).toEqual(golden.folderDomains);
  });

  it("passage splitting matches the golden", () => {
    const got = fixture.docs.map((d: { text: string }) => {
      const title = cleanMarkdown(d.text).title;
      return splitPassages(d.text, title).map((p) => ({
        heading: p.heading ?? "",
        text: p.text,
      }));
    });
    expect(got).toEqual(golden.passages);
  });

  it("clusterDomains matches the golden", () => {
    const got = clusterDomains(fixture.vectors.map(norm));
    expect(got).toEqual(golden.clusters);
  });

  it("bridge ranking matches the golden", () => {
    const map = buildBrainMap(fixture.concepts, fixture.vectors, "conformance");
    const got = (map.insights ?? []).map((i) => ({
      s: i.s,
      t: i.t,
      score: i.score,
      sim: i.evidence!.sim,
      overlap: i.evidence!.overlap,
      crossDomain: i.evidence!.crossDomain,
      sameDocument: i.evidence!.sameDocument,
    }));
    expect(got).toEqual(golden.bridges);
  });
});
