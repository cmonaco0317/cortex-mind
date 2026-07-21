import { describe, it, expect } from "vitest";
import { CuriosityEngine, type BrainGraph } from "../src/cortex/engine";

function ring(n: number): BrainGraph {
  const neighbors: number[][] = [];
  const weights: number[][] = [];
  for (let i = 0; i < n; i++) {
    neighbors.push([(i + 1) % n, (i + n - 1) % n]);
    weights.push([0.9, 0.9]);
  }
  return { n, neighbors, weights, longEdges: [[0, Math.floor(n / 2)]] };
}

describe("CuriosityEngine", () => {
  it("fires the focus neuron and spreads activation to neighbours", () => {
    const engine = new CuriosityEngine(ring(6), {}, () => 0.1);
    const events = engine.step();

    expect(events.some((e) => e.kind === "neuron")).toBe(true);
    expect(events.some((e) => e.kind === "synapse")).toBe(true);
    expect(engine.act[engine.focus]).toBe(1);
    expect(engine.thoughts).toBe(1);

    // at least one neighbour of the focus received spread activation
    const neighbours = ring(6).neighbors[engine.focus];
    expect(neighbours.some((j) => engine.act[j] > 0)).toBe(true);
  });

  it("makes an insight leap when the dream roll passes", () => {
    // rand=0.05 < insightProb(0.13) so an insight fires every step
    const engine = new CuriosityEngine(ring(6), {}, () => 0.05);
    const events = engine.step();
    expect(events.some((e) => e.kind === "insight")).toBe(true);
    expect(engine.insights).toBe(1);
  });

  it("decays activation over time", () => {
    const engine = new CuriosityEngine(ring(6), {}, () => 0.5);
    engine.act[0] = 1;
    engine.decay(1);
    expect(engine.act[0]).toBeLessThan(1);
    expect(engine.act[0]).toBeGreaterThan(0);
  });

  it("takes a curiosity jump (no throw) when the associative roll fails", () => {
    const engine = new CuriosityEngine(ring(6), {}, () => 0.99);
    expect(() => engine.step()).not.toThrow();
    expect(engine.focus).toBeGreaterThanOrEqual(0);
    expect(engine.focus).toBeLessThan(6);
  });

  it("never leaves activation above 1", () => {
    const engine = new CuriosityEngine(ring(6), {}, () => 0.2);
    for (let i = 0; i < 50; i++) {
      engine.step();
      engine.decay(0.13);
    }
    for (let i = 0; i < engine.act.length; i++) {
      expect(engine.act[i]).toBeLessThanOrEqual(1);
    }
  });
});
