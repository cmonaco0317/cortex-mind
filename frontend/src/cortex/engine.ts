// Curiosity engine — a live "train of thought" that runs
// spreading-activation over a real semantic graph (neurons = concepts, synapses
// = embedding associations). Nothing here is scripted: the sequence of firings
// emerges from the graph structure plus a curiosity policy (attend -> spread ->
// dream-jump). This is what makes the brain "real" rather than an animation.

export interface BrainGraph {
  n: number;
  neighbors: number[][]; // adjacency list per neuron
  weights: number[][]; // association strength, parallel to neighbors
  longEdges: Array<[number, number]>; // long-range "insight" bridges
}

export type FireKind = "neuron" | "synapse" | "insight";

export interface FireEvent {
  kind: FireKind;
  a: number;
  b?: number;
  w?: number;
}

function novelty(visits: number): number {
  return 1 / (1 + visits);
}

function pickWeighted(items: number[], scores: number[], rand: () => number): number {
  let total = 0;
  for (const s of scores) total += Math.max(0, s);
  if (total <= 0) return items[Math.floor(rand() * items.length)];
  let r = rand() * total;
  for (let i = 0; i < items.length; i++) {
    r -= Math.max(0, scores[i]);
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

export interface EngineParams {
  decay: number; // per-second activation decay factor
  spread: number; // fraction of activation pushed to neighbours
  associativeProb: number; // chance to follow a synapse vs. curiosity-jump
  insightProb: number; // chance of a dream/insight leap per step
}

const DEFAULTS: EngineParams = {
  decay: 0.35,
  spread: 0.6,
  associativeProb: 0.72,
  insightProb: 0.13,
};

export class CuriosityEngine {
  readonly act: Float32Array;
  readonly visits: Float32Array;
  focus: number;
  thoughts = 0;
  insights = 0;

  private readonly p: EngineParams;

  constructor(
    private readonly g: BrainGraph,
    params: Partial<EngineParams> = {},
    private readonly rand: () => number = Math.random,
  ) {
    this.p = { ...DEFAULTS, ...params };
    this.act = new Float32Array(g.n);
    this.visits = new Float32Array(g.n);
    this.focus = Math.floor(this.rand() * g.n);
  }

  /** Advance one "thought": attend to a neuron, spread activation, maybe dream. */
  step(): FireEvent[] {
    const g = this.g;
    const events: FireEvent[] = [];

    // 1) Attend — continue the train of thought along a strong/novel synapse,
    //    or make a curiosity jump to a novel region of the brain.
    const nbr = g.neighbors[this.focus];
    let next: number;
    if (nbr.length > 0 && this.rand() < this.p.associativeProb) {
      const w = g.weights[this.focus];
      const scores = nbr.map((j, idx) => Math.max(0, w[idx]) * novelty(this.visits[j]));
      next = pickWeighted(nbr, scores, this.rand);
    } else {
      next = this._curiosityJump();
    }

    this.focus = next;
    this.visits[next] += 1;
    this.thoughts += 1;
    this.act[next] = 1;
    events.push({ kind: "neuron", a: next });

    // 2) Spread — the firing cascades to associated concepts.
    const nb = g.neighbors[next];
    const nw = g.weights[next];
    for (let i = 0; i < nb.length; i++) {
      const j = nb[i];
      this.act[j] = Math.min(1, this.act[j] + Math.max(0, nw[i]) * this.p.spread);
      events.push({ kind: "synapse", a: next, b: j, w: nw[i] });
    }

    // 3) Dream — a curiosity-driven leap across a long-range bridge (insight).
    if (g.longEdges.length > 0 && this.rand() < this.p.insightProb) {
      const [a, b] = g.longEdges[Math.floor(this.rand() * g.longEdges.length)];
      this.act[a] = 1;
      this.act[b] = 1;
      this.insights += 1;
      events.push({ kind: "insight", a, b });
    }

    return events;
  }

  /** Curiosity jump: sample a handful of neurons, go to the most novel one. */
  private _curiosityJump(): number {
    let best = Math.floor(this.rand() * this.g.n);
    let bestScore = novelty(this.visits[best]);
    for (let s = 0; s < 6; s++) {
      const cand = Math.floor(this.rand() * this.g.n);
      const score = novelty(this.visits[cand]);
      if (score > bestScore) {
        best = cand;
        bestScore = score;
      }
    }
    return best;
  }

  /** Time-based activation decay; call once per frame with the frame delta. */
  decay(dt: number): void {
    const factor = Math.pow(this.p.decay, dt);
    for (let i = 0; i < this.act.length; i++) this.act[i] *= factor;
  }
}
