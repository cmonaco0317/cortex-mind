export interface BrainNeuron {
  id: string;
  label: string;
  domain: string;
  x: number;
  y: number;
  z: number;
  snippet?: string; // short source text, so each insight is traceable (evidence)
}

export interface BrainSynapse {
  s: number;
  t: number;
  w: number;
  long: boolean;
}

export interface BrainInsight {
  s: number;
  t: number;
  why: string;
  angle: string;
  /** Surprise score in [0,1]; insights are emitted sorted by this, descending. */
  score?: number;
  /** The measurements behind `why`, so a card's claim can be checked, not taken on faith. */
  evidence?: {
    sim: number; // cosine similarity in the full embedding space
    overlap: number; // Jaccard overlap of the two near-neighbour sets
    crossDomain: boolean;
  };
}

export interface BrainMap {
  meta: { name: string; count: number; dim: number; k: number; synapses: number; insights?: number };
  neurons: BrainNeuron[];
  synapses: BrainSynapse[];
  insights?: BrainInsight[];
}
