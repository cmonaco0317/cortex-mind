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
}

export interface BrainMap {
  meta: { name: string; count: number; dim: number; k: number; synapses: number; insights?: number };
  neurons: BrainNeuron[];
  synapses: BrainSynapse[];
  insights?: BrainInsight[];
}
