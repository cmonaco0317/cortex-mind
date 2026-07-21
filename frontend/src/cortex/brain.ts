import * as THREE from "three";
import type { BrainMap } from "./types";
import type { FireEvent } from "./engine";

// Ultron aesthetic: cold cyan/blue, additive glow on near-black, organic.
const BASE = new THREE.Color(0x0b2e3d); // dim resting neuron
const HOT = new THREE.Color(0x9be8ff); // fully-fired neuron (blooms white-cyan)

// Subtle per-domain hue so the brain has organic variation without leaving cyan.
const DOMAIN_TINT: Record<string, THREE.Color> = {
  ai: new THREE.Color(0x22d3ee),
  neuro: new THREE.Color(0x2dd4bf),
  cognition: new THREE.Color(0x38bdf8),
  philosophy: new THREE.Color(0x818cf8),
  systems: new THREE.Color(0x34d399),
  curiosity: new THREE.Color(0x67e8f9),
};

function radialSprite(): THREE.Texture {
  const s = 64;
  const cv = document.createElement("canvas");
  cv.width = cv.height = s;
  const ctx = cv.getContext("2d")!;
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.25, "rgba(255,255,255,0.9)");
  g.addColorStop(0.5, "rgba(255,255,255,0.35)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const tex = new THREE.Texture(cv);
  tex.needsUpdate = true;
  return tex;
}

interface Pulse {
  sprite: THREE.Sprite;
  from: THREE.Vector3;
  to: THREE.Vector3;
  t: number;
  speed: number;
  active: boolean;
}

interface Arc {
  line: THREE.Line;
  mat: THREE.LineBasicMaterial;
  life: number;
  active: boolean;
}

const PULSE_POOL = 260;
const ARC_POOL = 16;
const PULSE_SPEED = 2.4; // 1/seconds -> ~0.4s travel

export class BrainView {
  private readonly positions: Float32Array;
  private readonly neuronColors: THREE.BufferAttribute;
  private readonly neuronBase: THREE.Color[] = [];
  private readonly points: THREE.Points;
  private readonly sprite: THREE.Texture;
  private readonly pulses: Pulse[] = [];
  private readonly arcs: Arc[] = [];

  constructor(scene: THREE.Scene, map: BrainMap) {
    const n = map.neurons.length;
    this.sprite = radialSprite();
    this.positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const nu = map.neurons[i];
      this.positions[i * 3] = nu.x;
      this.positions[i * 3 + 1] = nu.y;
      this.positions[i * 3 + 2] = nu.z;
      const tint = DOMAIN_TINT[nu.domain] ?? DOMAIN_TINT.ai;
      this.neuronBase.push(BASE.clone().lerp(tint, 0.35));
      const b = this.neuronBase[i];
      colors[i * 3] = b.r;
      colors[i * 3 + 1] = b.g;
      colors[i * 3 + 2] = b.b;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.neuronColors = new THREE.BufferAttribute(colors, 3);
    geo.setAttribute("color", this.neuronColors);
    const mat = new THREE.PointsMaterial({
      size: 2.6,
      map: this.sprite,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.points = new THREE.Points(geo, mat);
    scene.add(this.points);

    // Synapse web — faint static additive lines.
    const seg = map.synapses.length;
    const lp = new Float32Array(seg * 6);
    const lc = new Float32Array(seg * 6);
    for (let e = 0; e < seg; e++) {
      const syn = map.synapses[e];
      const a = syn.s;
      const b = syn.t;
      lp.set([this.positions[a * 3], this.positions[a * 3 + 1], this.positions[a * 3 + 2]], e * 6);
      lp.set([this.positions[b * 3], this.positions[b * 3 + 1], this.positions[b * 3 + 2]], e * 6 + 3);
      const c = syn.long ? 0.18 : 0.09;
      for (let k = 0; k < 2; k++) {
        lc[e * 6 + k * 3] = 0.12 * c * 10;
        lc[e * 6 + k * 3 + 1] = 0.55 * c * 6;
        lc[e * 6 + k * 3 + 2] = 0.8 * c * 6;
      }
    }
    const lgeo = new THREE.BufferGeometry();
    lgeo.setAttribute("position", new THREE.BufferAttribute(lp, 3));
    lgeo.setAttribute("color", new THREE.BufferAttribute(lc, 3));
    const lmat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.5,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    scene.add(new THREE.LineSegments(lgeo, lmat));

    // Pulse pool (signals travelling along firing synapses).
    const pulseMat = new THREE.SpriteMaterial({
      map: this.sprite,
      color: 0xbdf3ff,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    for (let i = 0; i < PULSE_POOL; i++) {
      const sp = new THREE.Sprite(pulseMat);
      sp.visible = false;
      sp.scale.setScalar(1.6);
      scene.add(sp);
      this.pulses.push({ sprite: sp, from: new THREE.Vector3(), to: new THREE.Vector3(), t: 0, speed: PULSE_SPEED, active: false });
    }

    // Insight arc pool (bright cross-brain leaps that flash and fade).
    for (let i = 0; i < ARC_POOL; i++) {
      const mat2 = new THREE.LineBasicMaterial({
        color: 0xd6f6ff,
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const g2 = new THREE.BufferGeometry().setFromPoints(
        new Array(24).fill(0).map(() => new THREE.Vector3()),
      );
      const line = new THREE.Line(g2, mat2);
      line.visible = false;
      scene.add(line);
      this.arcs.push({ line, mat: mat2, life: 0, active: false });
    }
  }

  private _pos(i: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(this.positions[i * 3], this.positions[i * 3 + 1], this.positions[i * 3 + 2]);
  }

  private _spawnPulse(from: number, to: number): void {
    const p = this.pulses.find((x) => !x.active);
    if (!p) return;
    this._pos(from, p.from);
    this._pos(to, p.to);
    p.t = 0;
    p.speed = PULSE_SPEED * (0.8 + Math.random() * 0.5);
    p.active = true;
    p.sprite.visible = true;
  }

  private _spawnArc(from: number, to: number): void {
    const arc = this.arcs.find((x) => !x.active);
    if (!arc) return;
    const a = this._pos(from, new THREE.Vector3());
    const b = this._pos(to, new THREE.Vector3());
    const mid = a.clone().add(b).multiplyScalar(0.5);
    // bow the control point outward from the brain centre for a dramatic leap
    const out = mid.clone().normalize().multiplyScalar(mid.length() + a.distanceTo(b) * 0.4 + 12);
    const curve = new THREE.QuadraticBezierCurve3(a, out, b);
    (arc.line.geometry as THREE.BufferGeometry).setFromPoints(curve.getPoints(23));
    arc.mat.opacity = 1;
    arc.life = 1;
    arc.active = true;
    arc.line.visible = true;
  }

  fire(events: FireEvent[]): void {
    for (const e of events) {
      if (e.kind === "synapse" && e.b !== undefined) this._spawnPulse(e.a, e.b);
      else if (e.kind === "insight" && e.b !== undefined) this._spawnArc(e.a, e.b);
    }
  }

  update(dt: number, act: Float32Array): void {
    // Neuron brightness from activation.
    const col = this.neuronColors.array as Float32Array;
    for (let i = 0; i < this.neuronBase.length; i++) {
      const a = act[i];
      const base = this.neuronBase[i];
      col[i * 3] = base.r + (HOT.r - base.r) * a;
      col[i * 3 + 1] = base.g + (HOT.g - base.g) * a;
      col[i * 3 + 2] = base.b + (HOT.b - base.b) * a;
    }
    this.neuronColors.needsUpdate = true;

    // Travelling pulses.
    const tmp = new THREE.Vector3();
    for (const p of this.pulses) {
      if (!p.active) continue;
      p.t += p.speed * dt;
      if (p.t >= 1) {
        p.active = false;
        p.sprite.visible = false;
        continue;
      }
      tmp.copy(p.from).lerp(p.to, p.t);
      p.sprite.position.copy(tmp);
      p.sprite.scale.setScalar(1.6 * (1 - p.t) + 0.5);
    }

    // Insight arcs fade out.
    for (const arc of this.arcs) {
      if (!arc.active) continue;
      arc.life -= dt * 1.3;
      if (arc.life <= 0) {
        arc.active = false;
        arc.line.visible = false;
        continue;
      }
      arc.mat.opacity = arc.life;
    }
  }
}
