import "./cortex/cortex.css";
import { initScene, startLoop, brainCanvas } from "./cortex/scene";
import { BrainView } from "./cortex/brain";
import { CuriosityEngine, type BrainGraph, type FireEvent } from "./cortex/engine";
import { shareInsight, shareSnapshot, type ShareInsight } from "./cortex/share";
import { initDropUI } from "./cortex/dropui";
import type { BrainMap, BrainInsight } from "./cortex/types";

const canvas = document.getElementById("cortex-canvas") as HTMLCanvasElement;
const { scene } = initScene(canvas);

function el(id: string): HTMLElement {
  return document.getElementById(id)!;
}

function buildGraph(map: BrainMap): BrainGraph {
  const n = map.neurons.length;
  const neighbors: number[][] = Array.from({ length: n }, () => []);
  const weights: number[][] = Array.from({ length: n }, () => []);
  const longEdges: Array<[number, number]> = [];
  for (const syn of map.synapses) {
    neighbors[syn.s].push(syn.t);
    weights[syn.s].push(syn.w);
    neighbors[syn.t].push(syn.s);
    weights[syn.t].push(syn.w);
    if (syn.long) longEdges.push([syn.s, syn.t]);
  }
  return { n, neighbors, weights, longEdges };
}

async function setupPicker(current: string): Promise<void> {
  const sel = document.getElementById("cx-corpus") as HTMLSelectElement | null;
  if (!sel) return;
  try {
    const res = await fetch("brains.json");
    if (!res.ok) return;
    const brains = (await res.json()) as Array<{ name: string; label?: string; count?: number }>;
    sel.replaceChildren(
      ...brains.map((b) => {
        const o = document.createElement("option");
        o.value = b.name;
        o.textContent = b.count ? `${b.label ?? b.name} · ${b.count}` : (b.label ?? b.name);
        return o;
      }),
    );
    sel.value = current;
    sel.addEventListener("change", () => {
      location.search = `?corpus=${encodeURIComponent(sel.value)}`;
    });
  } catch {
    /* no manifest — picker stays empty */
  }
}

async function main(): Promise<void> {
  initDropUI();
  const params = new URLSearchParams(location.search);
  let map: BrainMap;

  if (params.get("mine") === "1") {
    // a brain the user just built in-browser from their own data (this session)
    const raw = sessionStorage.getItem("cortex:brain");
    if (!raw) {
      el("cx-focus").textContent = "no brain in this session — drop your data via ‘＋ your data’";
      return;
    }
    map = JSON.parse(raw) as BrainMap;
    void setupPicker("");
  } else {
    const corpus = params.get("corpus") ?? "safe";
    void setupPicker(corpus);
    try {
      const res = await fetch(`brain-${corpus}.json`);
      if (!res.ok) throw new Error(String(res.status));
      map = (await res.json()) as BrainMap;
    } catch {
      el("cx-focus").textContent = `no brain map (brain-${corpus}.json) — run cortex/build_brain.py`;
      return;
    }
  }

  const brain = new BrainView(scene, map);
  const graph = buildGraph(map);
  const engine = new CuriosityEngine(graph);

  el("cx-neurons").textContent = `Neurons: ${map.neurons.length}`;
  el("cx-synapses").textContent = `Synapses: ${map.synapses.length}`;

  // Insight Digest — the exportable, evidence-backed artifact (the product, per
  // the council: "the OUTPUT is the product, the brain is context"). The engine
  // surfaces leaps live as it wanders, but the full digest is available on demand
  // (⚡ all) and ranked most-non-obvious-first, so the first minute yields a
  // useful artifact instead of making the user wait for the wander to reach them.
  const allInsights = map.insights ?? [];
  const insightLookup = new Map<string, BrainInsight>();
  for (const ins of allInsights) {
    insightLookup.set(`${ins.s}-${ins.t}`, ins);
    insightLookup.set(`${ins.t}-${ins.s}`, ins);
  }
  const keyOf = (ins: BrainInsight): string => (ins.s < ins.t ? `${ins.s}-${ins.t}` : `${ins.t}-${ins.s}`);
  // Non-obviousness = low cosine similarity between the bridged pair; the long
  // synapse carries that weight, so lower w -> more surprising -> ranked first.
  const pairWeight = new Map<string, number>();
  for (const syn of map.synapses) if (syn.long) pairWeight.set(`${syn.s}-${syn.t}`, syn.w);
  const surprise = (ins: BrainInsight): number => pairWeight.get(keyOf(ins)) ?? 1;
  const ranked = [...allInsights].sort((x, y) => surprise(x) - surprise(y));
  el("cx-insight-total").textContent = String(allInsights.length);

  const discovered = new Set<string>();
  let cardTimer = 0;
  let currentShare: ShareInsight | null = null;

  function popCard(ins: BrainInsight): void {
    const labelA = map.neurons[ins.s].label;
    const labelB = map.neurons[ins.t].label;
    currentShare = { a: labelA, b: labelB, why: ins.why, angle: ins.angle };
    el("cx-insight-pair").textContent = `${labelA}  ✕  ${labelB}`;
    el("cx-insight-why").textContent = ins.why;
    el("cx-insight-angle").textContent = `→ ${ins.angle}`;
    el("cx-insight-card").classList.remove("hidden");
    cardTimer = 6;
  }

  // Rebuild the digest feed in rank order (most non-obvious first), numbered.
  function renderLog(): void {
    const shown = ranked.filter((ins) => discovered.has(keyOf(ins)));
    const log = el("cx-insights-log");
    log.replaceChildren(
      ...shown.map((ins, i) => {
        const entry = document.createElement("div");
        entry.className = "cx-log-entry";
        const pair = document.createElement("div");
        pair.className = "cx-log-pair";
        pair.textContent = `${i + 1}. ${map.neurons[ins.s].label} ✕ ${map.neurons[ins.t].label}`;
        const why = document.createElement("div");
        why.className = "cx-log-why";
        why.textContent = ins.why; // textContent — never innerHTML with corpus data
        entry.append(pair, why);
        return entry;
      }),
    );
    el("cx-insight-count").textContent = String(discovered.size);
  }

  function showInsight(a: number, b: number): void {
    const ins = insightLookup.get(`${a}-${b}`);
    if (!ins || discovered.has(keyOf(ins))) return; // each distinct leap once
    discovered.add(keyOf(ins));
    popCard(ins);
    renderLog();
  }

  // ⚡ all — surface every leap at once: the complete artifact in one click.
  function revealAll(): void {
    let added = false;
    for (const ins of ranked) if (!discovered.has(keyOf(ins))) (discovered.add(keyOf(ins)), (added = true));
    if (added) renderLog();
    for (const ins of ranked.slice(0, 6)) brain.fire([{ kind: "insight", a: ins.s, b: ins.t }]);
    if (ranked.length) popCard(ranked[0]); // spotlight the single most non-obvious leap
  }

  function buildDigest(): string {
    const shown = ranked.filter((ins) => discovered.has(keyOf(ins)));
    const lines = [
      `# Cortex — Insight Digest`,
      ``,
      `**${shown.length} non-obvious connection${shown.length === 1 ? "" : "s"}** surfaced from **${map.neurons.length} concepts** (“${map.meta.name}”), embedded locally in the browser — most surprising first.`,
      ``,
    ];
    shown.forEach((ins, i) => {
      const a = map.neurons[ins.s];
      const b = map.neurons[ins.t];
      const cross = a.domain !== b.domain ? ` · _${a.domain} ↔ ${b.domain}_` : "";
      lines.push(`## ${i + 1}. ${a.label} × ${b.label}${cross}`, ins.why, ``, `→ **${ins.angle}**`);
      if (a.snippet || b.snippet) {
        lines.push(``);
        if (a.snippet) lines.push(`> **${a.label}:** ${a.snippet}`);
        if (b.snippet) lines.push(`> **${b.label}:** ${b.snippet}`);
      }
      lines.push(``);
    });
    lines.push(`—`, `made with **Cortex** — a local curiosity engine. Nothing left your machine.`);
    return lines.join("\n");
  }

  el("cx-insights-all")?.addEventListener("click", revealAll);

  // Seed the digest with the top few leaps immediately, so the artifact exists
  // in the first seconds (the engine keeps surfacing the rest live as it wanders).
  for (const ins of ranked.slice(0, 3)) discovered.add(keyOf(ins));
  renderLog();

  (el("cx-insights-copy") as HTMLButtonElement).addEventListener("click", () => {
    void navigator.clipboard?.writeText(buildDigest());
    const btn = el("cx-insights-copy");
    btn.textContent = "copied";
    window.setTimeout(() => (btn.textContent = "copy"), 1200);
  });
  el("cx-insights-download")?.addEventListener("click", () => {
    const blob = new Blob([buildDigest()], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `cortex-insight-digest-${map.meta.name}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });

  el("cx-insight-share").addEventListener("click", () => {
    if (currentShare) shareInsight(currentShare);
  });
  el("cx-snapshot").addEventListener("click", () => {
    const focus = map.neurons[engine.focus]?.label ?? "";
    shareSnapshot(brainCanvas(), `thinking about ${focus} · ${discovered.size} insights discovered`);
  });

  const recent: string[] = [];
  let lastFocus = -1;
  const STEP_EVERY = 0.13; // seconds per thought (~7.7/s)
  let acc = 0;

  startLoop((dt) => {
    acc += dt;
    const pending: FireEvent[] = [];
    while (acc >= STEP_EVERY) {
      pending.push(...engine.step());
      acc -= STEP_EVERY;
    }
    if (pending.length) {
      brain.fire(pending);
      for (const ev of pending) {
        if (ev.kind === "insight" && ev.b !== undefined) showInsight(ev.a, ev.b);
      }
    }
    engine.decay(dt);
    brain.update(dt, engine.act);

    if (cardTimer > 0) {
      cardTimer -= dt;
      if (cardTimer <= 0) el("cx-insight-card").classList.add("hidden");
    }

    if (engine.focus !== lastFocus) {
      lastFocus = engine.focus;
      const nu = map.neurons[engine.focus];
      el("cx-focus").textContent = nu.label;
      el("cx-domain").textContent = nu.domain;
      recent.unshift(nu.label);
      if (recent.length > 6) recent.pop();
      const list = el("cx-recent");
      list.replaceChildren(
        ...recent.map((r, i) => {
          const div = document.createElement("div");
          div.textContent = r; // textContent — never innerHTML with corpus data
          div.style.opacity = String(1 - i * 0.14);
          return div;
        }),
      );
      el("cx-thoughts").textContent = `Thoughts: ${engine.thoughts}`;
      el("cx-leaps").textContent = `Leaps: ${engine.insights}`;
    }
  });
}

void main();
