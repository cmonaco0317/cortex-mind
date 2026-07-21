// The "drop your data" flow — the zero-install product surface. Reads dropped
// files / a chosen folder / pasted text / an agent trace, embeds it locally,
// builds the brain in the browser, and reloads into it. Nothing is uploaded.
//
// When something doesn't parse, this surface tells the user *why* (per input)
// instead of silently producing an empty brain — schema-tolerant by design.

import {
  embedTexts,
  buildBrainMap,
  ingestFiles,
  ingestText,
  isIngestable,
  type IngestResult,
  type ParseNote,
  type Progress,
} from "./ingest";

const MAX_CONCEPTS = 300; // keep in-browser embedding responsive
const MIN_CONCEPTS = 6;

function $(id: string): HTMLElement {
  return document.getElementById(id)!;
}

/** Read + filter a FileList to ingestable files, noting anything unsupported. */
async function ingestFileList(list: FileList): Promise<IngestResult> {
  const all = [...list];
  const kept = all.filter((f) => isIngestable(f.webkitRelativePath || f.name));
  const read = await Promise.all(
    kept.map(async (f) => ({ name: f.webkitRelativePath || f.name, text: await f.text() })),
  );
  const result = ingestFiles(read);
  const skipped = all.length - kept.length;
  if (skipped > 0) {
    result.notes.unshift({
      file: "(files)",
      level: kept.length ? "info" : "error",
      message: `${skipped} file${skipped === 1 ? "" : "s"} skipped — unsupported type. Supported: .md .markdown .mdx .txt .text .rst .json .jsonl .ndjson`,
    });
  }
  return result;
}

function progress(msg: string): void {
  $("cx-drop-progress").textContent = msg;
}

// Render parse diagnostics — the "couldn't parse, here's why" surface. Uses
// textContent only; file names are user-controlled and must never be innerHTML.
function renderNotes(notes: ParseNote[], heading?: string): void {
  const box = $("cx-drop-msg");
  box.replaceChildren();
  if (heading) {
    const h = document.createElement("div");
    h.className = "cx-note-head";
    h.textContent = heading;
    box.appendChild(h);
  }
  for (const n of notes) {
    const line = document.createElement("div");
    line.className = `cx-note cx-note-${n.level}`;
    line.textContent = `${n.level === "error" ? "✗" : n.level === "warn" ? "!" : "·"} ${n.message}`;
    box.appendChild(line);
  }
}

async function build(result: IngestResult): Promise<void> {
  const { concepts, notes } = result;
  if (concepts.length < MIN_CONCEPTS) {
    // Not enough to form a brain — tell the user exactly why, per input.
    const problems = notes.filter((n) => n.level !== "info");
    renderNotes(
      problems.length ? problems : notes,
      `Need ~${MIN_CONCEPTS}+ text chunks to form a brain; found ${concepts.length}. Here's what happened:`,
    );
    progress("");
    return;
  }
  const capped = concepts.slice(0, MAX_CONCEPTS);
  const summary: ParseNote[] = [...notes];
  if (capped.length < concepts.length) {
    summary.push({ file: "(cap)", level: "info", message: `Using the first ${MAX_CONCEPTS} of ${concepts.length} chunks (kept responsive).` });
  }
  renderNotes(summary);
  try {
    const vecs = await embedTexts(capped.map((c) => c.text), (p: Progress) => {
      if (p.stage === "model") progress(`loading embedding model… ${Math.round((p.loaded ?? 0) * 100)}% (once)`);
      else if (p.stage === "embed") progress(`embedding locally… ${p.i}/${p.total}`);
    });
    progress("building brain…");
    const map = buildBrainMap(capped, vecs, "yours");
    sessionStorage.setItem("cortex:brain", JSON.stringify(map));
    location.href = `${location.pathname}?mine=1`;
  } catch (e) {
    renderNotes([{ file: "(embedding)", level: "error", message: `Build failed while embedding locally: ${String(e).slice(0, 180)}` }]);
    progress("");
  }
}

export function initDropUI(): void {
  const overlay = document.getElementById("cx-drop");
  if (!overlay) return;

  document.getElementById("cx-byo")?.addEventListener("click", () => overlay.classList.remove("hidden"));
  document.getElementById("cx-drop-close")?.addEventListener("click", () => overlay.classList.add("hidden"));

  const folder = document.getElementById("cx-file-folder") as HTMLInputElement | null;
  const files = document.getElementById("cx-file-files") as HTMLInputElement | null;
  document.getElementById("cx-pick-folder")?.addEventListener("click", () => folder?.click());
  document.getElementById("cx-pick-files")?.addEventListener("click", () => files?.click());
  const onPick = async (inp: HTMLInputElement | null) => {
    if (inp?.files?.length) await build(await ingestFileList(inp.files));
  };
  folder?.addEventListener("change", () => void onPick(folder));
  files?.addEventListener("change", () => void onPick(files));

  const dz = document.getElementById("cx-dropzone");
  dz?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz?.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz?.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    const dropped = e.dataTransfer?.files;
    if (dropped?.length) void (async () => build(await ingestFileList(dropped)))();
  });

  document.getElementById("cx-build")?.addEventListener("click", () => {
    const t = (document.getElementById("cx-paste") as HTMLTextAreaElement).value;
    if (t.trim()) void build(ingestText(t));
  });
}
