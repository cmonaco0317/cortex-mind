// Shareable, watermarked exports — the self-distributing GTM. Every image a user
// saves carries "made with Cortex", so each share advertises the product. All
// compositing is client-side; nothing is uploaded.

export interface ShareInsight {
  a: string;
  b: string;
  why: string;
  angle: string;
}

const CARD_W = 1200;
const CARD_H = 630;

function download(dataUrl: string, filename: string): void {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines = 99,
): number {
  const words = text.split(/\s+/);
  let line = "";
  let lines = 0;
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, y);
      line = w;
      y += lineHeight;
      if (++lines >= maxLines - 1) {
        break;
      }
    } else {
      line = test;
    }
  }
  ctx.fillText(line, x, y);
  return y + lineHeight;
}

function background(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, "#04121c");
  g.addColorStop(1, "#020509");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
  const glow = ctx.createRadialGradient(w * 0.72, h * 0.3, 0, w * 0.72, h * 0.3, w * 0.5);
  glow.addColorStop(0, "rgba(40,180,230,0.16)");
  glow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, w, h);
}

function watermark(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.textAlign = "left";
  ctx.font = "600 22px Menlo, monospace";
  ctx.fillStyle = "#4fd6f5";
  ctx.fillText("◧ CORTEX", 56, h - 40);
  ctx.font = "16px -apple-system, system-ui, sans-serif";
  ctx.fillStyle = "#456"; // muted
  ctx.textAlign = "right";
  ctx.fillText("made with Cortex — a curiosity engine", w - 56, h - 40);
  ctx.textAlign = "left";
}

export function shareInsight(ins: ShareInsight): void {
  const c = document.createElement("canvas");
  c.width = CARD_W;
  c.height = CARD_H;
  const ctx = c.getContext("2d")!;
  background(ctx, CARD_W, CARD_H);

  ctx.textBaseline = "alphabetic";
  ctx.font = "600 18px Menlo, monospace";
  ctx.fillStyle = "#4fd6f5";
  ctx.fillText("⚡ INSIGHT", 56, 92);

  ctx.font = "700 46px -apple-system, system-ui, sans-serif";
  ctx.fillStyle = "#eafaff";
  const pairY = wrapText(ctx, `${ins.a}  ✕  ${ins.b}`, 56, 156, CARD_W - 112, 56, 2);

  ctx.font = "26px -apple-system, system-ui, sans-serif";
  ctx.fillStyle = "#bde8ff";
  const whyY = wrapText(ctx, ins.why, 56, pairY + 30, CARD_W - 112, 38, 4);

  ctx.font = "italic 22px -apple-system, system-ui, sans-serif";
  ctx.fillStyle = "#7fdcf5";
  wrapText(ctx, `→ ${ins.angle}`, 56, whyY + 16, CARD_W - 112, 32, 3);

  watermark(ctx, CARD_W, CARD_H);
  download(c.toDataURL("image/png"), `cortex-insight-${slug(ins.a)}-${slug(ins.b)}.png`);
}

export function shareSnapshot(brain: HTMLCanvasElement, subtitle: string): void {
  const w = 1280;
  const h = 800;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#02030a";
  ctx.fillRect(0, 0, w, h);
  // cover-fit the brain canvas
  const scale = Math.max(w / brain.width, (h - 64) / brain.height);
  const dw = brain.width * scale;
  const dh = brain.height * scale;
  ctx.drawImage(brain, (w - dw) / 2, (h - 64 - dh) / 2, dw, dh);

  const bar = ctx.createLinearGradient(0, h - 90, 0, h);
  bar.addColorStop(0, "rgba(2,3,10,0)");
  bar.addColorStop(1, "rgba(2,3,10,0.9)");
  ctx.fillStyle = bar;
  ctx.fillRect(0, h - 90, w, 90);
  if (subtitle) {
    ctx.font = "18px -apple-system, system-ui, sans-serif";
    ctx.fillStyle = "#9fd4e8";
    ctx.textAlign = "center";
    ctx.fillText(subtitle, w / 2, h - 46);
    ctx.textAlign = "left";
  }
  watermark(ctx, w, h);
  download(c.toDataURL("image/png"), "cortex-snapshot.png");
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 24) || "x";
}
