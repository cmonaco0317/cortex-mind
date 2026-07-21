#!/usr/bin/env python3
"""
Cortex · agent-insights — render insight cards to shareable images.

Emits a self-contained HTML page that draws each card (cards.json) as a
1200x630 Ultron-blue, watermarked social card on a <canvas>, shows them in a
gallery, and lets you download any/all as PNG. 100% local; open in a browser.

Usage:
  render.py cards.json [--out cards.html]
"""

import argparse
import json
import sys

PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Cortex — your agent, decoded</title>
<style>
  :root{color-scheme:dark}
  body{margin:0;background:#05070d;color:#cfe6f2;font:15px/1.5 -apple-system,system-ui,sans-serif}
  header{padding:34px 28px 10px;max-width:1120px;margin:0 auto}
  h1{font:600 24px/1.2 -apple-system,system-ui;color:#9fe6ff;margin:0 0 6px}
  .sub{color:#5f8296;font-size:14px}
  .bar{max-width:1120px;margin:14px auto 0;padding:0 28px}
  button{background:#123543;color:#bdf3ff;border:1px solid #3aa6cf;border-radius:8px;padding:9px 16px;font:inherit;font-size:13px;cursor:pointer}
  button:hover{background:#1a4a5e}
  .grid{max-width:1120px;margin:18px auto 60px;padding:0 28px;display:grid;grid-template-columns:1fr 1fr;gap:22px}
  .card{display:flex;flex-direction:column;gap:8px}
  canvas{width:100%;height:auto;border-radius:12px;border:1px solid rgba(80,160,200,.18);box-shadow:0 8px 40px rgba(0,0,0,.5)}
  .btns{display:flex;gap:8px}
  .dl,.x{font-size:12px;padding:5px 12px}
  .x{background:#0b2836;border-color:#2a7fa0;color:#cbeeff}
  .card.full{grid-column:1 / -1}
  @media(max-width:820px){.grid{grid-template-columns:1fr}.card.full{grid-column:auto}}
</style></head><body>
<header>
  <h1>Your agent, decoded.</h1>
  <div class="sub">__COUNT__ non-obvious things about how you actually work — computed locally from your own Claude Code sessions. Pick your favorite and post it.</div>
</header>
<div class="bar"><button id="all">⤓ download all PNGs</button></div>
<div class="grid" id="grid"></div>
<script>
const CARDS = __CARDS__;
const W=1200, H=630;

function wrap(ctx,text,x,y,maxW,lh,maxLines){
  const words=text.split(/\\s+/); let line='',lines=0;
  for(const w of words){
    const t=line?line+' '+w:w;
    if(ctx.measureText(t).width>maxW && line){ ctx.fillText(line,x,y); line=w; y+=lh; if(++lines>=maxLines-1) break; }
    else line=t;
  }
  ctx.fillText(line,x,y); return y+lh;
}

function draw(cv,c){
  const ctx=cv.getContext('2d'); cv.width=W; cv.height=H;
  const g=ctx.createLinearGradient(0,0,W,H); g.addColorStop(0,'#04121c'); g.addColorStop(1,'#020509');
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
  const glow=ctx.createRadialGradient(W*0.78,H*0.24,0,W*0.78,H*0.24,W*0.6);
  glow.addColorStop(0,'rgba(40,180,230,0.20)'); glow.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=glow; ctx.fillRect(0,0,W,H);
  // faint synapse lines for texture
  ctx.strokeStyle='rgba(90,200,240,0.06)'; ctx.lineWidth=1;
  for(let i=0;i<26;i++){ ctx.beginPath(); ctx.moveTo(Math.random()*W,Math.random()*H); ctx.lineTo(Math.random()*W,Math.random()*H); ctx.stroke(); }
  ctx.textBaseline='alphabetic'; ctx.textAlign='left';
  ctx.font='600 18px Menlo,monospace'; ctx.fillStyle='#4fd6f5';
  ctx.fillText('◧ CORTEX · AGENT INSIGHTS  ·  '+c.category.toUpperCase(),56,80);
  // hero number
  ctx.font='800 116px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#eafaff';
  ctx.fillText(c.hero,54,222);
  // title
  ctx.font='700 44px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#bdf3ff';
  const ty=wrap(ctx,c.title,56,300,W-112,52,2);
  // sub
  ctx.font='25px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#8fc7de';
  wrap(ctx,c.sub,56,ty+22,W-112,36,5);
  // watermark
  ctx.font='600 21px Menlo,monospace'; ctx.fillStyle='#4fd6f5'; ctx.textAlign='left';
  ctx.fillText('◧ CORTEX',56,H-38);
  ctx.font='16px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#41627a'; ctx.textAlign='right';
  ctx.fillText('made with Cortex — read your agent\\'s mind',W-56,H-38);
  ctx.textAlign='left';
}

// The archetype hero — a NAMED type (research's core wedge), not a stat.
function drawArchetype(cv,c){
  const ctx=cv.getContext('2d'); cv.width=W; cv.height=H;
  const g=ctx.createLinearGradient(0,0,W,H); g.addColorStop(0,'#06182a'); g.addColorStop(1,'#02060d');
  ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
  const glow=ctx.createRadialGradient(W*0.28,H*0.42,0,W*0.28,H*0.42,W*0.7);
  glow.addColorStop(0,'rgba(50,150,255,0.22)'); glow.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=glow; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(90,200,240,0.07)'; ctx.lineWidth=1;
  for(let i=0;i<34;i++){ ctx.beginPath(); ctx.moveTo(Math.random()*W,Math.random()*H); ctx.lineTo(Math.random()*W,Math.random()*H); ctx.stroke(); }
  ctx.textBaseline='alphabetic'; ctx.textAlign='left';
  ctx.font='600 18px Menlo,monospace'; ctx.fillStyle='#5fd6f5';
  ctx.fillText('◧ CORTEX  ·  YOUR CLAUDE CODE ARCHETYPE',56,84);
  ctx.font='800 104px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#eafaff';
  ctx.fillText(c.name,52,196);
  ctx.font='italic 27px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#7fdcf5';
  ctx.fillText(c.tagline,56,238);
  ctx.font='26px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#bfe4f5';
  const dy=wrap(ctx,c.definition,56,300,W-112,38,3);
  // trait chips
  ctx.font='600 20px Menlo,monospace';
  let ty=dy+24;
  for(const t of (c.traits||[])){
    ctx.fillStyle='#4fd6f5'; ctx.fillText('▸',56,ty);
    ctx.fillStyle='#dff2ff'; ctx.fillText(t,84,ty); ty+=38;
  }
  ctx.font='600 21px Menlo,monospace'; ctx.fillStyle='#4fd6f5'; ctx.textAlign='left';
  ctx.fillText('◧ CORTEX',56,H-38);
  ctx.font='16px -apple-system,system-ui,sans-serif'; ctx.fillStyle='#41627a'; ctx.textAlign='right';
  ctx.fillText('made with Cortex — decode your Claude Code agent',W-56,H-38);
  ctx.textAlign='left';
}

function tweet(c){
  const t = c.kind==='archetype'
    ? `My Claude Code archetype: ${c.name} — ${c.tagline}.\\n\\n${(c.traits||[]).join('  ·  ')}`
    : `${c.title} (${c.hero})\\n\\n${c.sub}`;
  return t + `\\n\\nDecoded from my own sessions with Cortex 🧠`;
}
function postX(c){ window.open('https://twitter.com/intent/tweet?text='+encodeURIComponent(tweet(c)),'_blank'); }
function dl(cv,name){ const a=document.createElement('a'); a.href=cv.toDataURL('image/png'); a.download=name; document.body.appendChild(a); a.click(); a.remove(); }

const grid=document.getElementById('grid');
const canvases=[];
CARDS.forEach((c,i)=>{
  const wrapEl=document.createElement('div'); wrapEl.className='card'+(c.kind==='archetype'?' full':'');
  const cv=document.createElement('canvas'); (c.kind==='archetype'?drawArchetype:draw)(cv,c); canvases.push(cv);
  const name='cortex-'+(c.kind==='archetype'?'archetype':'agent-insight-'+i)+'.png';
  const btns=document.createElement('div'); btns.className='btns';
  const b=document.createElement('button'); b.className='dl'; b.textContent='⤓ download'; b.onclick=()=>dl(cv,name);
  const x=document.createElement('button'); x.className='x'; x.textContent='𝕏 post'; x.onclick=()=>{ dl(cv,name); postX(c); };
  btns.append(b,x); wrapEl.append(cv,btns); grid.appendChild(wrapEl);
});
document.getElementById('all').onclick=()=>canvases.forEach((cv,i)=>setTimeout(()=>dl(cv,'cortex-'+i+'.png'),i*250));
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cards")
    ap.add_argument("--out", default="cards.html")
    args = ap.parse_args()
    cards = json.load(open(args.cards))
    n_insights = sum(1 for c in cards if c.get("kind") != "archetype")
    html = PAGE.replace("__COUNT__", str(n_insights)).replace(
        "__CARDS__", json.dumps(cards, ensure_ascii=False)
    )
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(cards)} cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
