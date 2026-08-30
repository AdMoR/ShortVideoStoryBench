"""
The Rubric Library Atlas — one page over `dataset/rubrics.yaml`.

The library is the thing that decides what the benchmark measures, and it is a
1,300-line YAML file. This renders it as a page you can read in one pass and filter:
every criterion with its evidence class, weight and tags, grouped by section; what
each tag actually covers; and, when a built dataset is passed, what a real corpus of
seeds reached for.

Regenerate it whenever the library changes — a stale atlas is worse than none, since
its whole claim is to be the library:

    python -m video_eval_bench.report.atlas out/atlas.html
    python -m video_eval_bench.report.atlas out/atlas.html --pilot dataset_finevideo_pilot

Self-contained output: inline CSS, no framework, one small script for the tag filter.
Google Fonts is the only external asset, and the stacks fall back cleanly without it.
"""

import argparse
import html
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from video_eval_bench.dataset import load_dataset
from video_eval_bench.dataset.dataset_schemas import RubricLibrary
from video_eval_bench.dataset.seed import Seed

e = html.escape

# The page's head: title, fonts and the whole stylesheet. Inline, like report/html.py
# — the atlas has to open from file:// and survive being published as an artifact.
# Shared with `report/site.py`, which builds the GitHub Pages site around it: three
# pages in one visual system beats three that each invented their own.
HEAD = """<title>Rubric Library Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&family=JetBrains+Mono:wght@400;600&family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..600&display=swap">
<style>
:root{
  --ground:#FAF9F6; --raise:#FFFFFF; --sunk:#F1EFEA;
  --ink:#14181D; --body:#333A42; --muted:#6C757E; --faint:#98A0A8;
  --line:#DFDCD5; --line-hard:#C6C2B9;
  --accent:#1E7285; --accent-soft:#E0EEF1;
  --crit:#A8432F; --crit-soft:#F6E5E1;
  --ev-description:#3C6E80; --ev-pixels:#8A6410; --ev-motion:#476E38;
  --ev-audio:#7E4470; --ev-container:#6E5241;
  --tier1:#1E7285; --tier2:#8A6410; --tier3:#7E4470;
  --shadow:0 1px 2px rgba(20,24,29,.05), 0 8px 24px -12px rgba(20,24,29,.14);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0F1317; --raise:#161B21; --sunk:#12171C;
    --ink:#EDEEEA; --body:#C3C7C6; --muted:#8B9299; --faint:#666D74;
    --line:#252C33; --line-hard:#38424B;
    --accent:#5FB6C9; --accent-soft:#16303A;
    --crit:#E08A75; --crit-soft:#3A211B;
    --ev-description:#6FAEC2; --ev-pixels:#C9A24A; --ev-motion:#8CBA76;
    --ev-audio:#C089B0; --ev-container:#B0917A;
    --tier1:#5FB6C9; --tier2:#C9A24A; --tier3:#C089B0;
    --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 28px -14px rgba(0,0,0,.6);
  }
}
:root[data-theme="dark"]{
  --ground:#0F1317; --raise:#161B21; --sunk:#12171C;
  --ink:#EDEEEA; --body:#C3C7C6; --muted:#8B9299; --faint:#666D74;
  --line:#252C33; --line-hard:#38424B;
  --accent:#5FB6C9; --accent-soft:#16303A;
  --crit:#E08A75; --crit-soft:#3A211B;
  --ev-description:#6FAEC2; --ev-pixels:#C9A24A; --ev-motion:#8CBA76;
  --ev-audio:#C089B0; --ev-container:#B0917A;
  --tier1:#5FB6C9; --tier2:#C9A24A; --tier3:#C089B0;
  --shadow:0 1px 2px rgba(0,0,0,.3), 0 10px 28px -14px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--body);
  font-family:"Newsreader",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.62; margin:0;
  -webkit-font-smoothing:antialiased;
}
h1,h2,h3,h4,.ui{font-family:"Archivo","Helvetica Neue",Arial,sans-serif}
code,.mono{font-family:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace}
h1,h2,h3{color:var(--ink); text-wrap:balance; margin:0}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:3px}
</style>
<style>
.wrap{max-width:1180px; margin:0 auto; padding:0 clamp(1rem,4vw,2.5rem)}
.shell{display:grid; grid-template-columns:200px minmax(0,1fr); gap:clamp(1.5rem,4vw,3.5rem); align-items:start}
@media (max-width:900px){ .shell{grid-template-columns:1fr} nav.rail{display:none} }

/* ── masthead ─────────────────────────────────────────── */
header.top{border-bottom:1px solid var(--line-hard); background:var(--sunk)}
header.top .wrap{padding-top:3.5rem; padding-bottom:2.25rem}
.eyebrow{font-family:"JetBrains Mono",monospace; font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--muted)}
h1{font-size:clamp(2.4rem,6vw,3.9rem); font-weight:800; letter-spacing:-.03em;
  line-height:1.02; margin:.6rem 0 0}
.standfirst{font-size:1.12rem; color:var(--body); max-width:62ch; margin:1rem 0 0}
.tierbar{display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.6rem}
.tierchip{display:flex; align-items:baseline; gap:.5rem; padding:.45rem .8rem;
  border:1px solid var(--line-hard); border-radius:2px; background:var(--raise);
  font-family:"Archivo",sans-serif; font-size:.82rem}
.tierchip b{font-weight:600; color:var(--ink)}
.tierchip span{color:var(--muted); font-size:.78rem}
.tierchip::before{content:""; width:3px; align-self:stretch; margin:-.45rem .3rem -.45rem -.8rem}
.tierchip.t1::before{background:var(--tier1)}
.tierchip.t2::before{background:var(--tier2)}
.tierchip.t3::before{background:var(--tier3)}

/* ── rail ─────────────────────────────────────────────── */
nav.rail{position:sticky; top:1.5rem; padding-top:3rem}
nav.rail ol{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.1rem}
nav.rail a{display:block; padding:.3rem .5rem; text-decoration:none; color:var(--muted);
  font-family:"Archivo",sans-serif; font-size:.82rem; border-left:2px solid var(--line);
  transition:color .12s, border-color .12s}
nav.rail a:hover{color:var(--ink); border-left-color:var(--accent)}
nav.rail .grp{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--faint); margin:1.1rem 0 .35rem .5rem}

main{padding:3rem 0 5rem; min-width:0}
section.block{margin-bottom:4rem; scroll-margin-top:1.5rem}
h2{font-size:1.55rem; font-weight:800; letter-spacing:-.02em}
.lede{color:var(--muted); max-width:64ch; margin:.5rem 0 1.5rem; font-size:1rem}
h3{font-size:1.05rem; font-weight:600; letter-spacing:-.01em}
.rule{height:1px; background:var(--line-hard); margin:0 0 1.4rem}

/* ── criteria ─────────────────────────────────────────── */
.sect{border:1px solid var(--line); border-radius:3px; background:var(--raise);
  margin-bottom:1.1rem; overflow:hidden; box-shadow:var(--shadow)}
.sect > summary{list-style:none; cursor:pointer; padding:1rem 1.15rem;
  display:grid; grid-template-columns:minmax(0,1fr) auto; gap:.6rem 1rem; align-items:center}
.sect > summary::-webkit-details-marker{display:none}
.sect > summary:hover{background:var(--sunk)}
.sect[data-tier="1"]{border-left:3px solid var(--tier1)}
.sect[data-tier="2"]{border-left:3px solid var(--tier2)}
.sname{font-family:"Archivo",sans-serif; font-weight:600; font-size:1.02rem; color:var(--ink)}
.skey{font-family:"JetBrains Mono",monospace; font-size:.72rem; color:var(--muted)}
.scount{font-family:"JetBrains Mono",monospace; font-size:.78rem; color:var(--muted);
  font-variant-numeric:tabular-nums; white-space:nowrap}
.applies{grid-column:1/-1; font-family:"JetBrains Mono",monospace; font-size:.72rem;
  color:var(--accent); background:var(--accent-soft); padding:.35rem .55rem;
  border-radius:2px; overflow-x:auto; white-space:nowrap}
.sdesc{grid-column:1/-1; font-size:.93rem; color:var(--muted); max-width:70ch}
.crits{border-top:1px solid var(--line); padding:.4rem 0}
.crit{display:grid; grid-template-columns:88px minmax(0,1fr); gap:.2rem .9rem;
  padding:.7rem 1.15rem; border-bottom:1px solid var(--line)}
.crit:last-child{border-bottom:none}
.cid{font-family:"JetBrains Mono",monospace; font-size:.76rem; color:var(--ink); font-weight:600;
  word-break:break-all}
.chead{display:flex; flex-wrap:wrap; align-items:baseline; gap:.5rem}
.cname{font-family:"Archivo",sans-serif; font-weight:600; font-size:.93rem; color:var(--ink)}
.cdesc{grid-column:2; font-size:.92rem; color:var(--body); max-width:72ch; margin-top:.15rem}
.meta{grid-column:1; font-family:"JetBrains Mono",monospace; font-size:.66rem; color:var(--faint);
  font-variant-numeric:tabular-nums}

.ev{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.06em;
  text-transform:uppercase; padding:.1rem .38rem; border-radius:2px; border:1px solid currentColor}
.ev.description{color:var(--ev-description)} .ev.pixels{color:var(--ev-pixels)}
.ev.motion{color:var(--ev-motion)} .ev.audio{color:var(--ev-audio)}
.ev.container{color:var(--ev-container)}
.crt{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--crit); background:var(--crit-soft);
  padding:.1rem .38rem; border-radius:2px}
.new{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--accent); background:var(--accent-soft);
  padding:.1rem .38rem; border-radius:2px}
</style>
<style>
/* ── resolver ─────────────────────────────────────────── */
.resolver{border:1px solid var(--line-hard); border-radius:3px; background:var(--raise);
  box-shadow:var(--shadow); overflow:hidden}
.rgrid{display:grid; grid-template-columns:minmax(0,340px) minmax(0,1fr)}
@media (max-width:760px){ .rgrid{grid-template-columns:1fr} }
.rpick{padding:1.2rem; border-right:1px solid var(--line); background:var(--sunk)}
@media (max-width:760px){ .rpick{border-right:none; border-bottom:1px solid var(--line)} }
.rout{padding:1.2rem; min-width:0}
.flabel{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--muted); margin:.9rem 0 .35rem}
.flabel:first-child{margin-top:0}
.opts{display:flex; flex-wrap:wrap; gap:.25rem}
.opt{font-family:"Archivo",sans-serif; font-size:.76rem; padding:.24rem .55rem;
  border:1px solid var(--line-hard); background:var(--raise); color:var(--muted);
  border-radius:2px; cursor:pointer; transition:all .12s}
.opt:hover{border-color:var(--accent); color:var(--ink)}
.opt[aria-pressed="true"]{background:var(--accent); border-color:var(--accent); color:var(--ground)}
:root[data-theme="dark"] .opt[aria-pressed="true"],
:root:not([data-theme="light"]) .opt[aria-pressed="true"]{color:#0F1317}
.rstat{font-family:"Archivo",sans-serif; font-size:.85rem; color:var(--muted);
  padding-bottom:.7rem; margin-bottom:.7rem; border-bottom:1px solid var(--line)}
.rstat b{color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums}
.rgroup{margin-bottom:.9rem}
.rghead{font-family:"JetBrains Mono",monospace; font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); margin-bottom:.35rem;
  display:flex; align-items:center; gap:.45rem}
.rghead::before{content:""; width:8px; height:8px; border-radius:1px; background:var(--tier2)}
.rgroup.general .rghead::before{background:var(--tier1)}
.pills{display:flex; flex-wrap:wrap; gap:.25rem}
.pill{font-family:"JetBrains Mono",monospace; font-size:.7rem; padding:.16rem .42rem;
  border:1px solid var(--line-hard); border-radius:2px; color:var(--body); background:var(--raise)}
.empty{color:var(--faint); font-style:italic; font-size:.9rem}
.reset{font-family:"Archivo",sans-serif; font-size:.74rem; margin-top:1.1rem;
  background:none; border:1px solid var(--line-hard); color:var(--muted);
  padding:.28rem .6rem; border-radius:2px; cursor:pointer}
.reset:hover{color:var(--ink); border-color:var(--accent)}

/* ── tables ───────────────────────────────────────────── */
.scroll{overflow-x:auto; border:1px solid var(--line); border-radius:3px; background:var(--raise)}
table{border-collapse:collapse; width:100%; font-size:.86rem}
th,td{text-align:left; padding:.5rem .75rem; border-bottom:1px solid var(--line); vertical-align:top}
th{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--muted); font-weight:400; background:var(--sunk);
  position:sticky; top:0}
td.n{font-family:"JetBrains Mono",monospace; font-variant-numeric:tabular-nums; text-align:right;
  white-space:nowrap}
tr.grp td{background:var(--sunk); font-family:"JetBrains Mono",monospace; font-size:.72rem;
  color:var(--ink); letter-spacing:.06em}
td.val{font-family:"JetBrains Mono",monospace; font-size:.78rem; padding-left:1.7rem}
.zero{color:var(--crit)}
.dim{color:var(--faint)}
tbody tr:last-child td{border-bottom:none}

/* ── seeds ────────────────────────────────────────────── */
.seed{border:1px solid var(--line); border-radius:3px; background:var(--raise);
  margin-bottom:.6rem; overflow:hidden}
.seed > summary{list-style:none; cursor:pointer; padding:.8rem 1rem;
  display:flex; flex-wrap:wrap; align-items:baseline; gap:.6rem}
.seed > summary::-webkit-details-marker{display:none}
.seed > summary:hover{background:var(--sunk)}
.sid{font-family:"JetBrains Mono",monospace; font-size:.8rem; font-weight:600; color:var(--ink)}
.scat{font-family:"Archivo",sans-serif; font-size:.78rem; color:var(--muted)}
.bar{margin-left:auto; display:flex; height:9px; width:150px; border-radius:1px; overflow:hidden;
  border:1px solid var(--line-hard); flex:none}
.bar i{display:block; height:100%}
.bar i.a{background:var(--tier1)} .bar i.s{background:var(--tier2)}
.sbody{padding:0 1rem 1rem; border-top:1px solid var(--line)}
.sbrief{font-size:.92rem; color:var(--body); margin:.8rem 0; max-width:70ch}
.tagrow{display:flex; flex-wrap:wrap; gap:.25rem; margin:.6rem 0}
.tg{font-family:"JetBrains Mono",monospace; font-size:.68rem; padding:.14rem .4rem;
  background:var(--sunk); border:1px solid var(--line); border-radius:2px; color:var(--muted)}
.tg b{color:var(--ink); font-weight:600}

footer{border-top:1px solid var(--line-hard); background:var(--sunk); padding:2rem 0 3rem;
  font-size:.85rem; color:var(--muted)}
footer code{font-size:.8rem; color:var(--body)}
@media (prefers-reduced-motion:reduce){ *{transition:none !important; animation:none !important} }
</style>
<style>
.flabel i{font-style:normal; text-transform:none; letter-spacing:0; color:var(--faint)}
.jumps{padding-bottom:1rem; margin-bottom:.4rem; border-bottom:1px solid var(--line)}
.jumprow{display:flex; flex-wrap:wrap; gap:.25rem}
.jump{font-family:"Archivo",sans-serif; font-size:.76rem; text-decoration:none;
  padding:.24rem .55rem; border:1px solid var(--line); border-radius:2px;
  color:var(--body); background:var(--sunk); display:inline-flex; gap:.35rem;
  align-items:baseline; transition:border-color .12s, color .12s}
.jump:hover{border-color:var(--accent); color:var(--ink)}
.jump b{font-family:"JetBrains Mono",monospace; font-size:.68rem; font-weight:600;
  color:var(--muted); font-variant-numeric:tabular-nums}
.jump[hidden]{display:none}
.pill i{font-style:normal; color:var(--faint)}
.pill.dim{color:var(--faint); text-decoration:line-through}
.tagline{grid-column:2; display:flex; flex-wrap:wrap; gap:.22rem; margin-top:.45rem}
.tg2{font-family:"JetBrains Mono",monospace; font-size:.63rem; letter-spacing:.04em;
  padding:.08rem .34rem; border-radius:2px; background:var(--sunk); color:var(--muted);
  border:1px solid var(--line)}
.crit[hidden], .sect[hidden]{display:none}
.scount2{margin-left:auto; font-family:"JetBrains Mono",monospace; font-size:.74rem;
  color:var(--muted); font-variant-numeric:tabular-nums}
.rpick{background:var(--raise); border:none; padding:1.2rem}
.notes{margin-top:1.1rem; font-size:.92rem; color:var(--body)}
.notes p{margin:.5rem 0; max-width:70ch}
.rstat{border-bottom:none; border-top:1px solid var(--line); margin:1rem 0 0;
  padding:.7rem 0 0; min-height:1.2rem}
td.mid{color:var(--ev-pixels)}
</style>
"""

# Criteria that were merged away or deleted in the review that produced
# `dataset/RUBRICS.md`. A dataset built before it — the FineVideo pilot — names ids
# the library no longer has, and the coverage numbers would silently lose those
# selections. Mapping them to their successor keeps the counts honest; the page says
# it is doing this rather than hiding it. `None` is a criterion with no successor.
SUPERSEDED: Dict[str, Optional[str]] = {
    "PROP1": "SEC1",     # named props folded into secondary elements
    "CTX1": "SEC1",      # background context, same
    "CLEAN1": "FOCUS1",  # distractors folded into subject emphasis
    "STEP1": "SEQ1",     # step order was scene order written twice
    "GEO1": None,        # removed: no judge could decide what "right" was
}

# What each tag group is for, in the page's own words. Keyed by the group names in
# `criterion_tags`; an unknown group still renders, just without a gloss.
GROUP_NOTE = {
    "subject": "the broad theme a check is about",
    "sub_theme": "a niche inside one of the subjects above — never carried alone",
    "span": "how much of the clip it has to see at once",
    "failure": "the kind of defect it catches",
}


def _tag_groups(lib: RubricLibrary) -> List[tuple]:
    return [(g, vs) for g, vs in lib.criterion_tags.items() if vs]


def _filter_bar(lib: RubricLibrary) -> str:
    out = []
    for group, values in _tag_groups(lib):
        note = GROUP_NOTE.get(group, "")
        out.append(
            f'<div class="flabel">{e(group)}'
            + (f' <i>{e(note)}</i>' if note else "")
            + "</div><div class=\"opts\">"
            + "".join(
                f'<button class="opt" type="button" data-tag="{e(v)}" '
                f'aria-pressed="false">{e(v)}</button>'
                for v in values
            )
            + "</div>"
        )
    return (
        '<div class="resolver"><div class="rpick">'
        + _section_chips(lib)
        + "".join(out)
        + '<button class="reset" type="button" id="clear">Clear all</button>'
        + '<div class="rstat" id="stat"></div></div></div>'
    )


def _section_chips(lib: RubricLibrary) -> str:
    """
    Shortcuts to every section, in the filter header.

    The rail carries the same links but disappears under 900px, and on a page this
    long the sections below the fold are the thing you are looking for. They also
    double as a readout of the filter: pick a tag and a chip's count drops to what
    that section still holds, or the chip goes away.
    """
    chips = "".join(
        f'<a class="jump" href="#s-{e(s.key)}" data-jump="{e(s.key)}">'
        f'{e(s.name)} <b>{len(s.criteria)}</b></a>'
        for s in lib.sections
    )
    return (
        '<div class="jumps"><div class="flabel">sections '
        "<i>in file order, generic first</i></div>"
        f'<div class="jumprow">{chips}</div></div>'
    )


def _criterion(c) -> str:
    flags = [f'<span class="ev {c.evidence}">{c.evidence}</span>']
    if c.critical:
        flags.append('<span class="crt">critical</span>')
    if c.requires_references:
        flags.append('<span class="new">refs only</span>')
    binds = (
        f'<span class="tg2">binds {e(", ".join(c.binds))}</span>' if c.binds else ""
    )
    tags = "".join(f'<span class="tg2">{e(t)}</span>' for t in c.tags)
    return (
        f'<div class="crit" id="c-{e(c.id)}" data-tags="{e(" ".join(c.tags))}">'
        f'<div class="cid">{e(c.id)}</div>'
        f'<div class="chead"><span class="cname">{e(c.name)}</span>{"".join(flags)}</div>'
        f'<div class="meta">w{c.weight:g} · {e(c.dimension)}</div>'
        f'<p class="cdesc">{e(_flatten(c.description))}</p>'
        f'<div class="tagline">{binds}{tags}</div></div>'
    )


def _flatten(text: str) -> str:
    """Folded YAML keeps its paragraph breaks; the card wants one flow of prose."""
    return " ".join(text.split())


def _sections(lib: RubricLibrary) -> str:
    out = []
    for i, s in enumerate(lib.sections):
        tier = "1" if s.key == "general" else "2"
        out.append(
            f'<details class="sect" id="s-{e(s.key)}" data-tier="{tier}"'
            f'{" open" if i == 0 else ""}><summary>'
            f'<span class="sname">{e(s.name)}</span> '
            f'<span class="skey">{e(s.key)}</span>'
            f'<span class="scount"><b class="shown">{len(s.criteria)}</b> criteria</span>'
            f'<p class="sdesc">{e(_flatten(s.description))}</p></summary>'
            f'<div class="crits">{"".join(_criterion(c) for c in s.criteria)}</div></details>'
        )
    return "".join(out)


def _coverage(lib: RubricLibrary, seeds: List[Seed]) -> tuple:
    """Supply (criteria carrying a tag) against demand (times a seed was judged on one)."""
    used = Counter()
    seeds_with = Counter()
    for seed in seeds:
        seen = set()
        for cid in seed.criterion_ids():
            cid = SUPERSEDED.get(cid, cid)
            if cid is None or cid not in lib:
                continue
            for tag in lib.get(cid).tags:
                used[tag] += 1
                seen.add(tag)
        for tag in seen:
            seeds_with[tag] += 1

    rows = []
    for group, values in _tag_groups(lib):
        rows.append(f'<tr class="grp"><td colspan="4">{e(group)}</td></tr>')
        for v in values:
            supply = len(lib.with_tag(v))
            rows.append(
                f'<tr><td class="val">{e(v)}</td>'
                f'<td class="n{"" if supply else " zero"}">{supply}</td>'
                f'<td class="n{"" if used[v] else " dim"}">{used[v] or "—"}</td>'
                f'<td class="n dim">{seeds_with[v] or "—"}</td></tr>'
            )
    sect_rows = "".join(
        f'<tr><td class="val">{e(s.key)}</td><td>{e(s.name)}</td>'
        f'<td class="n">{len(s.criteria)}</td></tr>'
        for s in lib.sections
    )
    return (
        '<div class="scroll"><table>'
        "<thead><tr><th>Tag</th><th>Supply</th><th>Used</th><th>Seeds</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        '<div class="scroll"><table>'
        "<thead><tr><th>Section</th><th>Name</th><th>Criteria</th></tr></thead>"
        f"<tbody>{sect_rows}</tbody></table></div>",
    )


def _seeds(lib: RubricLibrary, seeds: List[Seed]) -> str:
    out = []
    for s in seeds:
        tags = "".join(
            f'<span class="tg"><b>{e(k)}</b> {e(", ".join(v))}</span>'
            for k, v in sorted(s.tags.items())
        )
        by_section = Counter()
        pills = []
        for cid in s.criterion_ids():
            mapped = SUPERSEDED.get(cid, cid)
            if mapped is None:
                pills.append(f'<span class="pill dim">{e(cid)} ✗</span>')
                continue
            if mapped not in lib:
                pills.append(f'<span class="pill dim">{e(cid)}</span>')
                continue
            by_section[lib.section_of(mapped)] += 1
            label = e(mapped) if mapped == cid else f"{e(mapped)} <i>← {e(cid)}</i>"
            pills.append(f'<span class="pill">{label}</span>')
        drawn = "".join(
            f'<span class="pill">{e(k)} <b>{n}</b></span>'
            for k, n in by_section.most_common()
        )
        out.append(
            f'<details class="seed"><summary><span class="sid">{e(s.seed_id)}</span>'
            f'<span class="scat">{e(s.category)}</span>'
            f'<span class="scount2">{len(s.rubrics)} criteria</span></summary>'
            f'<div class="sbody"><p class="sbrief">{e(_flatten(s.prompt)[:420])}</p>'
            f'<div class="tagrow">{tags}</div>'
            f'<div class="rgroup general"><div class="rghead">Drawn from</div>'
            f'<div class="pills">{drawn}</div></div>'
            f'<div class="rgroup"><div class="rghead">Criteria</div>'
            f'<div class="pills">{"".join(pills)}</div></div></div></details>'
        )
    return "".join(out)


def render(
    lib: RubricLibrary,
    pilot: Optional[List[Seed]] = None,
    nav: str = "",
    head: Optional[str] = None,
    corpus: str = "the FineVideo pilot",
) -> str:
    """
    The whole page. `nav` is prepended above the masthead — the site build passes
    its tab bar there, and the standalone artifact passes nothing. `corpus` names
    whichever set of seeds the coverage and seed sections are counting, since that
    is the pilot locally and the benchmark's own seeds in CI, where the pilot is not
    checked in.
    """
    pilot = pilot or []
    n_tags = sum(len(v) for _, v in _tag_groups(lib))
    n_groups = len(_tag_groups(lib))
    cov_table, sect_table = _coverage(lib, pilot)
    gated = [c for c in lib.criteria if "Applies only" in c.description]
    bound = [c for c in lib.criteria if c.binds]

    rail = (
        '<nav class="rail"><div class="grp">Browse</div><ol>'
        '<li><a href="#filter">Filter by tag</a></li></ol>'
        '<div class="grp">Sections</div><ol>'
        + "".join(
            f'<li><a href="#s-{e(s.key)}">{e(s.name)}</a></li>' for s in lib.sections
        )
        + '</ol><div class="grp">Analysis</div><ol>'
        '<li><a href="#coverage">Coverage</a></li>'
        + ('<li><a href="#seeds">Built seeds</a></li>' if pilot else "")
        + '<li><a href="#reading">How to read it</a></li></ol></nav>'
    )

    seeds_block = (
        f'''
<section class="block" id="seeds">
<h2>Built seeds</h2>
<p class="lede">The {len(pilot)} seeds of {corpus}, and what each one ended up
carrying. Every criterion here was chosen for that brief specifically — nothing
arrived as part of a set. Where a seed predates the rubric review, ids that have since
merged are shown mapped to their successor (<code>SEC1 ← PROP1</code>) and the one
that was deleted outright is struck (<code>GEO1 ✗</code>).</p>
<div class="rule"></div>
{_seeds(lib, pilot)}
</section>
'''
        if pilot
        else ""
    )

    body = f'''
{nav}
<header class="top"><div class="wrap">
<div class="eyebrow">video-eval-bench · dataset/rubrics.yaml</div>
<h1>Rubric Library Atlas</h1>
<p class="standfirst">Every criterion the benchmark knows how to check —
{len(lib.criteria)} of them, filed into {len(lib.sections)} categories and tagged from
a closed vocabulary of {n_tags} across {n_groups} groups. Both axes are for reading and
reporting. <strong>Neither one puts a criterion on a seed.</strong></p>
<div class="tierbar">
<div class="tierchip t1"><b>Sections</b><span>{len(lib.sections)} categories, generic first</span></div>
<div class="tierchip t2"><b>Tags</b><span>{n_tags} across {" / ".join(g for g, _ in _tag_groups(lib))}</span></div>
<div class="tierchip t3"><b>Gated</b><span>{len(gated)} say when they apply · {len(bound)} bind a value</span></div>
</div>
</div></header>

<div class="wrap"><div class="shell">
{rail}
<main>

<section class="block" id="filter">
<h2>Filter by tag</h2>
<p class="lede">Tags are the axis that cuts across categories — what a check looks at,
how much of the video it needs at once, and what kind of failure it catches. Pick any
and the library below narrows to criteria carrying all of them.</p>
{_filter_bar(lib)}
</section>

<section class="block" id="library">
<h2>The library</h2>
<p class="lede">Sections in file order, generic first. A section says what
<em>kind</em> of check lives there so the file can be reviewed a category at a time —
it is not a bundle, and taking one wholesale is exactly the failure this library was
flattened to escape. The later sections hold the criteria that only mean something for
a kind of brief, and each one opens by saying which.</p>
<div class="rule"></div>
{_sections(lib)}
</section>

<section class="block" id="coverage">
<h2>Coverage</h2>
<p class="lede"><b>Supply</b> is how many criteria in the library carry the tag.
<b>Used</b> is how many times a seed in {corpus} actually carries one, and
<b>Seeds</b> how many distinct seeds that is. A tag with supply and no use is a
question the library can ask and this corpus never did.</p>
{cov_table}
<h3 style="margin-top:1.8rem">By section</h3>
<div class="rule" style="margin-top:.6rem"></div>
{sect_table}
</section>
{seeds_block}
<section class="block" id="reading">
<h2>How to read it</h2>
<div class="rule"></div>
<h3>Categories and tags decide nothing</h3>
<p>This is the property the whole design turns on. A seed is judged on the criteria it
names, chosen one at a time from this library by the seed builder, grounded in that
seed's own brief and checked by two judges. No category is applied on top.</p>
<p>The library has had to learn this twice. A genre once selected a rubric outright,
which is how a social-media brief came to be asked about geographical accuracy — and
to score full marks for it, because a criterion that could not apply was answered
"PASSED, not applicable" and still counted its full weight. Attaching a whole category
at once has the same shape and the same failure: it grades a video on questions its
brief never asked.</p>
<h3>Broad subjects, and the niches inside them</h3>
<p>The <code>subject</code> tags are the broad themes a check can be about. The
<code>sub_theme</code> ones — <code>brand</code>, <code>diagram</code> — are niches
inside those, and are kept in their own group for a reporting reason: listed beside
<code>people</code> and <code>environment</code> they read as peers, and a library with
five marketing criteria looks like it is a fifth about branding. Every criterion
carrying a sub-theme also carries the broad subject it sits inside, so filtering by
<code>text</code> still finds the diagram checks.</p>
<h3>Gates are part of the criterion</h3>
<p>{len(gated)} of the {len(lib.criteria)} criteria open by naming the kind of brief
they apply to, and {len(bound)} bind a value the seed must supply — the claim, the
look, the period, the payoff. Both exist for the same reason: a criterion that cannot
say when it applies gets applied anyway, answered "not applicable, passed", and banks
its full weight. The standard is written up in <code>dataset/RUBRICS.md</code>.</p>
<h3>Evidence is what a validator must be able to see</h3>
<p>The five classes route verification. A text judge reading a description of a video
can settle <span class="ev description">description</span>; it cannot settle
<span class="ev pixels">pixels</span> or <span class="ev motion">motion</span>, so
those record <code>unchecked</code> — nothing looked, as opposed to something found at
fault. <span class="ev container">container</span> goes to ffprobe.</p>
<h3>The audio criteria are written ahead of the judge</h3>
<p>Every criterion in the <code>audio</code> section is
<span class="ev audio">audio</span>, and no judge backend here is fed sound yet — the
judge hands the model still frames. They will record <code>unchecked</code> until that
changes. They are written now because brief synthesis already extracts the spoken
lines, so the seeds carry the grounding; and because <code>MUSCONT1</code> names the
failure a shot-by-shot generation pipeline commits by construction — music that
restarts at every cut.</p>
</section>
</main></div></div>

<footer><div class="wrap">
Generated from <code>dataset/rubrics.yaml</code> ({len(lib.criteria)} criteria ·
{len(lib.sections)} sections · {n_tags} tags){f" and {corpus}" if pilot else ""}
by <code>video_eval_bench/report/atlas.py</code>.
</div></footer>

<script>
(function () {{
  const chosen = new Set();
  const stat = document.getElementById("stat");
  const crits = Array.from(document.querySelectorAll(".crit"));
  const sections = Array.from(document.querySelectorAll(".sect"));

  function apply() {{
    let shown = 0;
    crits.forEach(function (el) {{
      const tags = (el.dataset.tags || "").split(" ");
      let ok = true;
      chosen.forEach(function (t) {{ if (tags.indexOf(t) < 0) ok = false; }});
      el.hidden = !ok;
      if (ok) shown++;
    }});
    sections.forEach(function (sec) {{
      const visible = Array.from(sec.querySelectorAll(".crit")).filter(function (c) {{
        return !c.hidden;
      }}).length;
      sec.hidden = visible === 0;
      const badge = sec.querySelector(".shown");
      if (badge) badge.textContent = visible;
      const key = sec.id.replace(/^s-/, "");
      const chip = document.querySelector('.jump[data-jump="' + key + '"]');
      if (chip) {{
        chip.hidden = visible === 0;
        chip.querySelector("b").textContent = visible;
      }}
    }});
    if (!chosen.size) {{
      stat.textContent = "";
      return;
    }}
    stat.innerHTML = "<b>" + shown + "</b> criteria carry " +
      Array.from(chosen).map(function (t) {{ return "<code>" + t + "</code>"; }}).join(" + ");
  }}

  document.querySelectorAll(".opt[data-tag]").forEach(function (b) {{
    b.addEventListener("click", function () {{
      const tag = b.dataset.tag;
      if (chosen.has(tag)) {{ chosen.delete(tag); }} else {{ chosen.add(tag); }}
      b.setAttribute("aria-pressed", chosen.has(tag) ? "true" : "false");
      apply();
    }});
  }});
  document.getElementById("clear").addEventListener("click", function () {{
    chosen.clear();
    document.querySelectorAll(".opt[data-tag]").forEach(function (b) {{
      b.setAttribute("aria-pressed", "false");
    }});
    apply();
  }});
  document.querySelectorAll(".jump[data-jump]").forEach(function (a) {{
    a.addEventListener("click", function () {{
      const sec = document.getElementById("s-" + a.dataset.jump);
      if (sec) sec.open = true;
    }});
  }});
  apply();
}})();
</script>
'''
    return (head or HEAD) + body


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Render the rubric library as one page.")
    ap.add_argument("out", type=Path, nargs="?", default=Path("atlas.html"))
    ap.add_argument("--dataset", type=Path, default=None, help="dataset directory")
    ap.add_argument(
        "--pilot",
        type=Path,
        default=None,
        help="a built dataset to draw the coverage and seed sections from",
    )
    args = ap.parse_args(argv)

    lib = load_dataset(args.dataset).rubrics
    seeds = load_dataset(args.pilot).seeds if args.pilot else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(lib, seeds))
    print(f"{args.out} — {len(lib.criteria)} criteria, {len(lib.sections)} sections")


if __name__ == "__main__":
    main()
