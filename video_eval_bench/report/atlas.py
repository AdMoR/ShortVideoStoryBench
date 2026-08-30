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
/* ── coverage figure ──────────────────────────────────── */
.hero{font-size:1.02rem; color:var(--body); max-width:66ch; margin:0 0 1.3rem}
.hero b{font-family:"Archivo",sans-serif; font-size:1.5rem; font-weight:800;
  color:var(--ink); font-variant-numeric:tabular-nums}
.cufig{margin:0; border:1px solid var(--line); border-radius:3px; background:var(--raise);
  box-shadow:var(--shadow); padding:1rem 1.1rem}
.culegend{display:flex; align-items:center; gap:.3rem; margin-bottom:1rem;
  font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--faint)}
.culegend span:last-child{margin-left:.3rem}
.culegend .key{width:26px; height:14px; padding:0; min-width:0}
.curow{display:grid; grid-template-columns:150px minmax(0,1fr); gap:.9rem;
  padding:.45rem 0; border-top:1px solid var(--line); align-items:baseline}
.curow:first-of-type{border-top:none}
.culab{font-family:"Archivo",sans-serif; font-size:.8rem; color:var(--ink);
  display:flex; justify-content:space-between; gap:.4rem; padding-top:.2rem}
.culab span{font-family:"JetBrains Mono",monospace; font-size:.66rem; color:var(--faint);
  font-variant-numeric:tabular-nums}
.cucells{display:flex; flex-wrap:wrap; gap:3px}
.cu{display:inline-flex; align-items:baseline; gap:.3rem; text-decoration:none;
  padding:.2rem .4rem; min-width:66px; border-radius:2px; border:1px solid var(--line-hard);
  background:var(--raise);
  background:color-mix(in oklab, var(--accent) calc(var(--r) * 52%), var(--raise));
  transition:border-color .12s}
.cu b{font-family:"JetBrains Mono",monospace; font-size:.66rem; font-weight:600;
  color:var(--ink)}
.cu i{font-family:"JetBrains Mono",monospace; font-style:normal; font-size:.62rem;
  color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums}
.cu:hover{border-color:var(--accent)}
.cu.z{background:var(--ground); border-style:dashed; border-color:var(--line)}
.cu.z b{color:var(--faint)}
figcaption{font-size:.84rem; color:var(--muted); margin-top:1rem; max-width:70ch}
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

    Every section is collapsed, so these are how the page is navigated: the chip is
    the section's name, its count, and the thing that opens it. They also double as a
    readout of the filter — pick a tag and a chip's count drops to what that section
    still holds, or the chip goes away.
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
    for s in lib.sections:
        tier = "1" if s.key == "general" else "2"
        # All closed, `general` included. Thirteen open sections is 65 criteria of
        # prose before the page has said anything; the summaries are the index.
        out.append(
            f'<details class="sect" id="s-{e(s.key)}" data-tier="{tier}"><summary>'
            f'<span class="sname">{e(s.name)}</span> '
            f'<span class="skey">{e(s.key)}</span>'
            f'<span class="scount"><b class="shown">{len(s.criteria)}</b> criteria</span>'
            f'<p class="sdesc">{e(_flatten(s.description))}</p></summary>'
            f'<div class="crits">{"".join(_criterion(c) for c in s.criteria)}</div></details>'
        )
    return "".join(out)


def _usage(lib: RubricLibrary, seeds: List[Seed]) -> Counter:
    """How many seeds of the corpus carry each criterion, by live id."""
    used = Counter()
    for seed in seeds:
        for cid in set(seed.criterion_ids()):
            cid = SUPERSEDED.get(cid, cid)
            if cid and cid in lib:
                used[cid] += 1
    return used


def _coverage_figure(lib: RubricLibrary, seeds: List[Seed], corpus: str) -> str:
    """
    One picture of what the corpus actually asks: every criterion, filed under its
    section, shaded by how many seeds carry it.

    A grid rather than a bar chart because the question here is coverage, not
    ranking — what you want to see is the *shape* of the library and where the holes
    are, and a hole is a criterion no brief has ever needed. Each cell prints its own
    number, so the shading is a second encoding rather than the only one.
    """
    used = _usage(lib, seeds)
    total = max(len(seeds), 1)
    reached = sum(1 for c in lib.criteria if used[c.id])

    rows = []
    for s in lib.sections:
        cells = []
        for c in s.criteria:
            n = used[c.id]
            ratio = n / total
            cells.append(
                f'<a class="cu{"" if n else " z"}" href="#c-{e(c.id)}" '
                f'style="--r:{ratio:.3f}" '
                f'title="{e(c.id)} — {e(c.name)}: carried by {n} of {total} seeds">'
                f"<b>{e(c.id)}</b><i>{n or '·'}</i></a>"
            )
        rows.append(
            f'<div class="curow"><div class="culab">{e(s.name)}'
            f'<span>{len(s.criteria)}</span></div>'
            f'<div class="cucells">{"".join(cells)}</div></div>'
        )

    ramp = "".join(
        f'<span class="cu key" style="--r:{r}"></span>' for r in (0, 0.25, 0.5, 0.75, 1)
    )
    return (
        f'<p class="hero"><b>{reached}</b> of the library\'s {len(lib.criteria)} '
        f"criteria are carried by at least one of the {total} seeds in {corpus}. "
        f"The rest are questions the benchmark can ask and has not yet had a brief "
        f"for.</p>"
        f'<figure class="cufig">'
        f'<div class="culegend"><span>never asked</span>{ramp}'
        f"<span>every seed</span></div>"
        f'{"".join(rows)}'
        f"<figcaption>One cell per criterion, in section order; the number is how many "
        f"seeds carry it. Click a cell to jump to its entry above.</figcaption>"
        f"</figure>"
    )


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

    body = f'''
{nav}
<header class="top"><div class="wrap">
<div class="eyebrow">video-eval-bench · dataset/rubrics.yaml</div>
<h1>Rubric Library Atlas</h1>
</div></header>

<div class="wrap"><main>

<section class="block" id="coverage">
<h2>Coverage</h2>
<div class="rule"></div>
{_coverage_figure(lib, pilot, corpus)}
</section>

<section class="block" id="filter">
<h2>Browse</h2>
<p class="lede">By category, or across them. A section says what <em>kind</em> of check
lives there; a tag is the axis that cuts through all of them — what a check looks at,
how much of the video it needs at once, what kind of failure it catches. Pick any
number of tags and the library below narrows to the criteria carrying all of them.</p>
{_filter_bar(lib)}
</section>

<section class="block" id="library">
<h2>The library</h2>
<p class="lede">Sections in file order, generic first. A section is not a bundle, and
taking one wholesale is exactly the failure this library was flattened to escape. The
later sections hold the criteria that only mean something for a kind of brief, and
each one opens by saying which.</p>
<div class="rule"></div>
{_sections(lib)}
</section>
</main></div>

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
      // Sections are collapsed by default, so a filter that only changed the counts
      // would look like it had done nothing. A live filter opens what still matches
      // and clearing it puts the page back the way it loaded.
      if (chosen.size) sec.open = visible > 0;
      else sec.open = false;
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
        help="another dataset to count coverage over (default: this one's own seeds)",
    )
    args = ap.parse_args(argv)

    ds = load_dataset(args.dataset)
    # Coverage is always counted over some real corpus: the benchmark's own seeds
    # unless another dataset is named. An empty one would render a grid of zeros and
    # read as "nothing is used", which is a different claim entirely.
    if args.pilot:
        seeds, corpus = load_dataset(args.pilot).seeds, f"{args.pilot.name}"
    else:
        seeds, corpus = ds.seeds, "the benchmark's dataset"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(ds.rubrics, seeds, corpus=corpus))
    print(
        f"{args.out} — {len(ds.rubrics.criteria)} criteria, "
        f"{len(ds.rubrics.sections)} sections, coverage over {len(seeds)} seeds"
    )


if __name__ == "__main__":
    main()
