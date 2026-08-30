"""
The public site: three pages, one visual system, built by CI.

    python -m video_eval_bench.report.site export runs/*/report.json   # snapshot
    python -m video_eval_bench.report.site build --out _site           # the site

  * `index.html` — what the benchmark is and how a score is produced.
  * `atlas.html` — the rubric library, rendered by `report/atlas.py`.
  * `performance.html` — what the arms actually scored, across every run.

**Why a snapshot.** `runs/` is gitignored: a run directory holds the generated mp4s
and is hundreds of megabytes, and CI has no GPU to regenerate one. So the performance
page is built from `site/data/runs.json`, a small committed export of the numbers in
each run's `report.json` — scores, verdicts and the arm's config, no media. Refresh it
with `make site-data` after a run you want published, and commit it. The build reads
that file and never touches `runs/`, so a Pages build is deterministic and cheap.

**Old criterion ids.** Runs predate the review that merged and deleted criteria, and
their verdicts name ids the library no longer has. They are mapped through
`atlas.SUPERSEDED` here exactly as they are there, so a criterion's difficulty is
counted against its successor rather than silently dropped, and the page says so.

The pages share the atlas's stylesheet (`atlas.HEAD`) and add a tab bar. Everything is
static: no build step beyond this module, no JS framework, no external asset but the
webfont.
"""

import argparse
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from video_eval_bench.dataset import load_dataset
from video_eval_bench.dataset.dataset_schemas import Dataset, RubricLibrary
from video_eval_bench.report import atlas

e = html.escape

# A headline score has to be comparable. A run over one seed is a smoke test, and
# putting its 100.0 on the front page would be the same lie the mock runs tell.
MIN_SEEDS_FOR_HEADLINE = 5

DEFAULT_DATA = Path("site/data/runs.json")

# One page per tab, in the order they appear in the bar.
TABS = [
    ("index.html", "Overview"),
    ("atlas.html", "Rubric atlas"),
    ("performance.html", "Performance"),
]

# Extra styles for the two pages the atlas does not already dress: the tab bar, the
# metric tiles, the score bars. Tokens only — the palette is the atlas's.
EXTRA_CSS = """
<style>
.tabs{position:sticky; top:0; z-index:9; background:var(--ground);
  border-bottom:1px solid var(--line-hard)}
.tabs .wrap{display:flex; align-items:center; gap:.2rem; padding-top:.5rem;
  padding-bottom:.5rem; flex-wrap:wrap}
.tabs .mark{font-family:"JetBrains Mono",monospace; font-size:.7rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin-right:auto}
.tab{font-family:"Archivo",sans-serif; font-size:.84rem; text-decoration:none;
  color:var(--muted); padding:.35rem .7rem; border:1px solid transparent; border-radius:2px}
.tab:hover{color:var(--ink); border-color:var(--line-hard)}
.tab[aria-current="page"]{color:var(--ink); background:var(--raise);
  border-color:var(--line-hard); font-weight:600}
.tab.ext{color:var(--faint)}

.tiles{display:grid; gap:.7rem; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  margin:1.6rem 0}
.tile{border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:3px;
  background:var(--raise); padding:.85rem 1rem; box-shadow:var(--shadow)}
.tile b{display:block; font-family:"Archivo",sans-serif; font-size:1.7rem; font-weight:800;
  color:var(--ink); line-height:1.1; font-variant-numeric:tabular-nums}
.tile span{font-family:"JetBrains Mono",monospace; font-size:.66rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--muted)}
.tile i{display:block; font-style:normal; font-size:.85rem; color:var(--muted); margin-top:.3rem}

.cards{display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  margin:1.4rem 0}
.card{display:block; text-decoration:none; border:1px solid var(--line-hard); border-radius:3px;
  background:var(--raise); padding:1.1rem 1.2rem; box-shadow:var(--shadow);
  transition:border-color .12s, transform .12s}
.card:hover{border-color:var(--accent); transform:translateY(-1px)}
.card h3{margin:0 0 .35rem}
.card p{margin:0; font-size:.92rem; color:var(--muted)}
.card .go{font-family:"JetBrains Mono",monospace; font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent); display:block; margin-top:.7rem}

ol.steps{counter-reset:step; list-style:none; padding:0; margin:1.2rem 0;
  display:grid; gap:.7rem}
/* Positioned rather than a grid. A grid container wraps each run of inline content
   in its own anonymous item, so `<b>Lead.</b> then prose` became two items and the
   prose landed in the 2rem counter column — one word per line, all the way down. */
ol.steps li{counter-increment:step; position:relative; padding-left:2.9rem;
  font-size:.97rem; max-width:74ch}
ol.steps li::before{content:counter(step,decimal-leading-zero); position:absolute;
  left:0; top:.15rem; font-family:"JetBrains Mono",monospace; font-size:.72rem;
  color:var(--accent)}
ol.steps b{color:var(--ink); font-family:"Archivo",sans-serif; font-weight:600}

.meter{display:block; height:6px; width:110px; background:var(--sunk);
  border:1px solid var(--line); border-radius:1px; overflow:hidden; margin-top:.25rem}
.meter i{display:block; height:100%; background:var(--accent)}
.meter.bad i{background:var(--crit)}
td.pass{color:var(--ev-motion)} td.fail{color:var(--crit)}
tr.best td{background:var(--accent-soft)}
.wide{max-width:none}
.foot-note{font-size:.86rem; color:var(--muted); margin-top:.8rem; max-width:70ch}

.diagram{overflow-x:auto; margin:1.4rem 0; padding-bottom:.4rem}
.diagram svg{display:block; min-width:880px; height:auto; margin:0 auto}
.dg-box{fill:var(--raise); stroke:var(--line-hard); stroke-width:1}
.dg-edge{fill:var(--accent-soft); stroke:var(--accent); stroke-width:1}
.dg-band{fill:var(--sunk); stroke:var(--line); stroke-width:1}
.dg-band-l{fill:var(--muted); font-family:"JetBrains Mono",monospace; font-size:10px;
  letter-spacing:.11em; text-transform:uppercase}
.dg-t{fill:var(--ink); font-family:"Archivo",sans-serif; font-size:14px; font-weight:600}
.dg-s{fill:var(--muted); font-family:"Newsreader",Georgia,serif; font-size:12px}
.dg-s2{fill:var(--body); font-family:"Newsreader",Georgia,serif; font-size:12px}
.dg-s3{fill:var(--faint); font-family:"Newsreader",Georgia,serif; font-size:11px}
.dg-c{fill:var(--faint); font-family:"JetBrains Mono",monospace; font-size:10px;
  letter-spacing:.06em}
.dg-c.llm{fill:var(--accent)}
.dg-o{fill:var(--muted); font-family:"JetBrains Mono",monospace; font-size:10px}
.dg-a{stroke:var(--line-hard); stroke-width:1.5; fill:none}
.dg-tip{fill:var(--line-hard)}
.dg-div{stroke:var(--line); stroke-width:1}
</style>
"""


# ── the snapshot ──────────────────────────────────────────────────────────────


# A run whose generator or judge was a mock is a self-test of the harness, not a
# benchmark result: the mock judge passes everything, so publishing one would put a
# 100.0 at the top of the table and mean nothing by it.
MOCK_CHOICES = {"mock", "fake"}


def _is_mock(data: dict) -> bool:
    choices = data.get("choices", {}) or {}
    if MOCK_CHOICES & {choices.get("generator"), choices.get("judge")}:
        return True
    variant = data.get("variant", "") or ""
    return "judge=mock" in variant or "generator=mock" in variant


def snapshot(paths: List[Path], include_mock: bool = False) -> dict:
    """
    The numbers from each run's `report.json`, small enough to commit.

    Verdict comments are kept but truncated: they are the most useful thing on the
    page — a judge saying *why* SUBJ1 failed — and the whole point of publishing a
    run is that someone can read them without cloning the repo.

    Skipped unless `include_mock`: mock runs, and runs where nothing was judged (a
    generation that died on its first seed leaves a report with no verdicts in it).
    """
    runs = []
    for path in sorted(paths):
        data = json.loads(Path(path).read_text())
        if not include_mock and _is_mock(data):
            continue
        summary = data.get("summary", {}) or {}
        seeds = []
        for result in data.get("results", []) or []:
            verdict = result.get("verdict") or {}
            seeds.append(
                {
                    "seed_id": result.get("seed_id", ""),
                    "category": result.get("category", ""),
                    "status": result.get("status", ""),
                    "score": verdict.get("total_score"),
                    "passed": verdict.get("passed"),
                    "critical_failures": verdict.get("critical_failures") or [],
                    "dimensions": [
                        {"dimension": d["dimension"], "score": d.get("score")}
                        for d in verdict.get("dimensions") or []
                    ],
                    "scores": [
                        {
                            "criterion": s.get("criterion"),
                            "passed": bool(s.get("passed")),
                            "comment": (s.get("comment") or "")[:400],
                        }
                        for s in verdict.get("scores") or []
                    ],
                }
            )
        if not include_mock and not any(s["scores"] for s in seeds):
            continue
        runs.append(
            {
                "run_id": summary.get("run_id", Path(path).parent.name),
                "variant": data.get("variant", "") or summary.get("variant", ""),
                "choices": data.get("choices", {}) or {},
                "note": data.get("note", "") or "",
                "started_at": data.get("started_at", ""),
                "summary": {
                    k: summary.get(k)
                    for k in (
                        "n_seeds", "n_ok", "n_skipped", "n_generation_errors",
                        "n_judge_errors", "mean_score", "n_passed", "n_safety_vetoes",
                        "total_duration_seconds",
                    )
                },
                "per_category": summary.get("per_category", {}) or {},
                "seeds": seeds,
            }
        )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "runs": runs}


def headline(runs: List[dict]) -> Optional[dict]:
    """The best run worth quoting: highest mean over a run that covered the dataset."""
    scored = [
        r for r in runs if isinstance(r["summary"].get("mean_score"), (int, float))
    ]
    broad = [r for r in scored if (r["summary"].get("n_ok") or 0) >= MIN_SEEDS_FOR_HEADLINE]
    return max(broad or scored, key=lambda r: r["summary"]["mean_score"], default=None)


def _live(cid: str, lib: RubricLibrary) -> Optional[str]:
    """A run's criterion id, mapped onto the library as it stands now."""
    cid = atlas.SUPERSEDED.get(cid, cid)
    return cid if cid and cid in lib else None


# ── shared chrome ─────────────────────────────────────────────────────────────


def nav(active: str, repo: str) -> str:
    tabs = "".join(
        f'<a class="tab" href="{href}"'
        + (' aria-current="page"' if href == active else "")
        + f">{e(label)}</a>"
        for href, label in TABS
    )
    return (
        '<nav class="tabs"><div class="wrap">'
        '<span class="mark">video-eval-bench</span>'
        f'{tabs}<a class="tab ext" href="{e(repo)}">GitHub ↗</a>'
        "</div></nav>"
    )


def _page(title: str, nav_html: str, body: str) -> str:
    head = atlas.HEAD.replace(
        "<title>Rubric Library Atlas</title>", f"<title>{e(title)}</title>", 1
    )
    return head + EXTRA_CSS + nav_html + body


def _footer(extra: str = "") -> str:
    return (
        '<footer><div class="wrap">'
        f"{extra}Built from the repository by "
        "<code>video_eval_bench/report/site.py</code>."
        "</div></footer>"
    )


# ── the landing page ──────────────────────────────────────────────────────────


def render_index(ds: Dataset, data: dict, nav_html: str, repo: str) -> str:
    lib = ds.rubrics
    runs = data.get("runs", [])
    best = headline(runs)
    n_critical = sum(1 for c in lib.criteria if c.critical)
    n_gated = sum(1 for c in lib.criteria if "Applies only" in c.description)

    best_tile = (
        f'<div class="tile"><span>best mean score</span>'
        f'<b>{best["summary"]["mean_score"]:.1f}</b>'
        f'<i>over {best["summary"].get("n_ok") or 0} judged seeds · '
        f'{e(best["variant"] or best["run_id"])}</i></div>'
        if best
        else ""
    )

    genres = "".join(
        f'<span class="pill">{e(name)}</span>' for name in sorted(ds.genres.values())
    )

    body = f"""
<header class="top"><div class="wrap">
<div class="eyebrow">video-eval-bench</div>
<h1>A benchmark for agents that make videos</h1>
<p class="standfirst">An agent is given a brief and the tools to generate a clip from
it. A judge then answers one binary question at a time — did the subject stay the same
person, did the four steps arrive in order, is the text real writing — and the seed's
score is the weight it earned over the weight it was asked for.
<strong>Nothing is graded on a question its brief never posed.</strong></p>
<div class="tierbar">
<div class="tierchip t1"><b>{len(ds.seeds)} seeds</b><span>hand-written briefs across {len(ds.genres)} genres</span></div>
<div class="tierchip t2"><b>{len(lib.criteria)} criteria</b><span>{n_critical} of them critical</span></div>
<div class="tierchip t3"><b>{len(data.get("runs", []))} published runs</b><span>every verdict, with the judge's reasoning</span></div>
</div>
</div></header>

<div class="wrap"><main>

<div class="tiles">
<div class="tile"><span>seeds</span><b>{len(ds.seeds)}</b><i>each naming its own rubric</i></div>
<div class="tile"><span>criteria</span><b>{len(lib.criteria)}</b><i>in {len(lib.sections)} sections, {n_gated} of them gated on the brief</i></div>
<div class="tile"><span>safety checks</span><b>{len(ds.safety_checks)}</b><i>binary vetoes, applied to every clip</i></div>
{best_tile}
</div>

<section class="block">
<h2>How a score is produced</h2>
<div class="rule"></div>
<ol class="steps">
<li><b>A brief.</b> One seed: a prompt a person could hand to a video team, plus any
reference images that come with it. {len(ds.seeds)} of them, in {len(ds.genres)} genres.</li>
<li><b>The agent generates.</b> It runs in a sandbox with a real video-generation tool
and, depending on the arm, the project's prompting skills. What it does with them is
the thing being measured.</li>
<li><b>The judge asks one question at a time.</b> One model call per criterion, each
answered pass or fail with a written reason — not one call that grades the whole clip
and averages its own impressions.</li>
<li><b>The seed scores what it earned.</b> Weight earned over weight asked for, flat
across dimensions. A criterion the brief cannot fail is not on the seed's list, so it
costs nothing and gives nothing.</li>
</ol>
</section>

<section class="block">
<h2>Where the seeds come from</h2>
<div class="rule"></div>
<p class="lede">Seeds are built from FineVideo: a real video's metadata is condensed
into a brief, and the criteria that brief puts at risk are chosen and then checked
twice. <strong>Nothing is attached in bulk.</strong></p>
<div class="diagram">{_pipeline_diagram()}</div>
</section>

<section class="block">
<h2>Where to go next</h2>
<div class="rule"></div>
<div class="cards">
<a class="card" href="atlas.html">
<h3>Rubric atlas</h3>
<p>Every criterion the benchmark knows how to check, with the kind of brief it applies
to, what a judge must see to settle it, and the tags that cut across all of it.
Filterable.</p>
<span class="go">Browse the library →</span></a>
<a class="card" href="performance.html">
<h3>Performance</h3>
<p>What the arms actually scored: every published run, the criteria that break models
most often, and the judge's own words on why a clip failed.</p>
<span class="go">Read the results →</span></a>
</div>
</section>

<section class="block">
<h2>The genres</h2>
<div class="rule"></div>
<p class="lede">A genre is a reporting label and nothing more. It groups seeds in the
summary tables; it selects no rubric. That distinction is load-bearing — a genre that
selected a rubric is how a kitchen-cleaning ad came to be graded on geographical
accuracy, and to score full marks for it.</p>
<div class="pills">{genres}</div>
</section>

</main></div>
{_footer(f'Dataset: {len(ds.seeds)} seeds · {len(lib.criteria)} criteria · {len(ds.safety_checks)} safety checks. ')}
"""
    return _page("Video Eval Bench", nav_html, body)



# ── the seed-building diagram ─────────────────────────────────────────────────


def _pipeline_diagram() -> str:
    """
    How a FineVideo source video becomes a seed, as one SVG.

    Inline rather than a diagram library: the Pages site is static and offline, and
    every colour is a CSS custom property so the drawing follows the viewer's theme
    like the rest of the page.

    The two bands are the fact worth showing. Everything above the break happens once
    per source video and is checkpointed, so a killed build resumes where it stopped;
    everything below needs the whole corpus in hand, which is why recurrence — several
    unrelated briefs reaching for the same missing check — is the only thing that can
    mint a criterion into the library.

    Three columns, fixed: what the stage costs on the left, the stage in the middle,
    what it produces on the right. The middle column is wide enough for its longest
    caption at 12px, which is the constraint that sets the whole viewBox — SVG text
    does not wrap, so a caption that outgrows its box silently draws across the
    neighbouring column.
    """
    BOX_X, BOX_W = 176, 340
    MID = BOX_X + BOX_W // 2
    COST_X, OUT_X = 158, 536

    def box(y, h, label, sub, cls="dg-box"):
        ty = y + (h / 2 if not sub else h / 2 - 6)
        out = [f'<rect class="{cls}" x="{BOX_X}" y="{y}" width="{BOX_W}" height="{h}" rx="2"/>',
               f'<text class="dg-t" x="{BOX_X + 15}" y="{ty}" dominant-baseline="middle">{label}</text>']
        if sub:
            out.append(f'<text class="dg-s" x="{BOX_X + 15}" y="{y + h / 2 + 11}"'
                       f' dominant-baseline="middle">{sub}</text>')
        return "".join(out)

    def cost(y, text, llm=False):
        return (f'<text class="dg-c{" llm" if llm else ""}" x="{COST_X}" y="{y}"'
                f' text-anchor="end" dominant-baseline="middle">{text}</text>')

    def out(y, text):
        return f'<text class="dg-o" x="{OUT_X}" y="{y}" dominant-baseline="middle">{text}</text>'

    def arrow(y1, y2):
        return f'<path class="dg-a" d="M{MID} {y1} L{MID} {y2}" marker-end="url(#dg-tip)"/>'

    svg = ['<svg viewBox="0 0 880 690" role="img" width="880"',
           ' aria-label="How a FineVideo source video becomes a benchmark seed:'
           ' digest, synthesize, select and two judges per video, then clustering,'
           ' minting and policy across the corpus.">',
           '<defs><marker id="dg-tip" viewBox="0 0 8 8" refX="7" refY="4"'
           ' markerWidth="7" markerHeight="7" orient="auto">'
           '<path class="dg-tip" d="M0 0 L8 4 L0 8 z"/></marker></defs>']

    svg.append(box(14, 40, "FineVideo", "one source video", "dg-edge"))
    svg.append(arrow(54, 74))

    svg.append('<rect class="dg-band" x="16" y="74" width="848" height="376" rx="3"/>')
    svg.append(f'<text class="dg-band-l" x="30" y="92">per source video &#183; '
               f'checkpointed after every stage</text>')

    for y, name, sub, c, llm, o in (
        (100, "digest", "metadata &#8594; bounded text", "local", False, "deterministic &#183; hashed"),
        (174, "synthesize", "the brief, and the seed&#8217;s tags", "1 call", True,
         "brief &#183; spoken lines &#183; tags"),
        (248, "select", "which criteria this brief risks", "1 call", True,
         "criteria &#183; proposals"),
    ):
        svg.append(box(y, 46, name, sub))
        svg.append(cost(y + 23, c, llm))
        svg.append(out(y + 23, o))
        svg.append(arrow(y + 46, y + 74))

    jy = 322
    svg.append(f'<rect class="dg-box" x="{BOX_X}" y="{jy}" width="{BOX_W}" height="106" rx="2"/>')
    svg.append(f'<text class="dg-t" x="{BOX_X + 15}" y="{jy + 22}" dominant-baseline="middle">the two judges</text>')
    svg.append(f'<line class="dg-div" x1="{BOX_X + 15}" y1="{jy + 36}" x2="{BOX_X + BOX_W - 15}" y2="{jy + 36}"/>')
    svg.append(f'<text class="dg-s2" x="{BOX_X + 15}" y="{jy + 54}" dominant-baseline="middle">judge_seed &#8212; does the brief ground it?</text>')
    svg.append(f'<text class="dg-s2" x="{BOX_X + 15}" y="{jy + 74}" dominant-baseline="middle">judge_metadata &#8212; does the source satisfy it?</text>')
    svg.append(f'<text class="dg-s3" x="{BOX_X + 15}" y="{jy + 93}" dominant-baseline="middle">container criteria &#8594; ffprobe, no model</text>')
    svg.append(cost(jy + 45, "1 call", True))
    svg.append(cost(jy + 61, "per criterion", True))
    svg.append(out(jy + 45, "grounded &#183; ungrounded"))
    svg.append(out(jy + 65, "verified &#183; contradicted"))
    svg.append(out(jy + 85, "unchecked &#8212; nothing looked"))
    svg.append(arrow(jy + 106, 478))

    svg.append('<rect class="dg-band" x="16" y="478" width="848" height="150" rx="3"/>')
    svg.append('<text class="dg-band-l" x="30" y="496">once, over the whole corpus</text>')

    svg.append(box(508, 46, "cluster &#8594; mint", "recurring proposals &#8594; the library"))
    svg.append(cost(523, "1 call", True))
    svg.append(cost(539, "per cluster", True))
    svg.append(out(521, "&#8805; 5 videos &#8594; new criterion"))
    svg.append(out(541, "fewer &#8594; kept on that one seed"))
    svg.append(arrow(554, 570))

    svg.append(box(570, 46, "policy", "which criteria are graded"))
    svg.append(cost(593, "local", False))
    svg.append(out(583, "reversible &#8212; no rebuild,"))
    svg.append(out(603, "no model calls"))
    svg.append(arrow(616, 636))

    svg.append(box(636, 40, "dataset", "seeds.yaml &#183; rubrics.yaml", "dg-edge"))
    svg.append("</svg>")
    return "".join(svg)


# ── the performance page ──────────────────────────────────────────────────────


def _runs_table(runs: List[dict]) -> str:
    rows = []
    best = headline(runs)
    for r in sorted(runs, key=lambda r: r.get("started_at", ""), reverse=True):
        s = r["summary"]
        score = s.get("mean_score")
        width = f"{max(0.0, min(100.0, float(score))):.0f}" if score is not None else "0"
        errors = (s.get("n_generation_errors") or 0) + (s.get("n_judge_errors") or 0)
        hours = (s.get("total_duration_seconds") or 0) / 3600
        klass = ' class="best"' if r is best else ""
        rows.append(
            f"<tr{klass}>"
            f'<td class="val">{e(r["run_id"])}</td>'
            f'<td>{e(r["variant"] or "—")}</td>'
            f'<td class="n">{s.get("n_seeds") or "—"}</td>'
            f'<td class="n">{f"{score:.1f}" if score is not None else "—"}'
            f'<span class="meter"><i style="width:{width}%"></i></span></td>'
            f'<td class="n">{s.get("n_passed") or 0}</td>'
            f'<td class="n{" fail" if errors else " dim"}">{errors or "—"}</td>'
            f'<td class="n dim">{hours:.1f}h</td></tr>'
        )
    return (
        '<div class="scroll"><table>'
        "<thead><tr><th>Run</th><th>Arm</th><th>Seeds</th><th>Mean score</th>"
        "<th>Passed</th><th>Errors</th><th>Wall clock</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _criterion_table(runs: List[dict], lib: RubricLibrary) -> str:
    """
    Which criteria models fail, over every judged seed in every run.

    This is the page's real content. A mean score says an arm is bad; this says what
    it is bad at, and each row links into the atlas entry that defines the question.
    """
    asked, passed = defaultdict(int), defaultdict(int)
    for run in runs:
        for seed in run["seeds"]:
            for score in seed["scores"]:
                cid = _live(score["criterion"], lib)
                if not cid:
                    continue
                asked[cid] += 1
                passed[cid] += int(score["passed"])
    rows = []
    for cid, n in sorted(asked.items(), key=lambda kv: (passed[kv[0]] / kv[1], -kv[1])):
        c = lib.get(cid)
        rate = 100 * passed[cid] / n
        crit = ' <span class="crt">critical</span>' if c.critical else ""
        rows.append(
            f'<tr><td class="val"><a href="atlas.html#c-{e(cid)}">{e(cid)}</a></td>'
            f"<td>{e(c.name)}{crit}</td>"
            f'<td class="dim">{e(lib.section_of(cid))}</td>'
            f'<td class="n">{n}</td>'
            f'<td class="n {"fail" if rate < 50 else "pass"}">{rate:.0f}%'
            f'<span class="meter{" bad" if rate < 50 else ""}">'
            f'<i style="width:{rate:.0f}%"></i></span></td></tr>'
        )
    return (
        '<div class="scroll"><table>'
        "<thead><tr><th>Criterion</th><th>Name</th><th>Section</th>"
        "<th>Asked</th><th>Passed</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _seed_matrix(runs: List[dict]) -> str:
    """Seeds down, runs across — where a seed is hard for everything, it shows."""
    ordered = sorted(runs, key=lambda r: r.get("started_at", ""))
    seeds: Dict[str, Dict[str, Optional[float]]] = {}
    for run in ordered:
        for seed in run["seeds"]:
            seeds.setdefault(seed["seed_id"], {})[run["run_id"]] = seed["score"]
    head = "".join(f'<th>{e(r["run_id"][-6:])}</th>' for r in ordered)
    rows = []
    for seed_id, by_run in sorted(seeds.items()):
        cells = []
        for run in ordered:
            score = by_run.get(run["run_id"])
            cells.append(
                f'<td class="n">{score:.0f}</td>' if isinstance(score, (int, float))
                else '<td class="n dim">—</td>'
            )
        rows.append(f'<tr><td class="val">{e(seed_id)}</td>{"".join(cells)}</tr>')
    return (
        '<div class="scroll"><table>'
        f"<thead><tr><th>Seed</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _verdict_blocks(runs: List[dict], lib: RubricLibrary) -> str:
    """The judge's own words, for the most recent run — the reason to publish at all."""
    if not runs:
        return ""
    run = max(runs, key=lambda r: r.get("started_at", ""))
    blocks = []
    for seed in run["seeds"]:
        if not seed["scores"]:
            continue
        failed = [s for s in seed["scores"] if not s["passed"]]
        rows = "".join(
            f'<div class="crit"><div class="cid">{e(_live(s["criterion"], lib) or s["criterion"])}</div>'
            f'<div class="chead"><span class="cname">'
            f'{"passed" if s["passed"] else "failed"}</span></div>'
            f'<p class="cdesc">{e(s["comment"] or "—")}</p></div>'
            for s in seed["scores"]
        )
        score = seed["score"]
        blocks.append(
            f'<details class="seed"><summary><span class="sid">{e(seed["seed_id"])}</span>'
            f'<span class="scat">{e(seed["category"])}</span>'
            f'<span class="scount2">'
            f'{f"{score:.0f}" if isinstance(score, (int, float)) else "—"} · '
            f'{len(failed)} of {len(seed["scores"])} failed</span></summary>'
            f'<div class="crits">{rows}</div></details>'
        )
    return (
        f'<p class="lede">Every criterion the judge answered on the most recent run, '
        f'<code>{e(run["run_id"])}</code>{e(" — " + run["variant"] if run["variant"] else "")}, '
        f"with the reason it gave. This is what a score is made of.</p>"
        f'<div class="rule"></div>{"".join(blocks)}'
    )


def render_performance(ds: Dataset, data: dict, nav_html: str) -> str:
    lib = ds.rubrics
    runs = data.get("runs", [])
    best = headline(runs)
    judged = sum(
        1 for r in runs for s in r["seeds"] if isinstance(s["score"], (int, float))
    )
    answered = sum(len(s["scores"]) for r in runs for s in r["seeds"])

    if not runs:
        body = """
<header class="top"><div class="wrap">
<div class="eyebrow">video-eval-bench</div><h1>Performance</h1>
<p class="standfirst">No run has been published yet. Export one with
<code>make site-data</code> and commit <code>site/data/runs.json</code>.</p>
</div></header>"""
        return _page("Benchmark Performance", nav_html, body + _footer())

    body = f"""
<header class="top"><div class="wrap">
<div class="eyebrow">video-eval-bench · published runs</div>
<h1>Performance</h1>
<p class="standfirst">Every run exported to this site, the criteria that break models
most often, and the judge's written reason for each verdict. Scores are
weight-earned over weight-asked, so they compare across seeds with different rubric
sizes — but only within the same dataset revision.</p>
<div class="tierbar">
<div class="tierchip t1"><b>{len(runs)} runs</b><span>{judged} judged seeds</span></div>
<div class="tierchip t2"><b>{answered} verdicts</b><span>one model call each</span></div>
<div class="tierchip t3"><b>best {best["summary"]["mean_score"]:.1f}</b>
<span>over {best["summary"].get("n_ok") or 0} judged seeds</span></div>
</div>
</div></header>

<div class="wrap"><main>

<section class="block" id="runs">
<h2>Runs</h2>
<p class="lede">One row per published run. The arm is the configuration that produced
it — which generator, which prompt, which skills, which judge. Wall clock is the whole
run, generation included, which is where the hours go.</p>
{_runs_table(runs)}
</section>

<section class="block" id="criteria">
<h2>What models fail</h2>
<p class="lede">Every verdict from every run, grouped by criterion and sorted by pass
rate — hardest first. <b>Asked</b> counts the times a seed carrying the criterion was
judged on it. Each id links to its entry in the atlas.</p>
{_criterion_table(runs, lib)}
<p class="foot-note">Runs made before the rubric review name criteria that have since
merged: their verdicts are counted against the successor
({", ".join(f"<code>{e(k)}</code> → <code>{e(v)}</code>" for k, v in atlas.SUPERSEDED.items() if v)}),
and the one criterion that was deleted outright is dropped rather than counted as
something else.</p>
</section>

<section class="block" id="seeds">
<h2>Seeds across runs</h2>
<p class="lede">The same seeds, run after run. A row that is low everywhere is a hard
brief; a row that moves is a brief the arm actually changed something for.</p>
{_seed_matrix(runs)}
</section>

<section class="block" id="verdicts">
<h2>Verdicts</h2>
{_verdict_blocks(runs, lib)}
</section>

</main></div>
{_footer(f'Snapshot generated {e(data.get("generated_at", "")[:19])}. ')}
"""
    return _page("Benchmark Performance", nav_html, body)


# ── build ─────────────────────────────────────────────────────────────────────


def build(
    out: Path,
    dataset_dir: Optional[Path] = None,
    data_file: Path = DEFAULT_DATA,
    pilot: Optional[Path] = None,
    repo: str = "https://github.com/AdMoR/ShortVideoStoryBench",
) -> List[Path]:
    ds = load_dataset(dataset_dir)
    data = json.loads(data_file.read_text()) if data_file.exists() else {"runs": []}
    # The pilot is not checked in (it is a build output), so CI renders the atlas's
    # coverage over the benchmark's own seeds instead. Both are real corpora; the
    # page names whichever one it counted.
    if pilot and pilot.exists():
        corpus_seeds, corpus = load_dataset(pilot).seeds, "the FineVideo pilot"
    else:
        corpus_seeds, corpus = ds.seeds, "the benchmark's dataset"

    out.mkdir(parents=True, exist_ok=True)
    pages = {
        "index.html": render_index(ds, data, nav("index.html", repo), repo),
        "atlas.html": atlas.render(
            ds.rubrics,
            corpus_seeds,
            nav=EXTRA_CSS + nav("atlas.html", repo),
            corpus=corpus,
        ),
        "performance.html": render_performance(ds, data, nav("performance.html", repo)),
    }
    written = []
    for name, page in pages.items():
        path = out / name
        path.write_text(page)
        written.append(path)
    # Pages runs the site through Jekyll unless told not to; a leading-underscore
    # directory would vanish and the build is already static.
    (out / ".nojekyll").write_text("")
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render the site")
    b.add_argument("--out", type=Path, default=Path("_site"))
    b.add_argument("--dataset", type=Path, default=None)
    b.add_argument("--data", type=Path, default=DEFAULT_DATA)
    b.add_argument("--pilot", type=Path, default=Path("dataset_finevideo_pilot"))
    b.add_argument("--repo", default="https://github.com/AdMoR/ShortVideoStoryBench")

    x = sub.add_parser("export", help="snapshot report.json files for publication")
    x.add_argument("reports", nargs="+", type=Path)
    x.add_argument("--out", type=Path, default=DEFAULT_DATA)
    x.add_argument(
        "--include-mock",
        action="store_true",
        help="publish mock/self-test runs too (they pass everything)",
    )

    args = ap.parse_args(argv)
    if args.cmd == "export":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = snapshot(args.reports, include_mock=args.include_mock)
        args.out.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
        print(f"{args.out} — {len(data['runs'])} run(s), {args.out.stat().st_size // 1024}KB")
        return
    written = build(args.out, args.dataset, args.data, args.pilot, args.repo)
    print(f"{args.out}/ — {', '.join(p.name for p in written)}")


if __name__ == "__main__":
    main()
