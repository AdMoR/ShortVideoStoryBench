"""
One published run, shown rather than tabulated.

    python -m video_eval_bench.report.site example runs/20260825-072222

The performance page counts every run; this is the one place a reader can watch what
an arm actually produced. It exports a single run — the arm's configuration, what the
generation cost, and for three of its seeds the brief, the clip, the agent's trace and
the judge's answers — into `site/data/example.json` plus a handful of small files in
`site/media/`.

**Why an export and not a read of `runs/`.** Same reason as the run snapshot: `runs/`
is gitignored and CI has no copy of it. A run directory holds ~150 MB of transcript
per seed and mp4s at generation bitrate; neither belongs in a repository. So the
export condenses and shrinks:

  * the trace goes through `report.html.condense_trace` (which drops streaming
    deltas) and is then trimmed again for the web — a public page is read, not
    audited, and the full stream stays in the run directory,
  * each clip is re-encoded to fit a 640px box at CRF 30, which turns 8 MB of 5-second
    video into a few hundred KB without changing anything a viewer is judging it on,
  * a poster frame is extracted so the page costs one small JPEG per seed until
    someone presses play.

**Which run, and which seeds.** The run is named on the command line; the seeds
default to the best, the median and the worst of the ones it judged, because three
clips that all scored the same teach nothing. The highest-scoring run is not
necessarily the right one to show: an arm with `generator=external` has no agent and
therefore no trace, and `--seeds` is there for when the automatic pick is not the
interesting one.
"""

import html
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from video_eval_bench.report.html import condense_trace

e = html.escape

# The trace on this page is an illustration, not the record. `condense_trace` has
# already thrown away the streaming deltas; these caps throw away the long tail of
# bash output that would otherwise be most of the committed JSON.
WEB_MAX_ITEMS = 120
WEB_MAX_TEXT = 900
WEB_MAX_ARGS = 320
WEB_MAX_OUTPUT = 420

# A 640px box at CRF 30. Small enough to commit, large enough that a reader can see
# the thing the judge saw — and the judge itself works from 8 sampled frames.
MEDIA_BOX = 640
MEDIA_CRF = 30

MEDIA_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi")


# ── picking what to show ──────────────────────────────────────────────────────


def pick_seeds(results: List[dict], n: int = 3) -> List[str]:
    """
    The best, the worst, and the middle — in that spread, not that order.

    A page that shows three of the run's strongest clips is a brochure. The point of
    an example is the range: what the arm does when it works, what it does when it
    does not, and the ordinary case in between.
    """
    scored = [
        r for r in results
        if isinstance((r.get("verdict") or {}).get("total_score"), (int, float))
    ]
    scored.sort(key=lambda r: r["verdict"]["total_score"])
    if not scored:
        return []
    if len(scored) <= n:
        return [r["seed_id"] for r in scored]
    picks = [scored[-1], scored[len(scored) // 2], scored[0]][:n]
    # Report them best-first: the reader meets the arm at its best, then sees the cost.
    return [r["seed_id"] for r in picks]


# ── media ─────────────────────────────────────────────────────────────────────


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args], check=True
    )


def _probe(path: Path) -> dict:
    """Width, height, duration and whether there is any audio at all."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,width,height:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "duration": round(float((data.get("format") or {}).get("duration") or 0), 2),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def transcode(src: Path, media_dir: Path, stem: str) -> dict:
    """
    A committable clip and its poster frame.

    The scale filter is written twice on purpose: `force_original_aspect_ratio` fits
    the clip inside the box but will happily produce an odd height, which libx264
    refuses at yuv420p. The second pass rounds both dimensions down to even.
    """
    fit = (
        f"scale=w={MEDIA_BOX}:h={MEDIA_BOX}:force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    video = media_dir / f"{stem}.mp4"
    poster = media_dir / f"{stem}.jpg"
    _ffmpeg(
        "-i", str(src), "-vf", fit,
        "-c:v", "libx264", "-crf", str(MEDIA_CRF), "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "64k", str(video),
    )
    _ffmpeg("-i", str(src), "-vf", fit, "-frames:v", "1", "-q:v", "6", str(poster))
    probe = _probe(video)
    return {
        "video": video.name,
        "poster": poster.name,
        "bytes": video.stat().st_size,
        "source_bytes": src.stat().st_size,
        **probe,
    }


def _source_video(run_dir: Path, seed_id: str) -> Optional[Path]:
    for suffix in MEDIA_SUFFIXES:
        candidate = run_dir / f"{seed_id}{suffix}"
        if candidate.exists():
            return candidate.resolve()  # runs may symlink a clip from an earlier run
    return None


# ── the export ────────────────────────────────────────────────────────────────


def _trim(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _headline(args) -> str:
    """
    The gist of a call, so the collapsed trace reads without opening anything.

    A bash call's `command` is the whole story; dumping the JSON envelope around it
    puts `{"command": "` in front of every line and pushes the story off the edge.
    """
    if isinstance(args, str):
        return _trim(args.replace("\n", " "), 110)
    if not isinstance(args, dict):
        return ""
    for key in ("command", "path", "file_path", "pattern", "query"):
        if key in args:
            return _trim(str(args[key]).replace("\n", " "), 110)
    return _trim(json.dumps(args, ensure_ascii=False), 110)


def web_trace(transcript: Path) -> List[dict]:
    """A condensed trace, trimmed again to what a web page should carry."""
    items = []
    for item in condense_trace(transcript)[:WEB_MAX_ITEMS]:
        kind = item.get("kind")
        if kind in ("text", "thinking"):
            trimmed = {"kind": kind, "turn": item["turn"],
                       "text": _trim(item.get("text", ""), WEB_MAX_TEXT)}
            if item.get("partial"):
                trimmed["partial"] = True
            if item.get("redacted"):
                trimmed["redacted"] = True
        elif kind == "tool":
            args = item.get("args")
            args_text = args if isinstance(args, str) else json.dumps(
                args, indent=1, ensure_ascii=False
            )
            trimmed = {
                "kind": "tool", "turn": item["turn"], "name": item.get("name", "?"),
                "head": _headline(args),
                "args": _trim(args_text, WEB_MAX_ARGS),
                "output": _trim(item.get("output") or "", WEB_MAX_OUTPUT),
                "error": bool(item.get("error")),
            }
        elif kind == "usage":
            trimmed = {"kind": "usage", "turn": item["turn"],
                       "tokens": item.get("tokens")}
        else:
            trimmed = {"kind": "note", "turn": item.get("turn", 0),
                       "text": _trim(item.get("text", ""), 200)}
        items.append(trimmed)
    if len(items) == WEB_MAX_ITEMS:
        items.append({"kind": "note", "turn": items[-1].get("turn", 0),
                      "text": "trace truncated for the web — the full stream is in "
                              "the run's transcript.jsonl"})
    return items


def _params(config: dict, choices: dict) -> dict:
    """
    The arm, as the fields a reader would need to reproduce it.

    `choices` are the Hydra group names — `system_prompt=director` — and the config
    is what those resolved to. Both are worth keeping: the group name is what you
    would type, the resolved value is what actually ran.
    """
    gen = config.get("generator", {}) or {}
    judge = config.get("judge", {}) or {}
    model = gen.get("model", {}) or {}
    prompt = gen.get("system_prompt", {}) or {}
    skills = (gen.get("skills", {}) or {}).get("paths") or []
    tools = (gen.get("tools", {}) or {}).get("builtin") or []
    return {
        "choices": choices,
        "generator": {
            "agent": gen.get("kind"),
            "model": model.get("name"),
            "provider": model.get("provider"),
            "system_prompt": choices.get("system_prompt")
            or Path(prompt.get("system_file") or "").stem
            or None,
            "skills": [Path(p).name for p in skills],
            "tools": list(tools),
            "sandbox": (gen.get("sandbox", {}) or {}).get("kind"),
            "offline": gen.get("offline"),
            "timeout_seconds": gen.get("timeout_seconds"),
        },
        "judge": {
            "backend": judge.get("backend"),
            "model": judge.get("model"),
            "provider": judge.get("provider"),
            "n_frames": judge.get("n_frames"),
            "timeout_seconds": judge.get("timeout_seconds"),
        },
    }


def _cost(results: List[dict]) -> dict:
    """What the generation cost over the whole run, not just the seeds shown."""
    turns = tokens = 0
    calls: Dict[str, int] = {}
    for result in results:
        meta = result.get("metadata") or {}
        turns += meta.get("turns") or 0
        tokens += (meta.get("usage") or {}).get("totalTokens") or 0
        for name, n in (meta.get("tool_calls") or {}).items():
            calls[name] = calls.get(name, 0) + n
    return {"turns": turns, "tokens": tokens, "tool_calls": calls}


def _generation(result: dict) -> dict:
    meta = result.get("metadata") or {}
    return {
        "turns": meta.get("turns"),
        "tokens": (meta.get("usage") or {}).get("totalTokens"),
        "tool_calls": meta.get("tool_calls") or {},
        "duration_seconds": result.get("duration_seconds"),
        "outcome": meta.get("outcome"),
        "submit_notes": _trim(meta.get("submit_notes") or "", 700),
    }


def export(
    run_dir: Path,
    media_dir: Path,
    seed_ids: Optional[List[str]] = None,
    n_seeds: int = 3,
    briefs: Optional[Dict[str, str]] = None,
) -> Tuple[dict, List[Path]]:
    """
    The example bundle, and the media files written for it.

    Seeds with no clip on disk are dropped rather than shown as an empty player: a
    generation that died has nothing to watch, and the runs table already says so.
    """
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    results = {r["seed_id"]: r for r in report.get("results", []) or []}

    wanted = seed_ids or pick_seeds(list(results.values()), n_seeds)
    missing = [s for s in wanted if s not in results]
    if missing:
        raise SystemExit(f"{run_dir}: no such seed in this run: {', '.join(missing)}")

    media_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    seeds = []
    for seed_id in wanted:
        result = results[seed_id]
        source = _source_video(run_dir, seed_id)
        if source is None:
            print(f"  {seed_id}: no video in the run directory — skipped")
            continue
        media = transcode(source, media_dir, seed_id)
        written += [media_dir / media["video"], media_dir / media["poster"]]
        verdict = result.get("verdict") or {}
        seeds.append(
            {
                "seed_id": seed_id,
                "category": result.get("category", ""),
                "brief": (briefs or {}).get(seed_id, ""),
                "score": verdict.get("total_score"),
                "passed": verdict.get("passed"),
                "media": media,
                "generation": _generation(result),
                "scores": [
                    {
                        "criterion": s.get("criterion"),
                        "passed": bool(s.get("passed")),
                        "comment": _trim(s.get("comment") or "", 400),
                    }
                    for s in verdict.get("scores") or []
                ],
                "safety_violations": [
                    s.get("check_id") for s in verdict.get("safety") or []
                    if s.get("violation")
                ],
                "trace": web_trace(run_dir / seed_id / "transcript.jsonl"),
            }
        )

    summary = report.get("summary", {}) or {}
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": summary.get("run_id", run_dir.name),
        "variant": report.get("variant", "") or summary.get("variant", ""),
        "started_at": report.get("started_at", ""),
        "note": report.get("note", "") or "",
        "params": _params(
            report.get("config", {}) or {}, report.get("choices", {}) or {}
        ),
        "summary": {
            k: summary.get(k)
            for k in ("n_seeds", "n_ok", "n_passed", "mean_score",
                      "n_generation_errors", "total_duration_seconds")
        },
        "cost": _cost(report.get("results", []) or []),
        "seeds": seeds,
    }
    return data, written


def prune(media_dir: Path, keep: List[Path]) -> List[Path]:
    """
    Drop media from a previous example that this one no longer references.

    Without this, re-exporting from a different run leaves the old run's clips in the
    repository forever — committed, deployed, and reachable by anyone who kept a link.
    """
    if not media_dir.exists():
        return []
    kept = {p.name for p in keep}
    removed = []
    for path in sorted(media_dir.iterdir()):
        if path.is_file() and path.name not in kept:
            path.unlink()
            removed.append(path)
    return removed


def copy_media(media_dir: Path, out_dir: Path) -> int:
    """Media into the built site. Returns how many files were copied."""
    if not media_dir.exists():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(media_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, out_dir / path.name)
            n += 1
    return n


# ── the rendered section ──────────────────────────────────────────────────────
#
# Everything below is closed by default. The section sits at the top of the
# performance page and would otherwise push the tables — which is what most readers
# came for — several screens down. `<details>` also keeps the videos out of the
# network until someone asks: `preload="none"` plus a poster means an unopened seed
# costs one JPEG.

CSS = """
<style>
#example .seed > summary::before{content:"▸"; color:var(--accent); font-size:.7rem}
#example .seed[open] > summary::before{content:"▾"}
#example .scount2 b{color:var(--ink); font-weight:600; font-size:.86rem}

.armspecs{display:grid; gap:.7rem; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  margin:1.2rem 0}
.armspec{border:1px solid var(--line); border-radius:3px; background:var(--raise);
  padding:.9rem 1rem}
.armspec .flabel{margin-top:0}
.spec{display:grid; grid-template-columns:7.5rem minmax(0,1fr); gap:.15rem .8rem;
  padding:.22rem 0; border-bottom:1px solid var(--line); align-items:baseline}
.spec:last-child{border-bottom:none}
.spec span{font-family:"JetBrains Mono",monospace; font-size:.66rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--faint)}
.spec b{font-family:"JetBrains Mono",monospace; font-size:.76rem; font-weight:600;
  color:var(--ink); word-break:break-word}
.spec b.off{color:var(--faint); font-weight:400}

.exgrid{display:grid; gap:1.2rem; grid-template-columns:minmax(0,260px) minmax(0,1fr);
  align-items:start; margin:.9rem 0 1.1rem}
@media (max-width:720px){.exgrid{grid-template-columns:minmax(0,1fr)}}
.clip{margin:0}
.clip video{width:100%; max-height:340px; border:1px solid var(--line-hard); border-radius:3px;
  background:#000; display:block}
.clip figcaption{font-family:"JetBrains Mono",monospace; font-size:.64rem; color:var(--faint);
  margin-top:.35rem; line-height:1.5}
.exside .flabel{margin-top:1rem}
.exside .flabel:first-child{margin-top:0}
.exside p{margin:.3rem 0 0; font-size:.92rem; color:var(--body); max-width:70ch}

details.trace{border:1px solid var(--line); border-radius:3px; background:var(--sunk);
  margin-top:.6rem}
details.trace > summary{list-style:none; cursor:pointer; padding:.55rem .8rem;
  font-family:"Archivo",sans-serif; font-size:.84rem; color:var(--ink);
  display:flex; gap:.6rem; align-items:baseline}
details.trace > summary::-webkit-details-marker{display:none}
details.trace > summary::before{content:"▸"; color:var(--accent); font-size:.7rem}
details.trace[open] > summary::before{content:"▾"}
details.trace > summary:hover{background:var(--raise)}
details.trace > summary .dim{margin-left:auto; font-family:"JetBrains Mono",monospace;
  font-size:.7rem}
.tbody{padding:.2rem .8rem .8rem; border-top:1px solid var(--line)}

.turnhead{font-family:"JetBrains Mono",monospace; font-size:.64rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--faint); margin:.9rem 0 .35rem;
  padding-bottom:.2rem; border-bottom:1px solid var(--line)}
.think{border-left:2px solid var(--line-hard); padding:.15rem 0 .15rem .7rem; margin:.35rem 0;
  font-size:.86rem; color:var(--muted); white-space:pre-wrap}
.think .lbl,.say .lbl{font-family:"JetBrains Mono",monospace; font-size:.6rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); display:block}
.say{margin:.35rem 0; font-size:.92rem; color:var(--body); white-space:pre-wrap}
.flag{font-family:"JetBrains Mono",monospace; font-size:.6rem; color:var(--crit);
  letter-spacing:.06em}
details.tool{border:1px solid var(--line); border-radius:2px; background:var(--raise);
  margin:.3rem 0}
details.tool > summary{list-style:none; cursor:pointer; padding:.3rem .6rem; display:flex;
  gap:.55rem; align-items:baseline; min-width:0}
details.tool > summary::-webkit-details-marker{display:none}
details.tool > summary code{font-size:.72rem; color:var(--accent); flex:none}
details.tool.bad{border-color:var(--crit-soft)}
details.tool.bad > summary code{color:var(--crit)}
.thead{font-family:"JetBrains Mono",monospace; font-size:.7rem; color:var(--muted);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
details.tool pre{margin:.2rem .6rem .5rem; padding:.45rem .6rem; background:var(--sunk);
  border:1px solid var(--line); border-radius:2px; overflow-x:auto;
  font-family:"JetBrains Mono",monospace; font-size:.68rem; line-height:1.5;
  color:var(--body); white-space:pre-wrap; word-break:break-word}
details.tool pre.out{color:var(--muted)}

.vmark{font-family:"JetBrains Mono",monospace; font-size:.62rem; letter-spacing:.08em;
  text-transform:uppercase; padding:.05rem .3rem; border-radius:2px; border:1px solid}
.vmark.ok{color:var(--ev-motion); border-color:var(--ev-motion)}
.vmark.no{color:var(--crit); border-color:var(--crit); background:var(--crit-soft)}
</style>
"""


def _hms(seconds) -> str:
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "—"
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _num(value) -> str:
    return f"{value:,}".replace(",", " ") if isinstance(value, int) else "—"


def _tools_line(calls: dict) -> str:
    return " · ".join(f"{k}×{v}" for k, v in sorted(calls.items(), key=lambda kv: -kv[1]))


def _spec(rows: List[Tuple[str, str]]) -> str:
    out = []
    for label, value in rows:
        off = "" if value else ' class="off"'
        out.append(
            f'<div class="spec"><span>{e(label)}</span>'
            f"<b{off}>{e(value or '—')}</b></div>"
        )
    return "".join(out)


def _arm(data: dict) -> str:
    """The configuration, as two spec sheets — what generated, and what judged."""
    gen = data["params"]["generator"]
    judge = data["params"]["judge"]
    generation = [
        ("agent", gen.get("agent") or ""),
        ("model", gen.get("model") or ""),
        ("provider", gen.get("provider") or ""),
        ("system prompt", gen.get("system_prompt") or ""),
        ("skills", ", ".join(gen.get("skills") or []) or "none"),
        ("tools", ", ".join(gen.get("tools") or []) or "none"),
        ("sandbox", gen.get("sandbox") or ""),
        ("network", "offline" if gen.get("offline") else "online"),
        ("budget", f"{gen['timeout_seconds'] // 60} min per seed"
                   if gen.get("timeout_seconds") else ""),
    ]
    judging = [
        ("backend", judge.get("backend") or ""),
        ("model", judge.get("model") or ""),
        ("provider", judge.get("provider") or ""),
        ("evidence", f"{judge['n_frames']} sampled frames"
                     if judge.get("n_frames") else ""),
        ("call budget", f"{judge['timeout_seconds']}s per criterion"
                        if judge.get("timeout_seconds") else ""),
        ("questions", "one model call each, answered pass or fail"),
    ]
    return (
        '<div class="armspecs">'
        f'<div class="armspec"><div class="flabel">Generation</div>{_spec(generation)}</div>'
        f'<div class="armspec"><div class="flabel">Judging</div>{_spec(judging)}</div>'
        "</div>"
    )


def _cost_tiles(data: dict) -> str:
    """What the run cost and what it got for it, before any of the detail."""
    summary, cost = data["summary"], data["cost"]
    score = summary.get("mean_score")
    calls = sum((cost.get("tool_calls") or {}).values())
    errors = summary.get("n_generation_errors") or 0

    tiles = []
    if isinstance(score, (int, float)):
        tiles.append(
            f"<div class=\"tile\"><span>mean score</span><b>{score:.1f}</b>"
            f"<i>over {summary.get('n_ok') or 0} judged seeds, "
            f"{summary.get('n_passed') or 0} of them passing</i></div>"
        )
    tiles += [
        f"<div class=\"tile\"><span>wall clock</span>"
        f"<b>{_hms(summary.get('total_duration_seconds'))}</b>"
        f"<i>for {summary.get('n_seeds') or 0} seeds, generation included</i></div>",

        f"<div class=\"tile\"><span>agent turns</span><b>{_num(cost.get('turns'))}</b>"
        f"<i>{_num(calls)} tool calls across the run</i></div>",

        f"<div class=\"tile\"><span>tokens</span><b>{_num(cost.get('tokens'))}</b>"
        f"<i>read and written by the agent</i></div>",

        f"<div class=\"tile\"><span>generations failed</span><b>{errors}</b>"
        f"<i>{'ran out of the per-seed budget' if errors else 'every seed produced a clip'}"
        f"</i></div>",
    ]
    return f"<div class=\"tiles\">{''.join(tiles)}</div>"


def _bytes(n: int) -> str:
    return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:,.0f} KB"


def _clip(seed: dict) -> str:
    media = seed.get("media") or {}
    if not media.get("video"):
        return '<p class="dim">No clip was kept for this seed.</p>'
    # Only claim a re-encode when it actually bought something: "51 KB, re-encoded
    # from 0.2 MB" is noise on a clip that was already small.
    source = media.get("source_bytes") or 0
    shrunk = (
        f' · re-encoded from {_bytes(source)}'
        if source > 2 * (media.get("bytes") or 0) else ""
    )
    return (
        f'<figure class="clip"><video controls preload="none" playsinline '
        f'poster="media/{e(media["poster"])}" src="media/{e(media["video"])}"></video>'
        f'<figcaption>{media.get("width")}×{media.get("height")} · '
        f'{media.get("duration")}s · '
        f'{"with audio" if media.get("has_audio") else "no audio track"} · '
        f'{_bytes(media["bytes"])}{shrunk}</figcaption></figure>'
    )


def _trace_items(items: List[dict]) -> str:
    """Grouped by turn, because one turn is one model call and everything it asked for."""
    out, seen = [], None
    tokens = {}
    for item in items:
        if item.get("kind") == "usage":
            tokens[item["turn"]] = max(tokens.get(item["turn"], 0), item.get("tokens") or 0)
    for item in items:
        kind = item.get("kind")
        if kind == "usage":
            continue
        turn = item.get("turn", 0)
        if turn != seen:
            seen = turn
            count = tokens.get(turn)
            label = f"turn {turn}" + (f" · {_num(count)} tokens" if count else "")
            out.append(f'<div class="turnhead">{e(label)}</div>')
        flag = ' <span class="flag">interrupted</span>' if item.get("partial") else ""
        if kind == "thinking":
            body = "(redacted)" if item.get("redacted") else item.get("text", "")
            out.append(f'<div class="think"><span class="lbl">thinking</span>{flag}'
                       f"{e(body)}</div>")
        elif kind == "text":
            out.append(f'<div class="say"><span class="lbl">says</span>{flag}'
                       f'{e(item.get("text", ""))}</div>')
        elif kind == "note":
            out.append(f'<p class="dim">{e(item.get("text", ""))}</p>')
        else:
            bad = " bad" if item.get("error") else ""
            head = item.get("head") or ""
            output = item.get("output")
            out.append(
                f'<details class="tool{bad}"><summary><code>{e(item.get("name", "?"))}</code>'
                f'<span class="thead">{e(head)}</span></summary>'
                f'<pre>{e(item.get("args") or "—")}</pre>'
                + (f'<pre class="out">{e(output)}</pre>' if output else "")
                + "</details>"
            )
    return "".join(out) or '<p class="dim">No transcript was recorded for this seed.</p>'


def _verdict(seed: dict, lib, live) -> str:
    """
    Every question the judge answered on this clip, and the answer it gave.

    A criterion the current library still defines is linked to its atlas entry and
    named. One it does not is shown as the run wrote it, under a note — the older
    runs were judged on a fixed per-category rubric that the review replaced, and
    printing those ids with an empty name and a dead link would read as a bug rather
    than as the history it is.
    """
    rows, orphans = [], 0
    for score in seed.get("scores") or []:
        cid = live(score["criterion"])
        if cid:
            head = (
                f'<a href="atlas.html#c-{e(cid)}">{e(cid)}</a>',
                e(lib.get(cid).name),
            )
        else:
            orphans += 1
            head = (e(score["criterion"]), "not in the current library")
        mark = "ok" if score["passed"] else "no"
        rows.append(
            f'<div class="crit"><div class="cid">{head[0]}</div>'
            f'<div class="chead"><span class="cname">{head[1]}</span>'
            f'<span class="vmark {mark}">'
            f'{"pass" if score["passed"] else "fail"}</span></div>'
            f'<p class="cdesc">{e(score.get("comment") or "—")}</p></div>'
        )
    note = ""
    if orphans:
        note = (
            f'<p class="dim">{orphans} of these criteria are not in the rubric library '
            "as it stands: this run predates the review that replaced the fixed "
            "per-category rubrics, so its ids have no atlas entry to link to. The "
            "scores are what the judge gave at the time.</p>"
        )
    violations = seed.get("safety_violations") or []
    if violations:
        note += (
            f'<p class="dim">Safety veto: {e(", ".join(violations))}.</p>'
        )
    return f'{note}<div class="crits">{"".join(rows)}</div>'


def _seed(seed: dict, lib, live) -> str:
    gen = seed.get("generation") or {}
    scores = seed.get("scores") or []
    failed = sum(1 for s in scores if not s["passed"])
    score = seed.get("score")
    calls = sum((gen.get("tool_calls") or {}).values())
    head = f"{score:.0f}" if isinstance(score, (int, float)) else "—"
    line = " · ".join(
        part for part in (
            f"{len(scores) - failed}/{len(scores)} criteria passed" if scores else "",
            f"{gen['turns']} turns" if gen.get("turns") else "",
            _hms(gen.get("duration_seconds")),
        ) if part
    )
    chips = " · ".join(
        part for part in (
            f"{_num(gen.get('tokens'))} tokens" if gen.get("tokens") else "",
            _tools_line(gen.get("tool_calls") or {}),
            f"exit: {gen['outcome']}" if gen.get("outcome") else "",
        ) if part
    )
    notes = (
        '<div class="flabel">What the agent said it delivered</div>'
        f'<p>{e(gen["submit_notes"])}</p>' if gen.get("submit_notes") else ""
    )
    return f"""
<details class="seed">
<summary><span class="sid">{e(seed["seed_id"])}</span>
<span class="scat">{e(seed.get("category", ""))}</span>
<span class="scount2"><b>{e(head)}</b> · {e(line)}</span></summary>
<div class="sbody">
<div class="exgrid">
{_clip(seed)}
<div class="exside">
<div class="flabel">The brief</div><p>{e(seed.get("brief") or "—")}</p>
{notes}
<div class="flabel">What this seed cost</div><p class="dim">{e(chips or "not recorded")}</p>
</div>
</div>
<details class="trace"><summary>Agent trace
<span class="dim">{gen.get("turns") or "?"} turns · {calls} tool calls</span></summary>
<div class="tbody">{_trace_items(seed.get("trace") or [])}</div></details>
<details class="trace"><summary>Judge verdict
<span class="dim">{len(scores)} criteria · {failed} failed</span></summary>
<div class="tbody">{_verdict(seed, lib, live)}</div></details>
</div>
</details>"""


def render_section(data: dict, lib, live) -> str:
    """The whole example, as one section of the performance page."""
    if not data or not data.get("seeds"):
        return ""
    seeds = data["seeds"]
    return f"""
<section class="block" id="example">
<h2>One run, up close</h2>
<p class="lede">The tables below count every run. This is one of them —
<code>{e(data["run_id"])}</code>{e(", " + data["variant"] if data["variant"] else "")} —
with the clips it produced, what the agent did to produce them, and what the judge
answered. {len(seeds)} of its seeds are shown: the best, the middling and the worst,
so the spread is visible rather than the highlight reel. Everything is collapsed;
open a seed to watch it.</p>
{_cost_tiles(data)}
{_arm(data)}
{"".join(_seed(seed, lib, live) for seed in seeds)}
<p class="foot-note">Clips are re-encoded to fit a 640px box so they can live in the
repository; the judge saw the originals. Traces are condensed — streaming deltas
dropped, long tool output truncated — and the full stream stays in the run directory,
which is not published.</p>
</section>"""
