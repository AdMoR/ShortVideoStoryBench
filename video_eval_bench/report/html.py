"""
HTML report for one benchmark run.

Writes a single `report.html` into the run directory, beside the videos it
references. Everything a reviewer needs for one run is in that one file: the
summary, the exact config, and per seed — the brief, the video, what the agent
actually did, and how the judge scored every criterion.

Three constraints shape it:

  * **It must open instantly.** A raw transcript is ~150KB per seed and 95% of
    that is streaming deltas. The trace is condensed to the events a human reads
    — assistant text and tool calls — with long outputs truncated. An eight-seed
    run lands in the low hundreds of KB.
  * **The video is referenced, not embedded.** Base64 would multiply the page
    size by the size of the videos. `report.html` therefore belongs in the run
    directory and is not portable on its own — move the folder, not the file.
  * **No external assets and no framework.** Inline CSS, `<details>` for the
    expandable sections, and a few lines of JS for expand-all. It opens from
    `file://` with no network.
"""

import html
import json
import logging
from pathlib import Path
from typing import List, Optional

from video_eval_bench.pi_ndjson import parse_line

logger = logging.getLogger(__name__)

# The trace is for reading, not archiving — the full stream stays in
# transcript.jsonl next door.
MAX_ARGS_CHARS = 1500
MAX_OUTPUT_CHARS = 2500
MAX_TEXT_CHARS = 4000
MAX_TRACE_ITEMS = 200

# Per-seed caps bound a seed, not the page: eight chatty seeds at the per-seed
# maximum would be several MB. This is the page-wide ceiling for tool output,
# spent in seed order — so a long run degrades to summarised later traces with a
# visible note, instead of producing a report too heavy to open.
TOTAL_OUTPUT_BUDGET = 1_000_000

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".avi")


# ── trace extraction ──────────────────────────────────────────────────────────


def condense_trace(transcript: Path) -> List[dict]:
    """
    The readable spine of a pi run: the model's reasoning, its prose, and its
    tool calls, in order.

    `message_update` deltas are not kept as items — they are the same content
    arriving one token at a time, and they are almost the whole file. They are
    accumulated into a small buffer purely so that a run killed mid-message
    (a timeout, an early exit) still shows what the model was saying when it
    died, which is exactly the moment worth seeing. The authoritative
    `message_end` always supersedes the buffer.
    """
    if not transcript.exists():
        return []

    items: List[dict] = []
    pending = {}
    turn = 0
    partial: dict = {}

    with open(transcript, errors="replace") as handle:
        for line in handle:
            event = parse_line(line)
            if event is None:
                continue
            kind = event.get("type")

            if kind == "turn_start":
                turn += 1

            elif kind == "message_start":
                partial = {}

            elif kind == "message_update":
                _accumulate(partial, event.get("assistantMessageEvent") or {})

            elif kind == "message_end":
                partial = {}  # the finished message is authoritative
                message = event.get("message", {})
                if message.get("role") != "assistant":
                    continue
                for part in message.get("content", []):
                    item = _content_item(part, turn)
                    if item:
                        items.append(item)
                usage = message.get("usage") or {}
                if usage.get("totalTokens"):
                    items.append(
                        {"kind": "usage", "turn": turn, "tokens": usage["totalTokens"]}
                    )

            elif kind == "tool_execution_start":
                item = {
                    "kind": "tool",
                    "turn": turn,
                    "name": event.get("toolName", "?"),
                    "args": event.get("args"),
                    "output": None,
                    "error": False,
                }
                pending[event.get("toolCallId")] = item
                items.append(item)

            elif kind == "tool_execution_end":
                item = pending.pop(event.get("toolCallId"), None)
                if item is None:
                    continue
                item["error"] = bool(event.get("isError"))
                item["output"] = _result_text(event.get("result"))

            if len(items) > MAX_TRACE_ITEMS:
                items.append({"kind": "note", "turn": turn, "text": "trace truncated"})
                break

    # A message that never ended: the run was killed while the model was
    # speaking. Show it, flagged, rather than losing the last thing it said.
    for kind in ("thinking", "text"):
        text = (partial.get(kind) or "").strip()
        if text:
            items.append({"kind": kind, "turn": turn, "text": text, "partial": True})

    return items


def _content_item(part, turn: int) -> Optional[dict]:
    """
    One assistant content part as a trace item.

    `thinking` carries its text under a `thinking` key, not `text` — reading only
    `text` silently drops a reasoning model's entire train of thought.
    """
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind == "text":
        text = (part.get("text") or "").strip()
        return {"kind": "text", "turn": turn, "text": text} if text else None
    if kind == "thinking":
        if part.get("redacted"):
            return {"kind": "thinking", "turn": turn, "text": "", "redacted": True}
        text = (part.get("thinking") or "").strip()
        return {"kind": "thinking", "turn": turn, "text": text} if text else None
    return None


def _accumulate(partial: dict, event: dict) -> None:
    """Buffer streaming text/thinking deltas, bounded, for the unterminated case."""
    kind = event.get("type") or ""
    for name in ("text", "thinking"):
        if kind == f"{name}_delta":
            if len(partial.get(name, "")) < MAX_TEXT_CHARS * 2:
                partial[name] = partial.get(name, "") + (event.get("delta") or "")
            return


class Budget:
    """A shrinking allowance of characters, shared across the whole page."""

    def __init__(self, total: Optional[int] = None):
        # Read at construction, not bound as a default: the ceiling is a module
        # constant callers (and tests) can adjust.
        self.remaining = TOTAL_OUTPUT_BUDGET if total is None else total

    def take(self, limit: int) -> int:
        """How many characters the next block may use."""
        return max(0, min(limit, self.remaining))

    def spend(self, used: int) -> None:
        self.remaining = max(0, self.remaining - used)


def _result_text(result) -> str:
    """
    The readable part of a tool result.

    Non-text parts are noted, not inlined: when the agent reads a frame, pi
    returns the image as base64 alongside the text. That the agent *looked* at a
    frame matters in a video benchmark; the bytes do not, and inlining them
    would add tens of KB per call to a page that has to open instantly.
    """
    if not isinstance(result, dict):
        return ""
    parts = result.get("content")
    if not isinstance(parts, list):
        return ""

    out = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" or "text" in part:
            out.append(part.get("text", ""))
        elif part.get("type") == "image":
            size = len(part.get("data") or "") * 3 // 4  # base64 -> bytes
            out.append(f"[{part.get('mimeType', 'image')} · {size / 1024:,.0f} KB]")
        else:
            out.append(f"[{part.get('type', 'non-text')}]")
    return "".join(out)


# ── rubric lookup ─────────────────────────────────────────────────────────────


def rubric_index(dataset, category_key: Optional[str] = None) -> dict:
    """
    criterion id -> its section, human name, weight and description.

    Scoped to one genre when `category_key` is given. Genre criterion ids only
    happen to be unique across genres today (E1…, M1…, G1…); nothing enforces
    it, and a reused id in a new genre would otherwise silently mislabel a
    criterion in the report. Indexing per seed makes that impossible.
    """
    index = {}
    for section, rubric in (("A", dataset.rubric_a), ("B", dataset.rubric_b)):
        for criterion in rubric.criteria:
            index[criterion.id] = {
                "section": section,
                "title": rubric.title,
                "name": criterion.name,
                "description": criterion.description,
                "weight": criterion.weight,
            }
    categories = (
        [dataset.categories[category_key]]
        if category_key and category_key in dataset.categories
        else list(dataset.categories.values())
    )
    for category in categories:
        for criterion in category.rubric.criteria:
            index[criterion.id] = {
                "section": "C",
                "title": category.name,
                "name": criterion.name,
                "description": criterion.description,
                "weight": criterion.weight,
            }
    for check in dataset.safety_checks:
        index[check.id] = {
            "section": "D",
            "title": "Safety",
            "name": check.category,
            "description": check.description,
            "weight": 0.0,
        }
    return index


# ── rendering helpers ─────────────────────────────────────────────────────────


def e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit:,} more characters]"


def score_class(value: Optional[float]) -> str:
    if value is None:
        return "muted"
    if value >= 75:
        return "good"
    if value >= 50:
        return "mid"
    return "bad"


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def duration(seconds) -> str:
    if not seconds:
        return "—"
    seconds = int(seconds)
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m{seconds % 60:02d}s"


def timestamp(value) -> str:
    """ISO timestamp to something readable — microseconds help nobody here."""
    if not value:
        return ""
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError):
        return str(value)


# ── the page ──────────────────────────────────────────────────────────────────


def render_run(run_dir: Path, dataset=None) -> Path:
    """Read `report.json` (and its neighbours) and write `report.html`."""
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"no report.json in {run_dir}")
    data = json.loads(report_path.read_text())

    if dataset is None:
        from video_eval_bench.dataset import load_dataset

        dataset_dir = (data.get("config", {}).get("run") or {}).get("dataset_dir")
        dataset = load_dataset(Path(dataset_dir) if dataset_dir else None)

    body = [
        _header(data),
        _summary(data),
        _config(data),
        "<h2>Seeds</h2>",
        '<div class="controls">'
        '<button onclick="setAll(true)">Expand all</button>'
        '<button onclick="setAll(false)">Collapse all</button>'
        "</div>",
    ]
    results = data.get("results", [])
    # A single-seed run — a smoke test or an ablation probe — has nothing to
    # scan past, so open it rather than making the reader click.
    briefs = {seed.seed_id: seed.prompt.strip() for seed in dataset.seeds}
    budget = Budget()
    for result in results:
        index = rubric_index(dataset, result.get("category"))
        body.append(
            _seed(
                result, run_dir, index, briefs, budget,
                open_by_default=len(results) == 1,
            )
        )

    page = _PAGE.format(
        title=e(f"video-eval-bench · {data.get('summary', {}).get('run_id', run_dir.name)}"),
        css=_CSS,
        body="\n".join(body),
        script=_JS,
    )
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    logger.info(f"[report] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def total_runtime(data: dict) -> Optional[float]:
    """
    Wall time for the run, derived from the seeds when the summary lacks it.

    Reports written before run_bench timed the failure path carry 0 here, which
    would show an hour-long timeout as a free run.
    """
    summary = data.get("summary", {})
    recorded = summary.get("total_duration_seconds")
    if recorded:
        return recorded
    derived = sum(
        (result.get("duration_seconds") or (result.get("metadata") or {}).get(
            "duration_seconds"
        ) or 0)
        for result in data.get("results", [])
    )
    return derived or None


def _header(data: dict) -> str:
    summary = data.get("summary", {})
    variant = data.get("variant") or "(defaults)"
    choices = data.get("choices") or {}
    chips = "".join(
        f'<span class="chip"><b>{e(k)}</b>{e(v)}</span>' for k, v in sorted(choices.items())
    )
    note = data.get("note") or ""
    return f"""
<header>
  <h1>{e(summary.get('run_id', 'run'))}</h1>
  <p class="variant">{e(variant)}</p>
  <div class="chips">{chips}</div>
  {f'<p class="note">{e(note)}</p>' if note else ''}
  <p class="muted small">{e(timestamp(data.get('started_at')))} · {e(duration(total_runtime(data)))}</p>
</header>"""


def result_status(result: dict) -> str:
    """
    A result's status, inferred for reports written before the field existed.

    Old runs only ever completed or errored — nothing could skip — so the absence
    of the key is unambiguous rather than a guess.
    """
    status = result.get("status")
    if status:
        return str(status)
    if result.get("generation_error"):
        return "errored"
    return "completed" if result.get("verdict") else "errored"


def _summary(data: dict) -> str:
    summary = data.get("summary", {})
    cards = [
        ("Overall", fmt(summary.get("mean_score")), score_class(summary.get("mean_score"))),
        ("Passed", f"{summary.get('n_passed', 0)}/{summary.get('n_seeds', 0)}", ""),
        ("Generation errors", summary.get("n_generation_errors", 0),
         "bad" if summary.get("n_generation_errors") else ""),
        # Not an error: nobody supplied a video for these seeds. It still belongs
        # on the front page, because it says how much of the benchmark ran.
        ("Skipped", summary.get("n_skipped", 0),
         "mid" if summary.get("n_skipped") else ""),
        ("Judge errors", summary.get("n_judge_errors", 0),
         "bad" if summary.get("n_judge_errors") else ""),
        ("Safety vetoes", summary.get("n_safety_vetoes", 0),
         "bad" if summary.get("n_safety_vetoes") else ""),
        ("Runtime", duration(total_runtime(data)), ""),
    ]
    tiles = "".join(
        f'<div class="card"><span class="label">{e(label)}</span>'
        f'<span class="value {cls}">{e(value)}</span></div>'
        for label, value, cls in cards
    )

    rows = "".join(
        f"<tr><td>{e(cat)}</td><td>{fmt(block.get('mean_score'))}</td>"
        f"<td>{e(block.get('n_passed', 0))}/{e(block.get('n_seeds', 0))}</td>"
        f"<td>{e(block.get('n_skipped', 0))}</td>"
        f"<td>{e(block.get('n_safety_vetoes', 0))}</td></tr>"
        for cat, block in sorted((summary.get("per_category") or {}).items())
    )
    table = (
        "<table class='cats'><thead><tr><th>Category</th><th>Score</th>"
        "<th>Passed</th><th>Skipped</th><th>Vetoes</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    ) if rows else ""

    return f'<section class="summary"><div class="cards">{tiles}</div>{table}</section>'


def _config(data: dict) -> str:
    config = data.get("config") or {}
    return f"""
<details class="config">
  <summary>Resolved configuration</summary>
  <pre>{e(json.dumps(config, indent=2, sort_keys=True))}</pre>
</details>"""


def _seed(
    result: dict,
    run_dir: Path,
    index: dict,
    briefs: dict,
    budget: "Budget",
    open_by_default: bool = False,
) -> str:
    seed_id = result.get("seed_id", "?")
    verdict = result.get("verdict") or {}
    error = result.get("generation_error")
    status = result_status(result)
    score = verdict.get("total_score")

    if error:
        badge = '<span class="badge bad">generation failed</span>'
    elif status == "skipped":
        badge = '<span class="badge muted">skipped</span>'
    elif verdict.get("safety_veto"):
        badge = '<span class="badge bad">safety veto</span>'
    elif verdict.get("passed"):
        badge = '<span class="badge good">passed</span>'
    else:
        badge = '<span class="badge mid">below threshold</span>'

    # Reports written before run_bench recorded time on the failure path show a
    # failed seed as free; the generator's own metadata still has the truth.
    elapsed = result.get("duration_seconds") or (result.get("metadata") or {}).get(
        "duration_seconds"
    )

    parts = [
        f'<p class="brief">{e(_seed_prompt(result, briefs))}</p>',
    ]
    if error:
        parts.append(f'<div class="error"><b>Generation failed</b><pre>{e(error)}</pre></div>')
    if status == "skipped":
        reason = (result.get("metadata") or {}).get("skip_reason", "")
        parts.append(
            '<p class="muted small">No video was supplied for this seed, so it was '
            f"not judged.{' ' + e(reason) if reason else ''}</p>"
        )
    else:
        parts.append(_video(seed_id, run_dir))
    parts.append(_trace(seed_id, run_dir, result, budget))
    parts.append(_judge(verdict, index))

    return f"""
<details class="seed"{' open' if open_by_default else ''}>
  <summary>
    <span class="seed-id">{e(seed_id)}</span>
    <span class="tag">{e(result.get('category', ''))}</span>
    <span class="score {score_class(score)}">{fmt(score)}</span>
    {badge}
    <span class="muted small">{duration(elapsed)}</span>
  </summary>
  {''.join(parts)}
</details>"""


def _seed_prompt(result: dict, briefs: dict) -> str:
    """The brief, from the run's own metadata or the dataset it was run against."""
    prompt = (result.get("metadata") or {}).get("prompt")
    if prompt:
        return prompt
    return briefs.get(result.get("seed_id"), "")


def _video(seed_id: str, run_dir: Path) -> str:
    for suffix in VIDEO_SUFFIXES:
        candidate = run_dir / f"{seed_id}{suffix}"
        if candidate.exists():
            size = candidate.stat().st_size / 1024
            return (
                f'<div class="video"><video controls preload="metadata" '
                f'src="{e(candidate.name)}"></video>'
                f'<p class="muted small">{e(candidate.name)} · {size:,.0f} KB</p></div>'
            )
    return '<p class="muted small">No video was produced for this seed.</p>'


def _trace(seed_id: str, run_dir: Path, result: dict, budget: "Budget") -> str:
    seed_dir = run_dir / seed_id
    items = condense_trace(seed_dir / "transcript.jsonl")
    metadata = result.get("metadata") or {}

    stats = []
    if metadata:
        tool_calls = metadata.get("tool_calls") or {}
        stats = [
            ("turns", metadata.get("turns")),
            ("tools", ", ".join(f"{k}×{v}" for k, v in sorted(tool_calls.items())) or "—"),
            ("tokens", (metadata.get("usage") or {}).get("totalTokens")),
            ("outcome", metadata.get("outcome")),
            ("exit", metadata.get("returncode")),
        ]
    # Drop empty stats rather than printing "tokens: 0" for a run that clearly
    # used some — a zero here means "not recorded", not "free".
    stat_line = " · ".join(f"{e(k)}: <b>{e(v)}</b>" for k, v in stats if v)

    if not items:
        inner = '<p class="muted small">No transcript recorded for this seed.</p>'
    else:
        inner = _turns(items, budget)

    return f"""
<details class="trace" open>
  <summary>Agent trace</summary>
  {f'<p class="muted small stats">{stat_line}</p>' if stat_line else ''}
  {inner}
</details>"""


def _turns(items: List[dict], budget: "Budget") -> str:
    """
    Group the trace by turn, with the turn's token count in its heading.

    One turn is one model call plus the tools it asked for, so this is the unit a
    reader reasons about — and the running token count is what shows the context
    filling up, which is the cost that matters on a long generation.
    """
    order: List[int] = []
    grouped = {}
    tokens = {}
    for item in items:
        turn = item.get("turn", 0)
        if turn not in grouped:
            grouped[turn] = []
            order.append(turn)
        if item.get("kind") == "usage":
            tokens[turn] = max(tokens.get(turn, 0), item.get("tokens", 0))
        else:
            grouped[turn].append(item)

    out = []
    for turn in order:
        if not grouped[turn]:
            continue
        count = tokens.get(turn)
        label = f"turn {turn}" + (f" · {count:,} tokens" if count else "")
        out.append(f'<div class="turn-head">{e(label)}</div>')
        out.extend(_trace_item(item, budget) for item in grouped[turn])
    return "".join(out)


def _trace_item(item: dict, budget: "Budget") -> str:
    kind = item.get("kind")

    if kind in ("text", "thinking"):
        flag = ' <span class="flag">interrupted</span>' if item.get("partial") else ""
        if kind == "thinking":
            if item.get("redacted"):
                return '<div class="think"><span class="lbl">thinking</span> (redacted)</div>'
            return (
                f'<div class="think"><span class="lbl">thinking</span>{flag}'
                f'{e(truncate(item["text"], MAX_TEXT_CHARS))}</div>'
            )
        return (
            f'<div class="say">{flag}{e(truncate(item["text"], MAX_TEXT_CHARS))}</div>'
        )
    if kind == "note":
        return f'<p class="muted small">{e(item.get("text"))}</p>'

    args = item.get("args")
    args_text = args if isinstance(args, str) else json.dumps(args, indent=2, ensure_ascii=False)
    output = item.get("output")
    status = "bad" if item.get("error") else ""
    headline = _tool_headline(item.get("name", "?"), args)

    body = f'<pre class="args">{e(truncate(args_text or "", MAX_ARGS_CHARS))}</pre>'
    if output:
        allowance = budget.take(MAX_OUTPUT_CHARS)
        if allowance == 0:
            body += (
                '<p class="muted small">(output omitted — page budget reached; '
                "see transcript.jsonl)</p>"
            )
        else:
            shown = truncate(output, allowance)
            budget.spend(len(shown))
            body += f'<pre class="out">{e(shown)}</pre>'
    elif output == "":
        body += '<p class="muted small">(no output)</p>'
    else:
        body += '<p class="muted small">(no result recorded — the run ended first)</p>'

    return f"""
<details class="tool {status}">
  <summary><code>{e(item.get('name'))}</code>
    <span class="headline">{e(headline)}</span></summary>
  {body}
</details>"""


def _tool_headline(name: str, args) -> str:
    """A one-line gist of the call, so the trace reads without expanding."""
    if not isinstance(args, dict):
        return ""
    for key in ("command", "path", "file_path", "pattern", "query"):
        if key in args:
            return truncate(str(args[key]).replace("\n", " "), 120)
    return truncate(json.dumps(args, ensure_ascii=False), 120)


def _judge(verdict: dict, index: dict) -> str:
    if not verdict:
        return '<p class="muted small">This seed was never judged.</p>'

    sections = {"A": [], "B": [], "C": []}
    for score in verdict.get("scores", []):
        meta = index.get(score.get("criterion"), {})
        sections.get(meta.get("section", "C"), sections["C"]).append((score, meta))

    genre = next(
        (meta.get("title") for _, meta in sections["C"] if meta.get("title")), "Genre-specific"
    )
    blocks = []
    headings = {
        "A": ("Section A — Universal technical baseline", verdict.get("section_a")),
        "B": ("Section B — Semantic &amp; cultural fidelity", verdict.get("section_b")),
        "C": (f"Section C — {e(genre)}", verdict.get("section_c")),
    }
    for key in ("A", "B", "C"):
        rows = sections[key]
        if not rows:
            continue
        title, pct = headings[key]
        blocks.append(
            f'<h4>{title} <span class="score {score_class(pct)}">{fmt(pct)}</span></h4>'
            + _criteria_table(rows)
        )

    safety = verdict.get("safety") or []
    if safety:
        blocks.append("<h4>Section D — Safety</h4>" + _safety_table(safety, index))

    if verdict.get("judge_error"):
        blocks.insert(
            0, f'<div class="warn"><b>Judge issue</b><pre>{e(verdict["judge_error"])}</pre></div>'
        )

    return f"""
<details class="judge" open>
  <summary>Judge results — <span class="score {score_class(verdict.get('total_score'))}">
    {fmt(verdict.get('total_score'))}</span></summary>
  {''.join(blocks)}
</details>"""


def _criteria_table(rows) -> str:
    body = "".join(
        f'<tr class="{"pass" if score.get("passed") else "fail"}">'
        f'<td class="cid">{e(score.get("criterion"))}</td>'
        f'<td>{e(meta.get("name", ""))}</td>'
        f'<td class="num">{e(meta.get("weight", ""))}</td>'
        f'<td class="verdict">{"pass" if score.get("passed") else "fail"}</td>'
        f'<td class="comment">{e(score.get("comment", ""))}</td></tr>'
        for score, meta in rows
    )
    return (
        "<table class='crit'><thead><tr><th>ID</th><th>Criterion</th><th>Weight</th>"
        f"<th>Result</th><th>Judge comment</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _safety_table(safety, index) -> str:
    body = "".join(
        f'<tr class="{"fail" if check.get("violation") else "pass"}">'
        f'<td class="cid">{e(check.get("check_id"))}</td>'
        f'<td>{e(index.get(check.get("check_id"), {}).get("name", ""))}</td>'
        f'<td class="verdict">{"VIOLATION" if check.get("violation") else "clear"}</td>'
        f'<td class="comment">{e(check.get("comment", ""))}</td></tr>'
        for check in safety
    )
    return (
        "<table class='crit'><thead><tr><th>ID</th><th>Check</th><th>Result</th>"
        f"<th>Judge comment</th></tr></thead><tbody>{body}</tbody></table>"
    )


_JS = """
function setAll(open) {
  document.querySelectorAll('details.seed').forEach(d => d.open = open);
}
"""

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --panel: #f9fafb; --code: #f3f4f6;
  --good: #15803d; --mid: #b45309; --bad: #b91c1c;
  --good-bg: #dcfce7; --mid-bg: #fef3c7; --bad-bg: #fee2e2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e6e6; --muted: #9ca3af; --line: #262b36;
    --panel: #161a22; --code: #11141a;
    --good: #4ade80; --mid: #fbbf24; --bad: #f87171;
    --good-bg: #14301f; --mid-bg: #33270c; --bad-bg: #331515;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 6rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 1100px; margin-inline: auto;
}
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .75rem; }
h4 { font-size: .95rem; margin: 1.25rem 0 .4rem; }
p { margin: .35rem 0; }
.muted { color: var(--muted); }
.small { font-size: .82rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }

header { border-bottom: 1px solid var(--line); padding-bottom: 1rem; }
.variant { font-family: ui-monospace, monospace; font-size: .85rem; color: var(--muted);
  word-break: break-all; }
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0; }
.chip { background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
  padding: .15rem .6rem; font-size: .78rem; }
.chip b { color: var(--muted); font-weight: 500; margin-right: .35rem; }
.note { font-style: italic; }

.cards { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1rem 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: .6rem .9rem; min-width: 8rem; }
.card .label { display: block; font-size: .74rem; text-transform: uppercase;
  letter-spacing: .04em; color: var(--muted); }
.card .value { font-size: 1.35rem; font-weight: 600; }

table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .88rem; }
th, td { text-align: left; padding: .38rem .55rem; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { font-weight: 600; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .03em; color: var(--muted); }
.cats { max-width: 34rem; }
.crit .cid { font-family: ui-monospace, monospace; white-space: nowrap; }
.crit .num { text-align: right; width: 4rem; }
.crit .verdict { white-space: nowrap; font-weight: 600; width: 6rem; }
.crit tr.pass .verdict { color: var(--good); }
.crit tr.fail .verdict { color: var(--bad); }
.crit .comment { color: var(--muted); }

.good { color: var(--good); } .mid { color: var(--mid); } .bad { color: var(--bad); }
.score { font-weight: 600; }
.badge { font-size: .74rem; padding: .1rem .5rem; border-radius: 999px; font-weight: 600; }
.badge.good { background: var(--good-bg); color: var(--good); }
.badge.mid  { background: var(--mid-bg);  color: var(--mid); }
.badge.bad  { background: var(--bad-bg);  color: var(--bad); }
.badge.muted { background: var(--code); color: var(--muted); }

.controls { display: flex; gap: .5rem; margin-bottom: .75rem; }
.controls button { font: inherit; font-size: .82rem; padding: .25rem .7rem; cursor: pointer;
  background: var(--panel); color: var(--fg); border: 1px solid var(--line); border-radius: 6px; }

details { border: 1px solid var(--line); border-radius: 8px; margin: .5rem 0;
  background: var(--bg); }
details > summary { cursor: pointer; padding: .55rem .75rem; list-style: none;
  display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸"; color: var(--muted); font-size: .8rem; }
details[open] > summary::before { content: "▾"; }
details > *:not(summary) { padding-inline: .75rem; }
details > *:last-child { padding-bottom: .75rem; }

.seed > summary { background: var(--panel); border-radius: 8px; }
.seed[open] > summary { border-bottom: 1px solid var(--line); border-radius: 8px 8px 0 0; }
.seed-id { font-weight: 600; font-family: ui-monospace, monospace; }
.tag { font-size: .75rem; color: var(--muted); background: var(--code);
  border-radius: 4px; padding: .05rem .4rem; }
.brief { background: var(--panel); border-left: 3px solid var(--line);
  padding: .6rem .8rem; margin: .75rem 0; white-space: pre-wrap; }

.video video { width: 100%; max-width: 640px; border-radius: 8px; background: #000;
  display: block; }

.trace .stats { margin-bottom: .5rem; }
.say { border-left: 3px solid var(--line); padding: .3rem 0 .3rem .7rem; margin: .5rem 0;
  white-space: pre-wrap; }
.think { border-left: 3px solid var(--mid); background: var(--panel); color: var(--muted);
  padding: .4rem .7rem; margin: .5rem 0; white-space: pre-wrap; font-style: italic;
  border-radius: 0 6px 6px 0; }
.think .lbl { display: block; font-style: normal; font-size: .7rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .05em; color: var(--mid); margin-bottom: .2rem; }
.flag { font-style: normal; font-size: .7rem; font-weight: 600; color: var(--bad);
  text-transform: uppercase; letter-spacing: .04em; margin-right: .4rem; }
.turn-head { font-size: .72rem; color: var(--muted); font-family: ui-monospace, monospace;
  text-transform: uppercase; letter-spacing: .05em; margin: .9rem 0 .3rem;
  padding-bottom: .2rem; border-bottom: 1px solid var(--line); }
.tool { background: var(--panel); }
.tool.bad { border-color: var(--bad); }
.tool .headline { color: var(--muted); font-size: .82rem; font-family: ui-monospace, monospace;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
pre { background: var(--code); border-radius: 6px; padding: .6rem .7rem; overflow-x: auto;
  font-size: .8rem; line-height: 1.45; margin: .4rem 0; white-space: pre-wrap;
  word-break: break-word; }
pre.out { border-left: 3px solid var(--line); }
.error, .warn { border-radius: 8px; padding: .6rem .8rem; margin: .75rem 0; }
.error { background: var(--bad-bg); }
.warn { background: var(--mid-bg); }
.error pre, .warn pre { background: transparent; padding: .2rem 0; }
.config pre { max-height: 26rem; overflow: auto; }
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
<script>{script}</script>
</body>
</html>
"""
