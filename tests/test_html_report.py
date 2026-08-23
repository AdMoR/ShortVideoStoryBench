"""
The HTML report: condensing a trace, and rendering a run to one readable page.

The page has to open from file:// with no network and stay small, so the tests
check both what it contains and what it deliberately leaves out.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from video_eval_bench.dataset import load_dataset
from video_eval_bench.report.html import (
    MAX_OUTPUT_CHARS,
    condense_trace,
    render_run,
    rubric_index,
    truncate,
)


class _Wellformed(HTMLParser):
    """Enough of a check to catch an unclosed tag or a broken nesting."""

    VOID = {"br", "img", "meta", "link", "hr", "input", "source"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closed while <{self.stack[-1]}> was open")
        else:
            self.stack.pop()


def check_wellformed(markup: str) -> None:
    parser = _Wellformed()
    parser.feed(markup)
    assert not parser.errors, parser.errors
    assert not parser.stack, f"unclosed: {parser.stack}"


# ── trace condensation ────────────────────────────────────────────────────────


def write_transcript(path: Path, events) -> Path:
    path.write_text("\n".join(json.dumps(e) for e in events))
    return path


def test_streaming_deltas_are_dropped(tmp_path):
    """message_update is the same text arriving one token at a time — and 95% of the file."""
    events = [{"type": "turn_start"}]
    events += [
        {"type": "message_update", "assistantMessageEvent": {"delta": "x"}} for _ in range(400)
    ]
    events.append(
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        }
    )
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert [i["kind"] for i in items] == ["text"]
    assert items[0]["text"] == "done"


def test_reasoning_is_captured(tmp_path):
    """
    pi puts reasoning in a `thinking` part whose text lives under a `thinking`
    key, not `text`. Reading only `text` drops a reasoning model's entire train
    of thought — and the report would look identical to a model that never
    reasoned at all.
    """
    events = [
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "The brief needs three scenes. I should check ffmpeg first."},
            {"type": "text", "text": "Checking the toolchain."},
        ]}},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert [i["kind"] for i in items] == ["thinking", "text"]
    assert "three scenes" in items[0]["text"]


def test_redacted_reasoning_is_marked_not_dropped(tmp_path):
    events = [
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "redacted": True},
        ]}},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert items[0]["kind"] == "thinking" and items[0]["redacted"] is True


def test_interrupted_message_is_recovered_from_deltas(tmp_path):
    """
    A run killed mid-message never emits message_end. Dropping every delta would
    lose the last thing the model said — which is the moment worth seeing when a
    generation times out.
    """
    events = [
        {"type": "turn_start"},
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_update", "assistantMessageEvent":
            {"type": "thinking_delta", "contentIndex": 0, "delta": "I am still "}},
        {"type": "message_update", "assistantMessageEvent":
            {"type": "thinking_delta", "contentIndex": 0, "delta": "deciding how to"}},
        {"type": "message_update", "assistantMessageEvent":
            {"type": "text_delta", "contentIndex": 1, "delta": "Let me render "}},
        # killed here: no message_end
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    kinds = {i["kind"]: i for i in items}
    assert kinds["thinking"]["text"] == "I am still deciding how to"
    assert kinds["thinking"]["partial"] is True
    assert kinds["text"]["text"] == "Let me render"


def test_completed_message_supersedes_its_deltas(tmp_path):
    """The finished message is authoritative — deltas must not double up."""
    events = [
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {"type": "message_update", "assistantMessageEvent":
            {"type": "text_delta", "contentIndex": 0, "delta": "partial tex"}},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "partial text, completed"}]}},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert [i["text"] for i in items] == ["partial text, completed"]


def test_tool_calls_pair_with_their_results(tmp_path):
    events = [
        {"type": "turn_start"},
        {"type": "tool_execution_start", "toolCallId": "1", "toolName": "bash",
         "args": {"command": "ls -la"}},
        {"type": "tool_execution_end", "toolCallId": "1", "toolName": "bash",
         "result": {"content": [{"type": "text", "text": "total 0"}]}, "isError": False},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert len(items) == 1
    assert items[0]["name"] == "bash"
    assert items[0]["args"]["command"] == "ls -la"
    assert items[0]["output"] == "total 0"
    assert items[0]["error"] is False


def test_image_results_are_noted_not_inlined(tmp_path):
    """
    When the agent reads a frame, pi returns base64 image data. That it looked
    matters; the bytes would add tens of KB per call to the page.
    """
    blob = "A" * 40_000
    events = [
        {"type": "tool_execution_start", "toolCallId": "1", "toolName": "read",
         "args": {"path": "frame.jpg"}},
        {"type": "tool_execution_end", "toolCallId": "1", "toolName": "read",
         "result": {"content": [
             {"type": "text", "text": "Read image file [image/jpeg]"},
             {"type": "image", "data": blob, "mimeType": "image/jpeg"},
         ]}, "isError": False},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    output = items[0]["output"]
    assert "Read image file" in output
    assert "image/jpeg · 29 KB" in output
    assert blob[:100] not in output


def test_failed_tool_call_is_marked(tmp_path):
    events = [
        {"type": "tool_execution_start", "toolCallId": "1", "toolName": "submit_video", "args": {}},
        {"type": "tool_execution_end", "toolCallId": "1", "toolName": "submit_video",
         "result": {"content": [{"type": "text", "text": "rejected"}]}, "isError": True},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert items[0]["error"] is True


def test_unfinished_tool_call_survives(tmp_path):
    """A run killed mid-call still has to render — output stays None, not a crash."""
    events = [
        {"type": "tool_execution_start", "toolCallId": "1", "toolName": "bash",
         "args": {"command": "sleep 999"}},
    ]
    items = condense_trace(write_transcript(tmp_path / "t.jsonl", events))
    assert items[0]["output"] is None


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"type": "turn_start"}\nnot json at all\n{"broken\n')
    assert condense_trace(path) == []


def test_missing_transcript_is_not_an_error(tmp_path):
    assert condense_trace(tmp_path / "nope.jsonl") == []


def test_truncate_reports_what_it_hid():
    out = truncate("x" * 5000, 100)
    assert len(out) < 200
    assert "4,900 more characters" in out


# ── rubric index ──────────────────────────────────────────────────────────────


def test_rubric_index_assigns_sections():
    index = rubric_index(load_dataset())
    assert index["U1"]["section"] == "A"
    assert index["S1"]["section"] == "B"
    assert index["D1"]["section"] == "D"
    assert index["U1"]["weight"] > 0
    assert index["U1"]["name"]


def test_rubric_index_scopes_section_c_to_one_genre():
    """
    Genre criterion ids only happen to be unique across genres; nothing enforces
    it. Scoping per seed means a future reused id cannot mislabel a criterion.
    """
    dataset = load_dataset()
    index = rubric_index(dataset, "marketing")
    assert index["M1"]["section"] == "C"
    assert index["M1"]["title"] == dataset.categories["marketing"].name
    assert "G1" not in index  # gaming's criteria are not in scope here


def test_section_c_heading_names_the_genre(rendered):
    assert "Section C — " in rendered()


# ── whole-page rendering ──────────────────────────────────────────────────────


@pytest.fixture
def rendered(tmp_path):
    """A run directory with one judged seed, one failed seed, and a video."""

    def build(**overrides):
        seed_dir = tmp_path / "marketing_001"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_transcript(
            seed_dir / "transcript.jsonl",
            [
                {"type": "turn_start"},
                {"type": "tool_execution_start", "toolCallId": "1", "toolName": "bash",
                 "args": {"command": "ffmpeg -i in.mp4 out.mp4"}},
                {"type": "tool_execution_end", "toolCallId": "1", "toolName": "bash",
                 "result": {"content": [{"type": "text", "text": "z" * 9000}]},
                 "isError": False},
                {"type": "message_end", "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Rendered the ad."}],
                    "usage": {"totalTokens": 4109}}},
            ],
        )
        (tmp_path / "marketing_001.mp4").write_bytes(b"\x00" * 2048)

        report = {
            "summary": {
                "run_id": "20260822-193000", "variant": "skills=none",
                "n_seeds": 2, "n_ok": 1, "n_generation_errors": 1, "n_judge_errors": 0,
                "mean_score": 61.5, "n_passed": 1, "n_safety_vetoes": 0,
                "total_duration_seconds": 214,
                "per_category": {"marketing": {"n_seeds": 1, "mean_score": 61.5,
                                               "n_passed": 1, "n_safety_vetoes": 0}},
            },
            "variant": "skills=none,tools=full",
            "choices": {"skills": "none", "tools": "full"},
            "note": "baseline",
            "started_at": "2026-08-22T19:30:00+00:00",
            "finished_at": "2026-08-22T19:33:34+00:00",
            "config": {"generator": {"kind": "pi"}},
            "results": [
                {
                    "seed_id": "marketing_001", "category": "marketing",
                    "video_path": str(tmp_path / "marketing_001.mp4"),
                    "generation_error": None, "duration_seconds": 180.0,
                    "verdict": {
                        "seed_id": "marketing_001", "category": "marketing",
                        "section_a": 66.7, "section_b": 50.0, "section_c": 67.8,
                        "total_score": 61.5,
                        "scores": [
                            {"criterion": "U1", "passed": True, "score": 3.0,
                             "comment": "Stable framing throughout."},
                            {"criterion": "S1", "passed": False, "score": 0.0,
                             "comment": "The <b>logo</b> never appears."},
                            {"criterion": "M1", "passed": True, "score": 15.0,
                             "comment": "Brand mark is legible in the final frame."},
                        ],
                        "safety": [{"check_id": "D1", "violation": False, "comment": "clear"}],
                        "safety_veto": False, "passed": True, "reasoning": "ok",
                        "judge_error": None,
                    },
                    "metadata": {"turns": 4, "tool_calls": {"bash": 3},
                                 "usage": {"totalTokens": 4109}, "outcome": "exited",
                                 "returncode": 0},
                },
                {
                    "seed_id": "gaming_001", "category": "gaming", "video_path": None,
                    "generation_error": "the agent finished without calling submit_video",
                    "duration_seconds": 34.0, "verdict": None, "metadata": {},
                },
            ],
        }
        report.update(overrides)
        (tmp_path / "report.json").write_text(json.dumps(report))
        return render_run(tmp_path).read_text()

    return build


def test_page_is_wellformed_and_self_contained(rendered):
    markup = rendered()
    check_wellformed(markup)
    # No network: everything inline, and the only src is the local video.
    assert "http://" not in markup and "https://" not in markup
    assert "<script src" not in markup and "<link" not in markup


def test_video_is_referenced_not_embedded(rendered):
    """Base64 would multiply the page size by the size of the videos."""
    markup = rendered()
    assert '<video controls preload="metadata" src="marketing_001.mp4">' in markup
    assert "data:video" not in markup and "base64" not in markup


def test_seed_sections_are_collapsible(rendered):
    markup = rendered()
    assert markup.count('<details class="seed">') == 2
    assert "marketing_001" in markup and "gaming_001" in markup


def test_single_seed_run_opens_expanded(rendered, tmp_path):
    """A smoke run or an ablation probe has nothing to scan past."""
    assert '<details class="seed">' in rendered()  # two seeds: collapsed

    data = json.loads((tmp_path / "report.json").read_text())
    data["results"] = data["results"][:1]
    (tmp_path / "report.json").write_text(json.dumps(data))
    single = render_run(tmp_path).read_text()
    assert '<details class="seed" open>' in single


def test_judge_results_render_every_criterion(rendered):
    markup = rendered()
    assert "U1" in markup and "S1" in markup
    assert "Stable framing throughout." in markup
    assert "Section A" in markup and "Section D" in markup
    assert "pass" in markup and "fail" in markup


def test_reasoning_renders_distinctly_from_prose(tmp_path):
    """Reasoning is the model's private working; it should not read as its answer."""
    seed_dir = tmp_path / "marketing_001"
    seed_dir.mkdir(parents=True)
    write_transcript(seed_dir / "transcript.jsonl", [
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Weighing ffmpeg against the API."},
            {"type": "text", "text": "I will use ffmpeg."},
        ]}},
    ])
    (tmp_path / "report.json").write_text(json.dumps({
        "summary": {"run_id": "r", "n_seeds": 1, "per_category": {}},
        "results": [{"seed_id": "marketing_001", "category": "marketing",
                     "generation_error": None, "verdict": None, "metadata": {}}],
    }))
    markup = render_run(tmp_path).read_text()
    assert 'class="think"' in markup
    assert "Weighing ffmpeg against the API." in markup
    assert 'class="say"' in markup
    check_wellformed(markup)


def test_judge_comments_are_escaped(rendered):
    """Judge comments are model output — they must never become live markup."""
    markup = rendered()
    assert "&lt;b&gt;logo&lt;/b&gt;" in markup
    assert "The <b>logo</b> never appears" not in markup


def test_failed_seed_shows_the_error_and_no_video(rendered):
    markup = rendered()
    assert "the agent finished without calling submit_video" in markup
    assert "No video was produced for this seed." in markup


def test_failed_seed_duration_falls_back_to_generator_metadata(tmp_path, rendered):
    """
    Reports written before run_bench timed the failure path show an hour-long
    timeout as free. The generator's own metadata still has the truth.
    """
    rendered()
    data = json.loads((tmp_path / "report.json").read_text())
    failed = data["results"][1]
    failed["duration_seconds"] = 0.0
    failed["metadata"] = {"duration_seconds": 3600.0, "outcome": "timeout"}
    (tmp_path / "report.json").write_text(json.dumps(data))

    assert "60m00s" in render_run(tmp_path).read_text()


def test_long_tool_output_is_truncated(rendered):
    markup = rendered()
    assert "z" * (MAX_OUTPUT_CHARS + 50) not in markup
    assert "more characters]" in markup


def test_trace_is_grouped_into_turns_with_token_counts(rendered):
    """
    One turn is one model call plus the tools it asked for, and the running token
    count is what shows the context filling up on a long generation.
    """
    markup = rendered()
    assert "turn 1 · 4,109 tokens" in markup


def test_page_stays_small(rendered):
    """A report nobody can open is not a report."""
    assert len(rendered().encode()) < 120_000


def test_page_wide_budget_bounds_a_chatty_run(tmp_path, monkeypatch):
    """
    Per-seed caps bound a seed, not the page. A run full of noisy tool output
    must degrade to summarised traces with a visible note, not a report too
    heavy to open.
    """
    from video_eval_bench.report import html as html_mod

    monkeypatch.setattr(html_mod, "TOTAL_OUTPUT_BUDGET", 5_000)

    seed_dir = tmp_path / "marketing_001"
    seed_dir.mkdir(parents=True)
    events = []
    for i in range(40):
        events += [
            {"type": "tool_execution_start", "toolCallId": str(i), "toolName": "bash",
             "args": {"command": f"step {i}"}},
            {"type": "tool_execution_end", "toolCallId": str(i), "toolName": "bash",
             "result": {"content": [{"type": "text", "text": "y" * 4000}]}, "isError": False},
        ]
    write_transcript(seed_dir / "transcript.jsonl", events)
    (tmp_path / "report.json").write_text(json.dumps({
        "summary": {"run_id": "r", "n_seeds": 1, "per_category": {}},
        "results": [{"seed_id": "marketing_001", "category": "marketing",
                     "generation_error": None, "verdict": None, "metadata": {}}],
    }))

    markup = render_run(tmp_path).read_text()
    assert "page budget reached" in markup
    assert len(markup.encode()) < 60_000


def test_render_needs_a_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="report.json"):
        render_run(tmp_path)


# ── skipped seeds ─────────────────────────────────────────────────────────────


def test_a_skipped_seed_is_not_shown_as_a_failing_one(rendered, tmp_path):
    """
    A skipped seed has no error and no verdict, so without a branch of its own it
    falls through the badge cascade to "below threshold" and reads as a seed that
    was judged and scored badly. It was never judged at all.
    """
    page = rendered(
        summary={
            "run_id": "r", "variant": "", "n_seeds": 2, "n_ok": 1,
            "n_generation_errors": 0, "n_judge_errors": 0, "n_skipped": 1,
            "mean_score": 61.5, "n_passed": 1, "n_safety_vetoes": 0,
            "total_duration_seconds": 180,
            "per_category": {},
        },
        results=[
            {
                "seed_id": "gaming_001", "category": "gaming", "status": "skipped",
                "video_path": None, "generation_error": None, "duration_seconds": 0.0,
                "verdict": None,
                "metadata": {"skip_reason": "no entry for 'gaming_001' in videos.yaml"},
            }
        ],
    )

    assert ">skipped<" in page
    assert "below threshold" not in page
    assert "No video was supplied" in page
    assert "no entry for" in page


def test_the_skipped_count_is_on_the_front_page(rendered):
    page = rendered(
        summary={
            "run_id": "r", "variant": "", "n_seeds": 8, "n_ok": 1,
            "n_generation_errors": 0, "n_judge_errors": 0, "n_skipped": 7,
            "mean_score": 61.5, "n_passed": 1, "n_safety_vetoes": 0,
            "total_duration_seconds": 180, "per_category": {},
        },
    )
    assert "Skipped" in page


def test_a_report_written_before_status_existed_still_renders(rendered):
    """No result carries a status; the old runs in runs/ must not stop rendering."""
    page = rendered()
    assert "<html" in page
    assert "generation failed" in page  # inferred, not read off a field
    assert "passed" in page
