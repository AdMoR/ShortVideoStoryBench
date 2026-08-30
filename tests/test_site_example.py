"""
Exporting one run as the site's worked example.

The export is the only thing standing between a 150 MB run directory and a git
repository, so the tests here are about what it refuses to carry: an untrimmed
transcript, a video at generation bitrate, and media left behind by a previous
example that nothing references any more.
"""

import json
from pathlib import Path

import pytest

from video_eval_bench.report import example


def _events(n_tools: int = 2, output: str = "ok") -> str:
    """A minimal pi NDJSON stream: one turn, some thinking, and tool calls."""
    lines = [
        {"type": "turn_start"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "T" * 4000}],
                "usage": {"totalTokens": 2750},
            },
        },
    ]
    for i in range(n_tools):
        lines += [
            {
                "type": "tool_execution_start",
                "toolCallId": str(i),
                "toolName": "bash",
                "args": {"command": f"echo {i}\nand a second line"},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": str(i),
                "result": {"content": [{"type": "text", "text": output}]},
                "isError": False,
            },
        ]
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def test_the_spread_is_shown_not_the_highlight_reel():
    """Best, median and worst. Three clips that all scored 70 teach nothing."""
    results = [
        {"seed_id": f"s{i}", "verdict": {"total_score": score}}
        for i, score in enumerate([10.0, 30.0, 50.0, 70.0, 90.0])
    ]
    assert example.pick_seeds(results) == ["s4", "s2", "s0"]


def test_seeds_that_were_never_judged_are_not_candidates():
    """A generation that died has no score to place in the spread."""
    results = [
        {"seed_id": "ok", "verdict": {"total_score": 40.0}},
        {"seed_id": "errored", "verdict": {}},
        {"seed_id": "no_verdict"},
    ]
    assert example.pick_seeds(results) == ["ok"]


def test_the_trace_is_trimmed_to_what_a_web_page_should_carry(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(_events(output="O" * 9000))
    items = example.web_trace(transcript)

    thinking = next(i for i in items if i["kind"] == "thinking")
    assert len(thinking["text"]) <= example.WEB_MAX_TEXT
    for tool in (i for i in items if i["kind"] == "tool"):
        assert len(tool["output"]) <= example.WEB_MAX_OUTPUT
        assert len(tool["args"]) <= example.WEB_MAX_ARGS


def test_a_long_run_is_cut_off_and_says_so(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(_events(n_tools=example.WEB_MAX_ITEMS + 50))
    items = example.web_trace(transcript)
    assert len(items) == example.WEB_MAX_ITEMS + 1
    assert "truncated" in items[-1]["text"]


def test_a_tool_call_carries_the_gist_of_what_it_ran(tmp_path):
    """
    The collapsed trace shows one line per call. That line has to be the command,
    not the JSON envelope around it.
    """
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(_events(n_tools=1))
    tool = next(i for i in example.web_trace(transcript) if i["kind"] == "tool")
    assert tool["head"].startswith("echo 0")
    assert "\n" not in tool["head"]


def test_a_missing_transcript_is_not_fatal(tmp_path):
    """An external generator produced the clip without an agent. There is no trace."""
    assert example.web_trace(tmp_path / "absent.jsonl") == []


def test_stale_media_from_a_previous_example_is_removed(tmp_path):
    """
    Re-exporting from another run must not leave the old run's clips in the
    repository — committed, deployed, and reachable by anyone who kept a link.
    """
    media = tmp_path / "media"
    media.mkdir()
    keep = media / "kept.mp4"
    keep.write_bytes(b"x")
    (media / "old.mp4").write_bytes(b"x")
    (media / "old.jpg").write_bytes(b"x")

    removed = example.prune(media, [keep])

    assert {p.name for p in removed} == {"old.mp4", "old.jpg"}
    assert [p.name for p in sorted(media.iterdir())] == ["kept.mp4"]


def test_pruning_an_absent_directory_is_a_no_op(tmp_path):
    assert example.prune(tmp_path / "never", []) == []


def test_the_arm_prefers_the_hydra_group_name_over_the_prompt_filename():
    """
    `system_prompt=director` is what a reader would type to reproduce the run;
    `director_system.md` is only where it happened to live on that machine.
    """
    config = {
        "generator": {"system_prompt": {"system_file": "/x/director_system.md"}},
        "judge": {},
    }
    assert example._params(config, {"system_prompt": "director"})["generator"][
        "system_prompt"
    ] == "director"
    assert example._params(config, {})["generator"]["system_prompt"] == "director_system"


@pytest.mark.skipif(
    not example.shutil.which("ffmpeg"), reason="ffmpeg is not installed"
)
def test_a_clip_is_re_encoded_small_enough_to_commit(tmp_path):
    """
    A run's mp4 is at generation bitrate — megabytes for five seconds. What goes into
    the repository has to fit a 640px box, and the page needs its dimensions to
    caption it honestly.
    """
    source = tmp_path / "big.mp4"
    example._ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=2",
        "-c:v", "libx264", "-crf", "5", "-pix_fmt", "yuv420p", str(source),
    )
    media = tmp_path / "media"
    media.mkdir()

    info = example.transcode(source, media, "seed_001")

    assert (media / "seed_001.mp4").exists() and (media / "seed_001.jpg").exists()
    assert max(info["width"], info["height"]) == example.MEDIA_BOX
    assert info["width"] % 2 == 0 and info["height"] % 2 == 0  # libx264 at yuv420p
    assert info["bytes"] < info["source_bytes"]
    assert info["has_audio"] is False
