"""
PiGenerator: argv construction, the handoff contract, and the process lifecycle.

Fully offline — a stub stands in for `pi` (see fake_pi.py). The lifecycle cases
matter most: they cover the failures that are invisible until a real run hangs.
"""

import json
import os
import signal
import time
from pathlib import Path

import pytest

from video_eval_bench.config import BENCH_TOOLS_EXTENSION
from video_eval_bench.dataset.seed import Seed
from video_eval_bench.generator.pi_generator import (
    PiGenerationError,
    PiGenerator,
    _SeedPaths,
)
from video_eval_bench.judge.frames import video_frame_count


def make_seed(seed_id: str = "marketing_001") -> Seed:
    return Seed(seed_id=seed_id, category="marketing", prompt="A 30-second product ad.")


def paths_for(config, run_dir: Path, seed_id: str = "marketing_001") -> _SeedPaths:
    """The host/agent path pairs for one seed, as __call__ derives them."""
    return _SeedPaths(config, run_dir / seed_id, run_dir / f"{seed_id}.mp4")


def argv_for(config, run_dir: Path) -> list:
    return PiGenerator(config).build_argv(make_seed(), paths_for(config, run_dir))


def task_for(config, run_dir: Path) -> str:
    return PiGenerator(config).render_task(make_seed(), paths_for(config, run_dir))


# ── argv ──────────────────────────────────────────────────────────────────────


def test_argv_carries_model_and_mode(pi_config, tmp_path):
    argv = argv_for(pi_config(), tmp_path)
    assert argv[1:5] == ["--provider", "gx10", "--model", "test-model"]
    assert "--print" in argv and "--no-session" in argv
    assert argv[argv.index("--mode") + 1] == "json"


def test_argv_always_loads_the_handoff_extension(pi_config, tmp_path):
    """Without submit_video the agent has no way to deliver a result at all."""
    argv = argv_for(pi_config(), tmp_path)
    assert str(BENCH_TOOLS_EXTENSION) in argv
    assert argv[argv.index(str(BENCH_TOOLS_EXTENSION)) - 1] == "-e"


def test_argv_tools_and_discovery_flags(pi_config, tmp_path):
    argv = argv_for(pi_config(), tmp_path)
    assert "--no-skills" in argv  # discovery off by default
    assert "--no-extensions" in argv
    assert "--no-context-files" in argv


def test_tool_allowlist_keeps_the_handoff_tool(pi_config, tmp_path):
    """
    pi's --tools allowlist covers extension tools too. Leaving submit_video out
    of it lets the agent generate a video and then discover it has no way to
    deliver it — which is exactly what happened before this was fixed.
    """
    argv = argv_for(pi_config(), tmp_path)
    allowed = argv[argv.index("--tools") + 1].split(",")
    assert allowed[:4] == ["read", "write", "edit", "bash"]
    assert "submit_video" in allowed


def test_extension_tools_reach_the_allowlist(pi_config, tmp_path):
    """
    An extension tool missing from --tools is loaded and then filtered out. On a
    real run the agent spent nine turns hunting for a generator that was loaded
    the whole time, then read the grading rubric instead.
    """
    ext = tmp_path / "video.ts"
    ext.write_text("// stub")
    argv = argv_for(
        pi_config(extensions=[ext], extension_tools=["generate_video"]), tmp_path
    )

    allowed = argv[argv.index("--tools") + 1].split(",")
    assert "generate_video" in allowed
    assert "submit_video" in allowed
    assert len(allowed) == len(set(allowed)), "no duplicates in the allowlist"


def test_no_builtin_tools_still_leaves_the_handoff_tool(pi_config, tmp_path):
    """--no-builtin-tools keeps extension tools, so no allowlist is needed."""
    from video_eval_bench.config import ToolsConfig

    argv = argv_for(pi_config(tools=ToolsConfig(no_builtin=True)), tmp_path)
    assert "--no-builtin-tools" in argv
    assert "--tools" not in argv


def test_argv_lists_each_configured_skill(pi_config, tmp_path):
    from video_eval_bench.config import SkillsConfig

    skill = tmp_path / "a-skill"
    skill.mkdir()
    argv = argv_for(pi_config(skills=SkillsConfig(paths=[skill])), tmp_path)
    assert argv[argv.index("--skill") + 1] == str(skill.resolve())


def test_task_prompt_carries_the_brief(pi_config, tmp_path):
    task = task_for(pi_config(), tmp_path)
    assert "A 30-second product ad." in task
    assert "marketing" in task
    assert str(tmp_path) in task


def test_bad_template_placeholder_names_itself(pi_config, tmp_path):
    """A shell snippet in a custom template must not fail every seed with a bare KeyError."""
    from video_eval_bench.config import PromptConfig

    config = pi_config(
        system_prompt=PromptConfig(system_text="s", task_text="run ${HOME}/x {prompt}")
    )
    with pytest.raises(PiGenerationError, match="Literal braces must be doubled"):
        PiGenerator(config).render_task(make_seed(), paths_for(config, tmp_path))


# ── happy path ────────────────────────────────────────────────────────────────


async def test_produces_a_readable_video(pi_config, mode, tmp_path):
    mode("submit")
    generator = PiGenerator(pi_config())
    result = await generator(make_seed(), tmp_path)
    path = result.video_path

    assert Path(path) == tmp_path / "marketing_001.mp4"
    assert video_frame_count(path) > 0


async def test_writes_transcript_and_run_metadata(pi_config, mode, tmp_path):
    mode("submit")
    generator = PiGenerator(pi_config())
    result = await generator(make_seed(), tmp_path)

    seed_dir = tmp_path / "marketing_001"
    transcript = (seed_dir / "transcript.jsonl").read_text()
    assert '"agent_start"' in transcript
    assert '"submit_video"' in transcript

    run_json = json.loads((seed_dir / "pi_run.json").read_text())
    assert run_json["outcome"] == "exited"
    assert run_json["returncode"] == 0
    assert run_json["tool_calls"]["submit_video"] == 1
    assert run_json["usage"]["input"] == 1200
    # The metadata comes back bound to the result, not fetched afterwards by id.
    assert result.seed_id == "marketing_001"
    assert result.metadata == run_json


def test_zero_usage_does_not_erase_the_token_count():
    """
    pi emits an all-zeros usage block on message_start. Accepting it reports a
    60-minute run as having used no tokens — observed on a real run.
    """
    from video_eval_bench.generator.pi_generator import _RunState

    state = _RunState()
    state.observe({"type": "message_end", "usage": {"totalTokens": 12910, "input": 12000}})
    state.observe({"type": "message_start", "usage": {"totalTokens": 0, "input": 0}})
    assert state.usage["totalTokens"] == 12910


# ── failure paths ─────────────────────────────────────────────────────────────


async def test_no_submission_is_a_clear_error(pi_config, mode, tmp_path):
    mode("no_submit")
    with pytest.raises(PiGenerationError, match="submit_video"):
        await PiGenerator(pi_config())(make_seed(), tmp_path)


async def test_nonzero_exit_carries_the_stderr_tail(pi_config, mode, tmp_path):
    mode("fail")
    with pytest.raises(PiGenerationError, match="fake pi exploded|exited with 3"):
        await PiGenerator(pi_config())(make_seed(), tmp_path)
    assert "fake pi exploded" in (tmp_path / "marketing_001" / "pi_stderr.log").read_text()


async def test_empty_file_is_rejected(pi_config, mode, tmp_path):
    """A submission event is not enough — the file has to be a usable video."""
    mode("empty_file")
    with pytest.raises(PiGenerationError, match="not a readable video"):
        await PiGenerator(pi_config())(make_seed(), tmp_path)


async def test_file_without_submission_is_rejected(pi_config, mode, tmp_path):
    """
    A video at the output path with no submit event must not count: otherwise a
    stale file from an earlier attempt reads as a fresh result.
    """
    mode("file_only")
    with pytest.raises(PiGenerationError, match="submit_video"):
        await PiGenerator(pi_config())(make_seed(), tmp_path)


async def test_timeout_kills_the_run(pi_config, mode, tmp_path):
    mode("hang")
    started = time.monotonic()
    with pytest.raises(PiGenerationError, match="budget"):
        await PiGenerator(pi_config(timeout_seconds=1.5))(make_seed(), tmp_path)
    assert time.monotonic() - started < 15  # killed, not waited out


# ── lifecycle ─────────────────────────────────────────────────────────────────


async def test_submission_then_nonzero_exit_is_still_a_result(pi_config, mode, tmp_path):
    """The video exists and the agent claimed it; the exit code is an anomaly."""
    mode("submit_then_fail")
    path = (await PiGenerator(pi_config())(make_seed(), tmp_path)).video_path

    assert video_frame_count(path) > 0
    run_json = json.loads((tmp_path / "marketing_001" / "pi_run.json").read_text())
    assert run_json["returncode"] == 3


async def test_work_continues_after_submitting(pi_config, mode, tmp_path):
    """
    Submitting must not end the run.

    The agent is told it can bank a result and keep improving it, so a model that
    is still working after a submission is behaving correctly and must be left
    alone until it exits or the budget runs out. Killing it here is what made
    submission all-or-nothing, and cost a real run an hour of work it never
    delivered.
    """
    mode("linger", FAKE_PI_LINGER=30)
    started = time.monotonic()
    result = await PiGenerator(pi_config(timeout_seconds=3))(make_seed(), tmp_path)
    elapsed = time.monotonic() - started

    # The budget, not the submission, is what stopped it.
    assert elapsed >= 3, f"stopped after {elapsed:.1f}s — the submission cut it short"
    assert video_frame_count(result.video_path) > 0
    run_json = json.loads((tmp_path / "marketing_001" / "pi_run.json").read_text())
    assert run_json["outcome"] == "timeout_after_submit"


async def test_the_last_submission_wins(pi_config, mode, tmp_path):
    """
    Re-submission is the whole point of letting the run continue: the agent banks
    something scoreable early, then replaces it as it improves. The harness must
    report the newest submission, not the first one it saw.
    """
    mode("resubmit")
    result = await PiGenerator(pi_config(timeout_seconds=30))(make_seed(), tmp_path)

    # The fake's second video is twice as long as its first.
    assert video_frame_count(result.video_path) == 24
    run_json = json.loads((tmp_path / "marketing_001" / "pi_run.json").read_text())
    assert run_json["tool_calls"]["submit_video"] == 2
    # submitted_path is the pinned destination, the same for both submissions —
    # which is why the frame count above is what proves the second one landed.
    assert run_json["submitted_path"] == str(result.video_path)


async def test_budget_expiry_with_a_submission_is_not_a_failure(pi_config, mode, tmp_path):
    """
    Running out of time having already delivered is a success. Before, the banked
    video was thrown away and the seed reported a bare generation error.
    """
    mode("linger", FAKE_PI_LINGER=30)
    result = await PiGenerator(pi_config(timeout_seconds=2))(make_seed(), tmp_path)

    assert video_frame_count(result.video_path) > 0
    assert result.metadata["outcome"] == "timeout_after_submit"


async def test_stderr_flood_does_not_deadlock(pi_config, mode, tmp_path):
    """
    Regression test for the pipe-buffer hang: draining stdout while stderr fills
    its ~64KB buffer blocks pi on write forever. Fails on any implementation
    that pipes stderr and reads it after stdout.
    """
    mode("stderr_flood")
    path = (
        await PiGenerator(pi_config(timeout_seconds=30))(make_seed(), tmp_path)
    ).video_path

    assert video_frame_count(path) > 0
    assert (tmp_path / "marketing_001" / "pi_stderr.log").stat().st_size > 1_000_000


async def test_long_silence_is_not_a_failure(pi_config, mode, tmp_path, caplog):
    """
    pi puts no timeout on bash, so a real generation can be silent for many
    minutes. Guards against anyone adding an inactivity timeout, which would
    kill exactly the runs this benchmark exists to measure.
    """
    mode("silent", FAKE_PI_SILENCE=2.0)
    with caplog.at_level("INFO"):
        path = (
            await PiGenerator(pi_config(heartbeat_seconds=0.4, timeout_seconds=30))(
                make_seed(), tmp_path
            )
        ).video_path

    assert video_frame_count(path) > 0
    assert any("elapsed" in record.message for record in caplog.records)


async def test_timeout_leaves_no_orphan_processes(pi_config, mode, tmp_path):
    """
    Killing pi alone would orphan whatever its bash tool was running and leak it
    into every later seed. The signal must reach the process group.
    """
    mode("hang")
    generator = PiGenerator(pi_config(timeout_seconds=1.5))
    with pytest.raises(PiGenerationError):
        await generator(make_seed(), tmp_path)

    # The transcript names the pid's group; check nothing from it survives.
    run_json = json.loads((tmp_path / "marketing_001" / "pi_run.json").read_text())
    assert run_json["outcome"] == "timeout"
    assert not _surviving_children(os.getpid())


def _surviving_children(parent_pid: int) -> list:
    """Any of our own children still alive (the fake pi should be reaped)."""
    survivors = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        if f"PPid:\t{parent_pid}\n" in status and "fake_pi" in _cmdline(entry):
            survivors.append(entry.name)
    return survivors


def _cmdline(proc_dir: Path) -> str:
    try:
        return proc_dir.joinpath("cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return ""


# ── through the bench loop ────────────────────────────────────────────────────


async def test_run_bench_records_generator_metadata(pi_config, mode, tmp_path):
    from video_eval_bench.bench import run_bench
    from video_eval_bench.dataset import load_dataset
    from video_eval_bench.judge.agent import VideoJudge
    from video_eval_bench.judge.llm import MockBackend

    mode("submit")
    dataset = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=dataset, n_frames=4),
        generate=PiGenerator(pi_config()),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=dataset,
        category="marketing",
    )

    summary = report.summary()
    assert summary["n_generation_errors"] == 0
    assert summary["n_ok"] == 1
    metadata = report.to_json()["results"][0]["metadata"]
    assert metadata["outcome"] == "exited"
    assert Path(metadata["transcript"]).exists()


async def test_generation_failure_does_not_abort_the_bench(pi_config, mode, tmp_path):
    from video_eval_bench.bench import run_bench
    from video_eval_bench.dataset import load_dataset
    from video_eval_bench.judge.agent import VideoJudge
    from video_eval_bench.judge.llm import MockBackend

    mode("no_submit")
    dataset = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=dataset, n_frames=4),
        generate=PiGenerator(pi_config()),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=dataset,
        category="entertainment",  # two seeds: both must be attempted
    )

    assert report.summary()["n_seeds"] == 2
    assert report.summary()["n_generation_errors"] == 2
    assert all(r.status == "errored" for r in report.results)

    # A failed run is the one whose turn count and token usage are most worth
    # having: it is the expensive one, and the only clue to why it failed. The
    # metadata now rides out on the exception rather than being stashed for the
    # bench to fetch, so this is the regression guard for that path.
    for result in report.results:
        assert result.metadata["outcome"] == "exited"
        assert result.metadata["turns"] >= 1
        assert Path(result.metadata["transcript"]).exists()


async def test_a_timed_out_run_still_reports_what_it_burned(pi_config, mode, tmp_path):
    """The most expensive failure there is; reporting it as a blank hides the cost."""
    from video_eval_bench.bench import run_bench
    from video_eval_bench.dataset import load_dataset
    from video_eval_bench.judge.agent import VideoJudge
    from video_eval_bench.judge.llm import MockBackend

    mode("hang")
    dataset = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=dataset, n_frames=4),
        generate=PiGenerator(pi_config(timeout_seconds=1.5)),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=dataset,
        category="marketing",
    )

    (result,) = report.results
    assert result.status == "errored"
    assert "budget" in result.generation_error
    assert result.metadata["outcome"] == "timeout"
