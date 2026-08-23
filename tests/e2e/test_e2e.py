"""
End-to-end: the real pi agent, the real model, a stand-in video backend.

No LLM call is mocked here. What is faked is the one thing that would otherwise
make these tests take hours — the video backend answers in milliseconds, or in
however long the scenario asks for. That is what lets the process lifecycle be
exercised honestly under short timeouts.

Opt-in:

    python -m pytest tests/e2e -m e2e -q
"""

import json
import time
from pathlib import Path

import pytest

from video_eval_bench.generator.pi_generator import PiGenerationError, PiGenerator
from video_eval_bench.judge.frames import video_frame_count

pytestmark = pytest.mark.e2e


def make_seed(seed_id: str = "marketing_001"):
    from video_eval_bench.dataset import load_seeds

    return next(s for s in load_seeds() if s.seed_id == seed_id)


def run_json(tmp_path: Path, seed_id: str = "marketing_001") -> dict:
    return json.loads((tmp_path / seed_id / "pi_run.json").read_text())


async def test_agent_generates_and_submits_a_video(e2e_config, video_service, tmp_path):
    """
    The whole generator path against the real model: argv, skill loading, the
    extension, the agent's tool loop, and the handoff.
    """
    service = video_service(delay=0.0)
    bench = e2e_config(service)
    path = (await PiGenerator(bench.generator)(make_seed(), tmp_path)).video_path

    assert video_frame_count(path) > 0, "submitted file is not a decodable video"
    assert service.calls, "the agent never called the video service"

    metadata = run_json(tmp_path)
    assert metadata["submitted_path"] == str(path)
    assert metadata["tool_calls"].get("submit_video") == 1
    # A real tool loop, not a single text turn.
    assert metadata["tool_calls"].get("bash", 0) >= 1


async def test_timeout_kills_the_tree_and_frees_the_bench(e2e_config, video_service, tmp_path):
    """
    A generation that outlives its budget must be killed with its children. An
    orphaned curl or ffmpeg would otherwise leak into every later seed.
    """
    service = video_service(delay=300.0)
    bench = e2e_config(service, "generator.timeout_seconds=25")

    started = time.monotonic()
    with pytest.raises(PiGenerationError, match="budget"):
        await PiGenerator(bench.generator)(make_seed(), tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < 60, f"kill took {elapsed:.0f}s"
    assert run_json(tmp_path)["outcome"] == "timeout"
    assert not _leftover_curls(service.url), "a child process outlived the kill"


async def test_finishes_soon_after_submitting(e2e_config, video_service, tmp_path):
    """
    Wall time should track the generation, not drift toward the timeout — which
    is what a broken terminate/grace path looks like.
    """
    service = video_service(delay=0.0)
    bench = e2e_config(service, "generator.timeout_seconds=180", "generator.exit_grace_seconds=5")

    started = time.monotonic()
    await PiGenerator(bench.generator)(make_seed(), tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < 150, f"took {elapsed:.0f}s of a 180s budget"
    assert run_json(tmp_path)["outcome"] in ("exited", "early_terminated")


async def test_a_slow_generation_still_succeeds(e2e_config, video_service, tmp_path):
    """
    The backend stays silent well past the heartbeat interval but inside the
    budget. This is the live proof that no inactivity timeout exists: pi puts no
    timeout on bash, so a real generation blocks silently for minutes.
    """
    service = video_service(delay=20.0)
    bench = e2e_config(
        service, "generator.timeout_seconds=180", "generator.heartbeat_seconds=5"
    )
    path = (await PiGenerator(bench.generator)(make_seed(), tmp_path)).video_path

    assert video_frame_count(path) > 0
    assert run_json(tmp_path)["duration_seconds"] >= 20


async def test_full_loop_including_the_judge(e2e_config, video_service, tmp_path):
    """Generate and grade one seed with the real model on both ends."""
    from video_eval_bench.bench import run_bench
    from video_eval_bench.config import build_judge
    from video_eval_bench.dataset import load_dataset

    service = video_service(delay=0.0)
    bench = e2e_config(service)
    dataset = load_dataset()

    report = await run_bench(
        judge=build_judge(bench.judge, dataset),
        generate=PiGenerator(bench.generator),
        output_dir=tmp_path,
        run_id="e2e",
        dataset=dataset,
        category=bench.run.category,
        max_seeds=1,
    )

    summary = report.summary()
    assert summary["n_generation_errors"] == 0
    assert summary["n_ok"] == 1
    verdict = report.results[0].verdict
    assert verdict is not None and verdict.judge_error is None
    assert 0.0 <= verdict.total_score <= 100.0


def _leftover_curls(service_url: str) -> list:
    """Any surviving process still holding the mock service's URL."""
    survivors = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = entry.joinpath("cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if service_url in cmdline:
            survivors.append(entry.name)
    return survivors
