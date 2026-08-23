"""
Tests for scoring videos generated elsewhere, and for replaying a run's videos
through the judge without regenerating them.

Fully offline: the "external" videos are written by MockGenerator into a
directory that stands in for whatever produced them.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from video_eval_bench.dataset import load_dataset
from video_eval_bench.generator.base import GenerationError
from video_eval_bench.generator.external_generator import ExternalGenerator
from video_eval_bench.generator.manifest import MANIFEST_NAME, load_manifest, write_manifest
from video_eval_bench.generator.mock_generator import MockGenerator
from video_eval_bench.judge.agent import VideoJudge
from video_eval_bench.judge.frames import video_frame_count
from video_eval_bench.judge.llm import MockBackend
from video_eval_bench.schemas import Seed


# ── helpers ───────────────────────────────────────────────────────────────────


def make_seed(seed_id: str = "entertainment_001", category: str = "entertainment") -> Seed:
    return Seed(seed_id=seed_id, category=category, prompt="A short story.")


async def a_video(tmp_path: Path, seed_id: str = "entertainment_001") -> Path:
    """A real, decodable video standing in for one produced outside the harness."""
    source_dir = tmp_path / "elsewhere"
    source_dir.mkdir(exist_ok=True)
    produced = await MockGenerator(n_frames=16)(make_seed(seed_id), source_dir)
    return Path(produced.video_path)


def write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


# ── the manifest format ───────────────────────────────────────────────────────


async def test_relative_paths_resolve_against_the_manifest(tmp_path):
    """
    Not against the cwd: a manifest and its videos travel together, and hydra
    never chdirs, so the cwd is not a stable anchor. This rule is also what lets
    a run directory be its own manifest.
    """
    video = await a_video(tmp_path)
    manifest = write_yaml(
        video.parent / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": video.name}]},
    )

    entry = load_manifest(manifest).get("entertainment_001")
    assert entry.path == video


def test_duplicate_seed_ids_are_rejected(tmp_path):
    """Two videos claiming one seed means one is ignored, silently and by luck."""
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "a", "path": "one.mp4"}, {"seed_id": "a", "path": "two.mp4"}]},
    )
    with pytest.raises(ValueError, match="duplicate entry"):
        load_manifest(manifest)


def test_entry_without_a_path_is_rejected(tmp_path):
    """The complaint names the manifest and the seed, not just the field."""
    manifest = write_yaml(tmp_path / MANIFEST_NAME, {"videos": [{"seed_id": "a"}]})
    with pytest.raises(ValueError, match=r"videos.yaml: entry 'a': path: Field required"):
        load_manifest(manifest)


def test_an_unknown_key_is_rejected(tmp_path):
    """
    A manifest is usually hand-written or scripted. A misspelled key that loads
    fine grades a video with the wrong provenance attached to it.
    """
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "a", "path": "a.mp4", "durationseconds": 12}]},
    )
    with pytest.raises(ValueError, match="durationseconds"):
        load_manifest(manifest)


# ── resolving a seed's video ──────────────────────────────────────────────────


async def test_video_is_placed_in_the_run_directory(tmp_path):
    """
    The report finds a seed's video by globbing the run directory for its name,
    not by reading the path off the result — so a video left where it was would
    be judged fine and then render as missing.
    """
    video = await a_video(tmp_path)
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": str(video)}]},
    )
    run_dir = tmp_path / "run"

    result = await ExternalGenerator(manifest)(make_seed(), run_dir)

    assert Path(result.video_path) == run_dir / "entertainment_001.mp4"
    assert video_frame_count(result.video_path) > 0
    assert result.status == "completed"


async def test_the_source_suffix_is_preserved(tmp_path):
    """
    Naming a .mov `.mp4` because that is the common case produces a file whose
    extension lies about its container. The report embeds all five suffixes.
    """
    video = await a_video(tmp_path)
    renamed = video.with_suffix(".mov")
    shutil.copy2(video, renamed)
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": str(renamed)}]},
    )

    result = await ExternalGenerator(manifest)(make_seed(), tmp_path / "run")
    assert Path(result.video_path).name == "entertainment_001.mov"


async def test_a_seed_absent_from_the_manifest_is_skipped(tmp_path):
    """Nobody claimed a video for it. That is not a generation failure."""
    manifest = write_yaml(tmp_path / MANIFEST_NAME, {"videos": []})

    result = await ExternalGenerator(manifest)(make_seed(), tmp_path / "run")

    assert result.status == "skipped"
    assert result.video_path is None
    assert "entertainment_001" in result.metadata["skip_reason"]


async def test_an_unreadable_video_is_an_error_not_a_skip(tmp_path):
    """
    Someone asked for this one to be judged and it could not be. Letting it
    through would earn a permissive default verdict scoring a broken file 50/100.
    """
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": str(broken)}]},
    )

    with pytest.raises(GenerationError, match="not a readable video"):
        await ExternalGenerator(manifest)(make_seed(), tmp_path / "run")


async def test_a_missing_file_names_the_manifest(tmp_path):
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": "gone.mp4"}]},
    )
    with pytest.raises(GenerationError, match="does not exist"):
        await ExternalGenerator(manifest)(make_seed(), tmp_path / "run")


async def test_symlink_mode_avoids_duplicating_the_file(tmp_path):
    video = await a_video(tmp_path)
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": str(video)}]},
    )

    result = await ExternalGenerator(manifest, copy=False)(make_seed(), tmp_path / "run")

    placed = Path(result.video_path)
    assert placed.is_symlink()
    assert video_frame_count(str(placed)) > 0


async def test_provenance_reaches_the_result(tmp_path):
    """
    The prompt matters especially: the report prefers it over the dataset brief,
    so it shows the prompt that actually produced the video.
    """
    video = await a_video(tmp_path)
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {
            "label": "some-other-model",
            "videos": [
                {
                    "seed_id": "entertainment_001",
                    "path": str(video),
                    "prompt": "the brief actually used",
                    "duration_seconds": 1320,
                    "source": "MiniMax-H3 turbo",
                    "metadata": {"steps": 4},
                }
            ],
        },
    )

    generator = ExternalGenerator(manifest)
    result = await generator(make_seed(), tmp_path / "run")

    assert generator.label == "some-other-model"
    assert result.duration_seconds == 1320
    assert result.metadata["prompt"] == "the brief actually used"
    assert result.metadata["source"] == "MiniMax-H3 turbo"
    assert result.metadata["steps"] == 4
    assert result.metadata["frames"] > 0


# ── through the bench loop ────────────────────────────────────────────────────


async def test_a_partial_import_keeps_the_benchmark_whole(tmp_path):
    """
    One video against a multi-seed category: the covered seed is scored, the rest
    are skipped, and nothing is counted as a generation error.
    """
    from video_eval_bench.bench import run_bench

    video = await a_video(tmp_path)
    manifest = write_yaml(
        tmp_path / MANIFEST_NAME,
        {"videos": [{"seed_id": "entertainment_001", "path": str(video)}]},
    )

    ds = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4),
        generate=ExternalGenerator(manifest),
        output_dir=tmp_path / "run",
        run_id="test_run",
        dataset=ds,
        category="entertainment",
    )

    summary = report.summary()
    assert summary["n_seeds"] == 2
    assert summary["n_skipped"] == 1
    assert summary["n_generation_errors"] == 0
    assert summary["mean_score"] == 100.0  # the one judged seed, not an average over blanks


# ── replay: a run is its own manifest ─────────────────────────────────────────


async def test_a_run_replays_through_the_judge_without_regenerating(tmp_path):
    """
    The whole point of writing the manifest: an agentic seed costs minutes to
    hours, and re-judging it must not cost that again.
    """
    from video_eval_bench.bench import run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    first_dir = tmp_path / "run_a"
    original = await run_bench(
        judge=judge,
        generate=MockGenerator(n_frames=16),
        output_dir=first_dir,
        run_id="run_a",
        dataset=ds,
        category="entertainment",
        manifest_label="run_a",
        manifest_source="generator=mock",
    )

    # The run wrote a manifest of what it produced; feed it straight back.
    replay = await run_bench(
        judge=judge,
        generate=ExternalGenerator(first_dir / MANIFEST_NAME, copy=False),
        output_dir=tmp_path / "run_b",
        run_id="run_b",
        dataset=ds,
        category="entertainment",
    )

    # Nothing was regenerated: every replayed video points at the original run's file.
    for result in replay.results:
        placed = Path(result.video_path)
        assert placed.is_symlink()
        assert placed.resolve().parent == first_dir.resolve()

    assert replay.summary()["n_skipped"] == 0
    assert replay.summary()["mean_score"] == original.summary()["mean_score"]
    assert [r.seed.seed_id for r in replay.results] == [
        r.seed.seed_id for r in original.results
    ]
    # The replay carries forward what produced the videos.
    assert replay.results[0].metadata["source"] == "generator=mock"


async def test_write_manifest_round_trips_through_load_manifest(tmp_path):
    from video_eval_bench.report.base import SeedResult

    video = await a_video(tmp_path)
    result = SeedResult(
        seed=make_seed(),
        video_path=str(video),
        metadata={"turns": 6, "duration_seconds": 42.0, "prompt": "the real brief"},
    )
    path = write_manifest([result], video.parent / MANIFEST_NAME, label="L", source="S")

    manifest = load_manifest(path)
    entry = manifest.get("entertainment_001")
    assert manifest.label == "L"
    assert entry.path == video
    assert entry.duration_seconds == 42.0
    assert entry.prompt == "the real brief"
    assert entry.source == "S"
    assert entry.metadata["turns"] == 6


async def test_skipped_and_errored_seeds_are_left_out_of_the_manifest(tmp_path):
    """A manifest describes videos that exist, so a replay skips them again."""
    from video_eval_bench.report.base import SeedResult

    video = await a_video(tmp_path)
    results = [
        SeedResult(seed=make_seed(), video_path=str(video)),
        SeedResult(seed=make_seed("entertainment_002"), status="skipped"),
        SeedResult(
            seed=make_seed("educational_001"), status="errored", generation_error="boom"
        ),
    ]
    path = write_manifest(results, tmp_path / MANIFEST_NAME)

    assert load_manifest(path).seed_ids == ["entertainment_001"]
