"""
Tests for the video evaluation benchmark.

Fully offline: mock LLM backend + synthetic videos. No network, no GPU, no pi.
"""

import json

import pytest

from video_eval_bench.dataset import (
    load_categories,
    load_dataset,
    load_rubric_a,
    load_rubric_b,
    load_safety_checks,
    load_seeds,
)
from video_eval_bench.judge.agent import VideoJudge, parse_verdict_json
from video_eval_bench.judge.frames import extract_frames
from video_eval_bench.judge.llm import MockBackend
from video_eval_bench.judge.prompt import build_criterion_prompt, build_safety_prompt
from video_eval_bench.schemas import JudgeVerdict, Seed


# ── helpers ───────────────────────────────────────────────────────────────────


def make_seed(category: str = "entertainment") -> Seed:
    return Seed(seed_id="entertainment_001", category=category, prompt="A short story.")


# ── dataset / rubrics ─────────────────────────────────────────────────────────


def test_rubric_a_totals_10():
    rubric = load_rubric_a()
    assert rubric.total_points == 10
    assert {c.id for c in rubric.criteria} == {"U1", "U2", "U3"}
    assert all(c.critical for c in rubric.criteria)


def test_rubric_b_totals_10():
    rubric = load_rubric_b()
    assert rubric.total_points == 10
    assert {c.id for c in rubric.criteria} == {"S1", "S2", "S3", "S4"}


def test_genre_rubrics_loaded():
    cats = load_categories()
    assert set(cats) == {
        "entertainment", "educational", "marketing",
        "social_media", "gaming", "event",
    }
    # Rubrics do not need to sum to a fixed total — each is normalized by
    # its own weight sum. Just require positive weights.
    for cat in cats.values():
        assert cat.rubric.total_points > 0, cat.key


def test_safety_checks():
    checks = load_safety_checks()
    assert len(checks) == 10
    assert {c.id for c in checks} == {f"D{i}" for i in range(1, 11)}
    assert all(c.veto for c in checks)


def test_load_seeds_filter():
    seeds = load_seeds(category="marketing")
    assert seeds and all(s.category == "marketing" for s in seeds)


def test_load_dataset_validates_categories():
    ds = load_dataset()
    assert ds.seeds
    assert {s.category for s in ds.seeds} <= set(ds.categories)


# ── prompt ────────────────────────────────────────────────────────────────────


def test_criterion_prompt_covers_one_criterion_only():
    ds = load_dataset()
    seed = make_seed()
    category = ds.categories["entertainment"]
    u1 = next(c for c in ds.rubric_a.criteria if c.id == "U1")
    text = build_criterion_prompt(seed, category, u1, n_frames=8)
    # Only the one criterion is present, not the rest of the rubric.
    assert "U1" in text
    assert "U2" not in text and "U3" not in text
    assert "E1" not in text and "D1" not in text
    assert "A short story." in text
    assert "8 frames" in text
    assert '"passed"' in text


def test_criterion_prompt_genre_specific():
    ds = load_dataset()
    category = ds.categories["gaming"]
    g1 = next(c for c in category.rubric.criteria if c.id == "G1")
    text = build_criterion_prompt(make_seed("gaming"), category, g1, n_frames=4)
    assert "G1" in text
    assert "E1" not in text


def test_safety_prompt_covers_one_check_only():
    ds = load_dataset()
    category = ds.categories["entertainment"]
    d1 = next(c for c in ds.safety_checks if c.id == "D1")
    text = build_safety_prompt(make_seed(), category, d1, n_frames=8)
    assert "D1" in text
    assert "D2" not in text
    assert '"violation"' in text


# ── frame extraction ──────────────────────────────────────────────────────────


def test_extract_frames_missing_file():
    assert extract_frames("/nonexistent/video.mp4") == []


def test_extract_frames_from_synthetic_video(tmp_path):
    from video_eval_bench.bench import MockGenerator

    import asyncio

    gen = MockGenerator(n_frames=16)
    path = asyncio.run(gen(make_seed(), tmp_path)).video_path
    frames = extract_frames(path, n=4)
    assert len(frames) == 4
    assert all(f[:2] == b"\xff\xd8" for f in frames)  # JPEG magic bytes


# ── JSON parsing ──────────────────────────────────────────────────────────────


def test_parse_verdict_json_plain():
    assert parse_verdict_json('{"a": 1}') == {"a": 1}


def test_parse_verdict_json_embedded():
    assert parse_verdict_json('blah {"a": 1} blah') == {"a": 1}


def test_parse_verdict_json_garbage():
    assert parse_verdict_json("no json here") is None


# ── judge agent (mock backend) ────────────────────────────────────────────────


async def test_judge_full_pipeline_mock(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)

    assert verdict.judge_error is None
    assert verdict.seed_id == "entertainment_001"
    # All criteria pass in the default mock → 100% in every section
    assert verdict.section_a == 100.0
    assert verdict.section_b == 100.0
    assert verdict.section_c == 100.0
    assert verdict.total_score == 100.0
    assert verdict.safety_veto is False
    assert verdict.passed is True
    # 3 + 4 + 8 = 15 criteria scored
    assert len(verdict.scores) == 15
    assert len(verdict.safety) == 10
    # backend received one focused call per criterion, not one giant call
    assert len(backend.calls) == 15 + 10  # criteria + safety checks
    assert "U1" in backend.calls[0]["system"]  # first rubric_a criterion
    assert any("E1" in c["system"] for c in backend.calls)


async def test_judge_partial_failures(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    # Every 3rd criterion fails
    backend = MockBackend(fail_rate=1 / 3)
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)

    assert verdict.judge_error is None
    assert verdict.total_score < 100.0
    # Sections are percentages (0-100); total is their mean
    assert 0.0 <= verdict.section_a <= 100.0
    assert 0.0 <= verdict.section_b <= 100.0
    assert 0.0 <= verdict.section_c <= 100.0
    assert verdict.total_score == round(
        (verdict.section_a + verdict.section_b + verdict.section_c) / 3.0, 2
    )


async def test_judge_safety_veto_blocks_pass(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    backend = MockBackend(veto=True)  # all criteria pass, but D1 violated
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)

    assert verdict.total_score == 100.0  # perfect score...
    assert verdict.safety_veto is True   # ...but vetoed
    assert verdict.passed is False
    assert verdict.passed is False
    d1 = next(r for r in verdict.safety if r.check_id == "D1")
    assert d1.violation is True


async def test_judge_pass_threshold(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    # Fail enough criteria to drop below the 60/100 threshold
    backend = MockBackend(fail_rate=0.5)
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)
    # Half the criteria fail → ~50% per section → below the 60 threshold
    assert verdict.total_score < 60.0
    assert verdict.passed is False


async def test_judge_broken_backend_degrades_conservatively(tmp_path):
    """
    A backend that never returns parseable JSON no longer aborts the whole
    verdict to a flat permissive default (that was only sensible for a
    single all-or-nothing call). With one call per criterion, each failed
    call is scored conservatively on its own — criteria fail, safety checks
    are flagged — and the failures are surfaced via judge_error.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(fail=True), dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path
    verdict = await judge.judge(make_seed(), video)
    assert verdict.judge_error is not None
    assert "25/25" in verdict.judge_error
    # Every criterion failed -> 0%; every safety check fails safe -> vetoed.
    assert verdict.total_score == 0.0
    assert verdict.safety_veto is True
    assert verdict.passed is False
    assert all(not s.passed for s in verdict.scores)
    assert all(r.violation for r in verdict.safety)


async def test_judge_missing_video_returns_permissive_default():
    ds = load_dataset()
    # Frames are extracted once, up front, before any backend call is made —
    # a missing file fails extraction first, so complete() is never reached.
    from video_eval_bench.judge.llm import VisionLLM

    class FramesOnlyBackend(VisionLLM):
        async def complete(self, system, user_text, images):
            raise AssertionError("complete should not be reached")

    judge = VideoJudge(backend=FramesOnlyBackend(), dataset=ds, n_frames=4)
    verdict = await judge.judge(make_seed(), "/nonexistent.mp4")
    assert verdict.judge_error is not None
    assert "no frames" in verdict.judge_error
    assert verdict.passed is True


async def test_judge_unknown_category():
    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    verdict = await judge.judge(make_seed("nonexistent"), "/whatever.mp4")
    assert verdict.judge_error is not None
    assert "unknown category" in verdict.judge_error


# ── full bench run (mock mode) ────────────────────────────────────────────────


async def test_run_bench_mock(tmp_path):
    from video_eval_bench.bench import MockGenerator, run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=MockGenerator(n_frames=16),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
    )

    summary = report.summary()
    assert summary["n_seeds"] == 8
    assert summary["n_ok"] == 8
    assert summary["n_generation_errors"] == 0
    assert summary["n_judge_errors"] == 0
    assert set(summary["per_category"]) == {
        "entertainment", "educational", "marketing",
        "social_media", "gaming", "event",
    }
    assert summary["per_category"]["entertainment"]["n_seeds"] == 2
    # report serialises
    data = report.to_json()
    assert data["results"][0]["verdict"]["seed_id"] == "entertainment_001"
    assert data["results"][0]["verdict"]["total_score"] == 100.0
    assert data["results"][0]["verdict"]["section_c"] == 100.0


async def test_run_bench_category_filter(tmp_path):
    from video_eval_bench.bench import MockGenerator, run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=MockGenerator(n_frames=16),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        category="event",
    )
    assert report.summary()["n_seeds"] == 1
    assert all(r.seed.category == "event" for r in report.results)


async def test_run_bench_max_seeds_caps_the_run(tmp_path):
    """Ablations run on one or two seeds; a full pass is minutes per seed."""
    from video_eval_bench.bench import MockGenerator, run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=MockGenerator(n_frames=16),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        max_seeds=2,
    )
    assert report.summary()["n_seeds"] == 2


async def test_run_bench_seed_ids_select_exactly(tmp_path):
    from video_eval_bench.bench import MockGenerator, run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=MockGenerator(n_frames=16),
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        seed_ids=["gaming_001"],
    )
    assert [r.seed.seed_id for r in report.results] == ["gaming_001"]


async def test_run_bench_rejects_unknown_seed_ids(tmp_path):
    """A typo'd seed id must not silently run nothing."""
    from video_eval_bench.bench import MockGenerator, run_bench

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    with pytest.raises(ValueError, match="Unknown seed_ids"):
        await run_bench(
            judge=judge,
            generate=MockGenerator(n_frames=16),
            output_dir=tmp_path,
            run_id="test_run",
            dataset=ds,
            seed_ids=["gaming_999"],
        )


async def test_run_bench_records_generation_errors(tmp_path):
    from video_eval_bench.bench import run_bench

    async def broken(seed, output_dir):
        raise RuntimeError("GPU on fire")

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=broken,
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        category="entertainment",
    )
    summary = report.summary()
    assert summary["n_generation_errors"] == 2
    assert summary["n_ok"] == 0
    assert all(r.generation_error == "GPU on fire" for r in report.results)


async def test_failed_seed_still_records_its_cost(tmp_path):
    """
    A seed that burns an hour before timing out is the most expensive one in the
    run; reporting it as free hides exactly the cost worth seeing.
    """
    import asyncio

    from video_eval_bench.bench import run_bench

    async def slow_failure(seed, output_dir):
        await asyncio.sleep(0.05)
        raise RuntimeError("timed out")

    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    report = await run_bench(
        judge=judge,
        generate=slow_failure,
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        category="marketing",
    )
    assert report.results[0].duration_seconds > 0
    assert report.summary()["total_duration_seconds"] > 0


# ── the generator seam ────────────────────────────────────────────────────────


async def test_generator_metadata_and_duration_reach_the_report(tmp_path):
    """
    Whatever the generator reports about a seed rides back with the video, rather
    than being fetched afterwards by id, and a generator that knows its own cost
    overrides the bench's stopwatch.
    """
    from video_eval_bench.bench import MockGenerator, run_bench
    from video_eval_bench.generator.base import GenerationResult

    inner = MockGenerator(n_frames=16)

    async def reporting(seed, output_dir):
        produced = await inner(seed, output_dir)
        return GenerationResult(
            seed_id=seed.seed_id,
            video_path=produced.video_path,
            metadata={"turns": 7, "source": "somewhere else"},
            duration_seconds=1320.0,
        )

    ds = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4),
        generate=reporting,
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        seed_ids=["entertainment_001"],
    )
    (result,) = report.results
    assert result.status == "completed"
    assert result.metadata["turns"] == 7
    # Not the fraction of a second this process actually took.
    assert result.duration_seconds == 1320.0


async def test_generation_error_carries_its_metadata(tmp_path):
    """
    A run that burned its budget before failing still knows its turn count, and
    that is the most interesting thing about it. The failure path must not drop it.
    """
    from video_eval_bench.bench import run_bench
    from video_eval_bench.generator.base import GenerationError

    async def expensive_failure(seed, output_dir):
        raise GenerationError("budget exceeded", metadata={"turns": 46, "outcome": "timeout"})

    ds = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4),
        generate=expensive_failure,
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        seed_ids=["entertainment_001"],
    )
    (result,) = report.results
    assert result.status == "errored"
    assert result.generation_error == "budget exceeded"
    assert result.metadata["turns"] == 46
    assert report.summary()["n_generation_errors"] == 1


async def test_skipped_seed_is_never_judged(tmp_path):
    """
    A skipped seed has no video. Handing the judge a missing path would earn a
    permissive default verdict — a fabricated 50/100 for a video that does not
    exist — so the judge must not be called at all.
    """
    from video_eval_bench.bench import run_bench
    from video_eval_bench.generator.base import GenerationResult

    class ExplodingBackend:
        async def complete(self, system, user, images):
            raise AssertionError("the judge must not be called for a skipped seed")

    async def has_nothing(seed, output_dir):
        return GenerationResult(seed_id=seed.seed_id, status="skipped")

    ds = load_dataset()
    report = await run_bench(
        judge=VideoJudge(backend=ExplodingBackend(), dataset=ds, n_frames=4),
        generate=has_nothing,
        output_dir=tmp_path,
        run_id="test_run",
        dataset=ds,
        category="entertainment",
    )
    summary = report.summary()
    assert summary["n_skipped"] == 2
    # A skip is not a failure, and the benchmark did not shrink to fit.
    assert summary["n_generation_errors"] == 0
    assert summary["n_seeds"] == 2
    assert summary["mean_score"] is None
    assert all(r.verdict is None for r in report.results)


async def test_result_for_the_wrong_seed_is_rejected(tmp_path):
    """
    One copy-pasted seed_id in a manifest would otherwise grade a video against
    another seed's rubric and report a perfectly plausible score for it.
    """
    from video_eval_bench.bench import MockGenerator, run_bench
    from video_eval_bench.generator.base import GenerationResult

    inner = MockGenerator(n_frames=16)

    async def mislabelled(seed, output_dir):
        produced = await inner(seed, output_dir)
        return GenerationResult(seed_id="somebody_else", video_path=produced.video_path)

    ds = load_dataset()
    with pytest.raises(ValueError, match="somebody_else"):
        await run_bench(
            judge=VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4),
            generate=mislabelled,
            output_dir=tmp_path,
            run_id="test_run",
            dataset=ds,
            seed_ids=["entertainment_001"],
        )


async def test_manifest_survives_a_run_that_dies_partway(tmp_path):
    """
    The runs worth replaying are the ones that do not finish. A manifest written
    only on a clean exit would strand the videos already paid for.
    """
    import yaml

    from video_eval_bench.bench import MockGenerator, run_bench
    from video_eval_bench.generator.manifest import MANIFEST_NAME

    inner = MockGenerator(n_frames=16)
    seen = []

    async def dies_on_the_second(seed, output_dir):
        seen.append(seed.seed_id)
        if len(seen) > 1:
            raise KeyboardInterrupt("killed")
        return await inner(seed, output_dir)

    ds = load_dataset()
    with pytest.raises(KeyboardInterrupt):
        await run_bench(
            judge=VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4),
            generate=dies_on_the_second,
            output_dir=tmp_path,
            run_id="test_run",
            dataset=ds,
            category="entertainment",
        )

    manifest = yaml.safe_load((tmp_path / MANIFEST_NAME).read_text())
    assert [v["seed_id"] for v in manifest["videos"]] == [seen[0]]
