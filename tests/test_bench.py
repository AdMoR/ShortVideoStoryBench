"""
Tests for the video evaluation benchmark.

Fully offline: mock LLM backend + synthetic videos. No network, no GPU, no pi.
"""

import json
import pathlib

import pytest

from video_eval_bench.dataset import (
    load_dataset,
    load_genres,
    load_rubrics,
    load_safety_checks,
    load_seeds,
)
from video_eval_bench.judge.agent import VideoJudge, parse_verdict_json
from video_eval_bench.judge.frames import extract_frames
from video_eval_bench.judge.llm import MockBackend
from video_eval_bench.judge.prompt import build_criterion_prompt, build_safety_prompt
from video_eval_bench.schemas import JudgeVerdict, Seed


# ── helpers ───────────────────────────────────────────────────────────────────

# One criterion from each dimension, so a verdict built from this seed exercises
# the whole breakdown. Weights: 3 + 3 + 2 + 3 + 2 = 13.
TEST_RUBRICS = ["SUBJ1", "ART1", "PHYS1", "SEQ1", "CAM1"]


def make_seed(category: str = "entertainment", rubrics=None, **kwargs) -> Seed:
    return Seed(
        seed_id="entertainment_001",
        category=category,
        prompt="A short story.",
        rubrics=list(TEST_RUBRICS if rubrics is None else rubrics),
        **kwargs,
    )


# ── dataset / rubrics ─────────────────────────────────────────────────────────


def test_rubric_library_loads():
    lib = load_rubrics()
    assert len(lib.criteria) > 20
    assert {"SUBJ1", "ENV1", "TEMP1", "ART1", "REF1"} <= {c.id for c in lib.criteria}
    assert all(c.weight > 0 for c in lib.criteria)
    # Every criterion sits in a declared dimension (RubricLibrary enforces it,
    # but this is the invariant the report's grouping depends on).
    keys = {d.key for d in lib.dimensions}
    assert {c.dimension for c in lib.criteria} <= keys


def test_rubric_ids_are_unique():
    """Ids index the library and label report rows; a collision would hide one."""
    lib = load_rubrics()
    ids = [c.id for c in lib.criteria]
    assert len(ids) == len(set(ids))


def test_reference_dependent_criteria_are_marked():
    """
    The `references` section is the one section with a hard activation rule, and
    the flag is what carries it: every criterion in it, and nothing outside it.
    """
    lib = load_rubrics()
    marked = [c.id for c in lib.criteria if c.requires_references]
    assert marked == ["REF1", "REFCOV1"]
    assert {lib.section_of(cid) for cid in marked} == {"references"}
    assert [c.id for c in lib.sections[[s.key for s in lib.sections].index("references")].criteria] == marked


def test_subject_consistency_covers_more_than_characters():
    """
    SUBJ1 replaced a character-only identity check: an object, an animal or a
    vehicle is just as much the subject of a brief, and drifts just as visibly.
    """
    description = load_rubrics().get("SUBJ1").description.lower()
    for kind in ("animal", "object", "vehicle", "person"):
        assert kind in description


def test_environment_consistency_exists_and_is_scoped_to_repeated_places():
    """
    ENV1 must not fire on a brief that deliberately changes location — only on
    a place that contradicts itself between two shots of it.
    """
    env1 = load_rubrics().get("ENV1")
    assert env1.dimension == "consistency"
    assert env1.critical
    assert "same place" in env1.description.lower()


def test_genres_loaded():
    genres = load_genres()
    assert set(genres) == {
        "entertainment", "educational", "marketing",
        "social_media", "gaming", "event",
    }


def test_safety_checks():
    checks = load_safety_checks()
    assert len(checks) == 10
    assert {c.id for c in checks} == {f"D{i}" for i in range(1, 11)}
    assert all(c.veto for c in checks)


def test_load_seeds_filter():
    seeds = load_seeds(category="marketing")
    assert seeds and all(s.category == "marketing" for s in seeds)


def test_load_dataset_validates_genres_and_criteria():
    ds = load_dataset()
    assert ds.seeds
    assert {s.category for s in ds.seeds} <= set(ds.genres)
    for seed in ds.seeds:
        assert seed.rubrics
        # Library ids, or one of the seed's own namespaced local criteria — the
        # third tier, for a check one brief needs and no other does.
        known = set(seed.local_criteria_by_id())
        assert all(cid in ds.rubrics or cid in known for cid in seed.criterion_ids())
        assert all(cid.startswith(f"{seed.seed_id}.") for cid in known)


def test_load_dataset_rejects_a_seed_naming_an_unknown_criterion(tmp_path):
    """
    A typo in a seed's rubric list must fail the load. Silently dropping it
    would shrink that seed's rubric without saying so, and its score would still
    look like a valid number.
    """
    import shutil

    from video_eval_bench.dataset import DEFAULT_DATASET_DIR

    shutil.copytree(DEFAULT_DATASET_DIR, tmp_path / "dataset")
    seeds = (tmp_path / "dataset" / "seeds.yaml")
    seeds.write_text(seeds.read_text().replace("      - SUBJ1\n", "      - SUBJ99\n", 1))
    with pytest.raises(ValueError, match="SUBJ99"):
        load_dataset(tmp_path / "dataset")


def test_a_seed_lists_only_criteria_its_brief_can_fail():
    """
    The regression this whole selection mechanism exists for: the social-media
    seed used to be asked about geography, historical period and cultural
    symbolism, pass all three by default, and bank the weight.
    """
    ds = load_dataset()
    seed = next(s for s in ds.seeds if s.seed_id == "social_media_001")
    # GEO1, the third of the three, no longer exists at all — it could not be
    # graded even where it did apply.
    assert "GEO1" not in ds.rubrics
    assert not ({"HIST1", "CULT1"} & set(seed.criterion_ids()))
    # It does carry the ones its brief is actually about.
    assert {"ENV1", "HOOK1", "REVEAL1", "CTA1", "AR1"} <= set(seed.criterion_ids())


def test_reference_criterion_is_only_listed_by_seeds_that_have_references():
    ds = load_dataset()
    for seed in ds.seeds:
        if "REF1" in seed.criterion_ids():
            assert seed.references, seed.seed_id


# ── prompt ────────────────────────────────────────────────────────────────────


def test_criterion_prompt_covers_one_criterion_only():
    ds = load_dataset()
    seed = make_seed()
    subj1 = ds.rubrics.get("SUBJ1")
    text = build_criterion_prompt(seed, "Entertainment & Storytelling", subj1, n_frames=8)
    # Only the one criterion is present, not the rest of the seed's rubric.
    assert "SUBJ1" in text
    assert "ART1" not in text and "SEQ1" not in text and "D1" not in text
    assert "A short story." in text
    assert "8 frames" in text
    assert '"passed"' in text


def test_criterion_prompt_in_video_mode_says_so():
    """
    In frames mode the model is told to infer motion from the gaps; in video
    mode that instruction is wrong and would invite it to hedge.
    """
    ds = load_dataset()
    text = build_criterion_prompt(
        make_seed(), "Entertainment & Storytelling", ds.rubrics.get("SUBJ1"),
        n_frames=0, video=True,
    )
    assert "watch in full" in text
    assert "frames sampled" not in text
    assert "cannot watch the video directly" not in text


def test_criterion_prompt_does_not_offer_a_not_applicable_pass():
    """
    The old prompt told the judge to pass a criterion it thought inapplicable.
    A seed now only lists criteria that apply, so that escape hatch would just
    hand back free weight.
    """
    ds = load_dataset()
    text = build_criterion_prompt(
        make_seed(), "Entertainment & Storytelling", ds.rubrics.get("SUBJ1"), n_frames=8
    )
    assert "mark it as PASSED" not in text
    assert "it applies" in text


def test_safety_prompt_covers_one_check_only():
    ds = load_dataset()
    d1 = next(c for c in ds.safety_checks if c.id == "D1")
    text = build_safety_prompt(make_seed(), "Entertainment & Storytelling", d1, n_frames=8)
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
    assert verdict.total_score == 100.0
    assert all(d.score == 100.0 for d in verdict.dimensions)
    assert verdict.critical_failures == []
    assert verdict.safety_veto is False
    assert verdict.passed is True
    # Exactly the criteria the seed listed — the loader attaches no section on top.
    # The order is the library's, not the seed's, and the library is ordered by
    # section: all five are `general` ones, in the order that section declares
    # them. That is what keeps report columns in the same order across seeds.
    assert [s.criterion for s in verdict.scores] == ["SUBJ1", "SEQ1", "ART1", "CAM1", "PHYS1"]
    assert len(verdict.safety) == 10
    # One focused backend call per criterion, not one giant call.
    assert len(backend.calls) == len(TEST_RUBRICS) + 10
    assert "SUBJ1" in backend.calls[0]["system"]
    # Frames mode: the clip is sent as stills, never as a video.
    assert all(c["n_video_bytes"] == 0 for c in backend.calls)
    assert all(c["n_images"] == 4 for c in backend.calls)


async def test_judge_scores_flat_over_the_seeds_own_weight(tmp_path):
    """
    The total is weight earned over weight asked for. It used to be the mean of
    three fixed sections, which gave a 3-criterion baseline the same say as an
    8-criterion genre rubric.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()

    class FailOne(MockBackend):
        def _mock_response(self, system):
            if "## Criterion to check: SUBJ1" in system:
                return json.dumps({"passed": False, "comment": "drifted"})
            return super()._mock_response(system)

    judge = VideoJudge(backend=FailOne(), dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path
    verdict = await judge.judge(make_seed(), video)

    # SUBJ1 is worth 3 of the seed's 13 points.
    assert verdict.total_score == round(100 * 10 / 13, 2)
    # ...and it is the only criterion in its dimension for this seed, so that
    # dimension is a clean zero while the rest are untouched.
    assert verdict.dimension_score("consistency") == 0.0
    assert verdict.dimension_score("technical") == 100.0
    assert verdict.critical_failures == ["SUBJ1"]


async def test_judge_dimensions_only_cover_what_the_seed_asked_for(tmp_path):
    """An empty dimension row would read as a failed dimension, not an unasked one."""
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    seed = make_seed(rubrics=["SUBJ1", "ENV1"])  # consistency only
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path

    verdict = await judge.judge(seed, video)
    assert [d.dimension for d in verdict.dimensions] == ["consistency"]
    assert verdict.dimensions[0].total == 6


async def test_judge_orders_criteria_by_the_library_not_the_seed(tmp_path):
    """Two seeds sharing criteria should produce report tables that read alike."""
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    seed = make_seed(rubrics=["CAM1", "SUBJ1", "ART1"])
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path
    verdict = await judge.judge(seed, video)
    assert [s.criterion for s in verdict.scores] == ["SUBJ1", "ART1", "CAM1"]


async def test_judge_partial_failures(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    backend = MockBackend(fail_rate=1 / 3)  # every 3rd criterion fails
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)

    assert verdict.judge_error is None
    assert 0.0 < verdict.total_score < 100.0
    earned = sum(s.score for s in verdict.scores)
    asked = sum(d.total for d in verdict.dimensions)
    assert verdict.total_score == round(100 * earned / asked, 2)


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
    d1 = next(r for r in verdict.safety if r.check_id == "D1")
    assert d1.violation is True


async def test_judge_pass_threshold(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    backend = MockBackend(fail_rate=0.5)
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)
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
    assert f"{len(TEST_RUBRICS) + 10}/{len(TEST_RUBRICS) + 10}" in verdict.judge_error
    assert verdict.total_score == 0.0
    assert all(d.score == 0.0 for d in verdict.dimensions)
    assert verdict.safety_veto is True
    assert verdict.passed is False
    assert all(not s.passed for s in verdict.scores)
    assert all(r.violation for r in verdict.safety)


async def test_reference_criterion_passes_without_a_model_call(tmp_path):
    """
    A seed listing REF1 whose images will not load must not be scored zero for
    it: its weight is already in the denominator, so a dataset problem would
    read as a video defect.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    seed = make_seed(rubrics=["SUBJ1", "REF1"])  # lists REF1, carries no images
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path

    verdict = await judge.judge(seed, video)
    ref = next(s for s in verdict.scores if s.criterion == "REF1")
    assert ref.passed and ref.score == 3
    assert not any("REF1" in c["system"] for c in backend.calls)


async def test_judge_sends_the_clip_in_video_mode(tmp_path):
    """
    media=video hands the backend the file itself, once per call, and stops
    extracting frames — the whole point is not inferring motion from stills.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, media="video")
    video = (await MockGenerator(n_frames=16)(make_seed(), tmp_path)).video_path

    verdict = await judge.judge(make_seed(), video)
    assert verdict.total_score == 100.0
    size = pathlib.Path(video).stat().st_size
    assert all(c["n_video_bytes"] == size for c in backend.calls)
    assert all(c["n_images"] == 0 for c in backend.calls)
    assert "watch in full" in backend.calls[0]["system"]


async def test_judge_video_mode_missing_file_is_a_permissive_default():
    ds = load_dataset()

    class Exploding(MockBackend):
        async def complete(self, *a, **kw):
            raise AssertionError("complete should not be reached")

    judge = VideoJudge(backend=Exploding(), dataset=ds, media="video")
    verdict = await judge.judge(make_seed(), "/nonexistent.mp4")
    assert verdict.judge_error is not None
    assert "no media loaded" in verdict.judge_error


async def test_judge_rejects_an_unknown_media_mode():
    ds = load_dataset()
    with pytest.raises(ValueError, match="media must be"):
        VideoJudge(backend=MockBackend(), dataset=ds, media="stills")


async def test_judge_missing_video_returns_permissive_default():
    ds = load_dataset()
    # Frames are extracted once, up front, before any backend call is made —
    # a missing file fails extraction first, so complete() is never reached.
    from video_eval_bench.judge.llm import VisionLLM

    class FramesOnlyBackend(VisionLLM):
        async def complete(self, system, user_text, images, video=None):
            raise AssertionError("complete should not be reached")

    judge = VideoJudge(backend=FramesOnlyBackend(), dataset=ds, n_frames=4)
    verdict = await judge.judge(make_seed(), "/nonexistent.mp4")
    assert verdict.judge_error is not None
    assert "no media loaded" in verdict.judge_error
    assert verdict.passed is True


async def test_judge_unknown_genre():
    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    verdict = await judge.judge(make_seed("nonexistent"), "/whatever.mp4")
    assert verdict.judge_error is not None
    assert "unknown genre" in verdict.judge_error


async def test_judge_unknown_criterion_on_a_hand_built_seed():
    """load_dataset refuses this; a seed built in code must not slip past it."""
    ds = load_dataset()
    judge = VideoJudge(backend=MockBackend(), dataset=ds, n_frames=4)
    verdict = await judge.judge(make_seed(rubrics=["SUBJ1", "NOPE1"]), "/whatever.mp4")
    assert verdict.judge_error is not None
    assert "NOPE1" in verdict.judge_error


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
    # Counted off the dataset, not pinned: adding a seed is a routine dataset
    # change and should not have to come with a test edit.
    assert summary["n_seeds"] == len(ds.seeds)
    assert summary["n_ok"] == len(ds.seeds)
    assert summary["n_generation_errors"] == 0
    assert summary["n_judge_errors"] == 0
    assert set(summary["per_category"]) == {s.category for s in ds.seeds}
    assert summary["per_category"]["entertainment"]["n_seeds"] == sum(
        s.category == "entertainment" for s in ds.seeds
    )
    # report serialises
    data = report.to_json()
    assert data["results"][0]["verdict"]["seed_id"] == "entertainment_001"
    assert data["results"][0]["verdict"]["total_score"] == 100.0
    dims = data["results"][0]["verdict"]["dimensions"]
    assert dims and all(d["score"] == 100.0 for d in dims)


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
    assert report.summary()["n_seeds"] == sum(s.category == "event" for s in ds.seeds)
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
        seed_ids=["entertainment_001", "entertainment_002"],
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
        async def complete(self, system, user, images, video=None):
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
        seed_ids=["entertainment_001", "entertainment_002"],
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
