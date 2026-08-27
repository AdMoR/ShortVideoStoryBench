"""
Per-seed reference images: loading, staging, and what each consumer is told.

The whole point of a reference is that a *specific image* reaches the agent and
the judge. Every test here pins one link of that chain — a break anywhere is
silent at runtime (the agent generates a fine video against nothing, the judge
scores it against nothing) and would only show up as a mysteriously low S5.
"""

import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image

from video_eval_bench.dataset import DEFAULT_DATASET_DIR, load_dataset, load_seeds
from video_eval_bench.dataset.seed import Seed, SeedReference
from video_eval_bench.generator.pi_generator import (
    REFERENCE_DIR,
    PiGenerator,
    _SeedPaths,
    _stage_references,
)
from video_eval_bench.judge.agent import VideoJudge
from video_eval_bench.judge.llm import MockBackend
from video_eval_bench.judge.prompt import build_criterion_prompt


# ── helpers ───────────────────────────────────────────────────────────────────


def write_image(path: Path, color=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def write_dataset(tmp_path: Path, references: list[dict]) -> Path:
    """A minimal dataset directory: the real rubrics, one seed we control."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("rubric_a.yaml", "rubric_b.yaml", "rubric_c.yaml", "rubric_d.yaml"):
        shutil.copy2(DEFAULT_DATASET_DIR / name, dataset / name)
    seed = {"seed_id": "marketing_001", "category": "marketing", "prompt": "An ad."}
    if references:
        seed["references"] = references
    (dataset / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [seed]}))
    return dataset


def make_reference(tmp_path: Path, ref_id: str = "hero", suffix: str = ".png") -> SeedReference:
    return SeedReference(
        id=ref_id,
        role="character",
        label="Hero",
        description="The protagonist.",
        path=write_image(tmp_path / f"{ref_id}{suffix}"),
    )


def make_seed(references=()) -> Seed:
    return Seed(
        seed_id="marketing_001",
        category="marketing",
        prompt="An ad.",
        references=list(references),
    )


# ── loading ───────────────────────────────────────────────────────────────────


def test_paths_resolve_against_the_dataset_directory(tmp_path):
    """
    Seeds are written with paths relative to the dataset, but every consumer
    downstream just wants to open the file — so load_seeds is where the two meet.
    """
    dataset = write_dataset(tmp_path, [_entry("hero", "assets/hero.png")])
    write_image(dataset / "assets" / "hero.png")

    (seed,) = load_seeds(dataset)
    assert seed.references[0].path == (dataset / "assets" / "hero.png").resolve()
    assert seed.references[0].path.is_file()


def test_a_missing_image_fails_the_load_naming_the_seed(tmp_path):
    """Fail before the run, not per-seed mid-run: generation costs minutes."""
    dataset = write_dataset(tmp_path, [_entry("hero", "assets/hero.png")])

    with pytest.raises(FileNotFoundError) as exc:
        load_seeds(dataset)
    assert "marketing_001" in str(exc.value)
    assert "hero" in str(exc.value)


def test_duplicate_reference_ids_are_rejected(tmp_path):
    """The id is the staged filename — a duplicate would overwrite an image."""
    dataset = write_dataset(
        tmp_path, [_entry("hero", "assets/a.png"), _entry("hero", "assets/b.png")]
    )
    write_image(dataset / "assets" / "a.png")
    write_image(dataset / "assets" / "b.png")

    with pytest.raises(ValueError, match="hero"):
        load_seeds(dataset)


def test_an_unknown_key_on_a_reference_is_an_error(tmp_path):
    """Seed itself drops unknown keys; a reference must not, silently or not."""
    entry = _entry("hero", "assets/hero.png")
    entry["rol"] = "character"
    dataset = write_dataset(tmp_path, [entry])
    write_image(dataset / "assets" / "hero.png")

    with pytest.raises(Exception, match="rol"):
        load_seeds(dataset)


@pytest.mark.parametrize("bad_id", ["../escape", "sub/dir", "Hero", ""])
def test_an_id_that_is_not_a_slug_is_rejected(tmp_path, bad_id):
    """The id is joined onto a path — a separator in it escapes the workspace."""
    with pytest.raises(Exception):
        SeedReference(
            id=bad_id,
            role="character",
            label="Hero",
            description="The protagonist.",
            path=write_image(tmp_path / "hero.png"),
        )


def test_the_shipped_dataset_has_references_and_they_exist():
    ds = load_dataset()
    with_refs = {s.seed_id for s in ds.seeds if s.references}
    assert with_refs, "the dataset should exercise the reference path"
    for seed in ds.seeds:
        for ref in seed.references:
            assert ref.path.is_file(), f"{seed.seed_id}/{ref.id}"


def _entry(ref_id: str, path: str) -> dict:
    return {
        "id": ref_id,
        "role": "character",
        "label": "Hero",
        "description": "The protagonist.",
        "path": path,
    }


# ── staging into the workspace ────────────────────────────────────────────────


def test_references_are_staged_under_their_id(tmp_path):
    """
    The id becomes the filename so the brief, the workspace and the tool call all
    name the same path — and so no host filename reaches the agent.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    refs = [make_reference(tmp_path, "hero"), make_reference(tmp_path, "set", ".jpg")]

    _stage_references(make_seed(refs), workspace)

    assert (workspace / REFERENCE_DIR / "hero.png").is_file()
    assert (workspace / REFERENCE_DIR / "set.jpg").is_file()


def test_a_seed_without_references_gets_no_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _stage_references(make_seed(), workspace)
    assert not (workspace / REFERENCE_DIR).exists()


def test_generation_stages_references_before_the_agent_starts(pi_config, tmp_path):
    """
    Staged in __call__, not by the agent: an image the agent has to fetch is an
    image it can fail to fetch, and the run would still look like a video problem.
    """
    import asyncio

    config = pi_config()
    seed = make_seed([make_reference(tmp_path, "hero")])
    run_dir = tmp_path / "run"
    asyncio.run(PiGenerator(config)(seed, run_dir))

    staged = run_dir / seed.seed_id / "workspace" / REFERENCE_DIR / "hero.png"
    assert staged.is_file()


# ── what the agent is told ────────────────────────────────────────────────────


def test_the_brief_names_workspace_relative_paths(pi_config, tmp_path):
    config = pi_config()
    seed = make_seed([make_reference(tmp_path, "hero")])
    paths = _SeedPaths(config, tmp_path / "run" / seed.seed_id, tmp_path / "run" / "x.mp4")

    task = PiGenerator(config).render_task(seed, paths)

    assert f"`{REFERENCE_DIR}/hero.png`" in task
    assert "character" in task and "Hero" in task
    assert "The protagonist." in task
    # Relative, so one string is correct on the host and in the container, and
    # the sandbox does not have to be told about references at all.
    assert str(paths.workspace_agent / REFERENCE_DIR) not in task


def test_a_bare_seed_gets_the_brief_it_always_got(pi_config, tmp_path):
    """
    A prompt ablation measured on reference-less seeds must not shift because
    this feature exists, so the block collapses to nothing at all.
    """
    config = pi_config()
    paths = _SeedPaths(config, tmp_path / "run" / "s", tmp_path / "run" / "x.mp4")
    task = PiGenerator(config).render_task(make_seed(), paths)

    assert REFERENCE_DIR not in task
    assert "Work inside the working directory.\n\n## Time" in task


def test_a_description_containing_braces_survives_rendering(pi_config, tmp_path):
    """render_task str.formats the template; a description is substituted into it."""
    config = pi_config()
    ref = make_reference(tmp_path, "hero")
    ref.description = "Wears a shirt reading {brand} in {white}."
    paths = _SeedPaths(config, tmp_path / "run" / "s", tmp_path / "run" / "x.mp4")

    task = PiGenerator(config).render_task(make_seed([ref]), paths)
    assert "{brand}" in task


# ── what the judge is told ────────────────────────────────────────────────────


async def test_the_judge_sees_the_references_before_the_frames(tmp_path):
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    seed = make_seed([make_reference(tmp_path, "hero"), make_reference(tmp_path, "set")])
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path

    verdict = await judge.judge(seed, video)

    assert verdict.judge_error is None
    assert all(call["n_images"] == 2 + 4 for call in backend.calls)
    system = backend.calls[0]["system"]
    assert "2 reference image(s)" in system
    assert 'Image 1 — character, "Hero"' in system


async def test_s5_is_passed_without_a_model_call_when_there_are_no_references(tmp_path):
    """
    Passed, not skipped: a section scores as a percentage of its full weight, so
    an absent S5 would quietly cost a reference-less seed 3 of Section B's 13.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    seed = make_seed()
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path

    verdict = await judge.judge(seed, video)

    s5 = next(s for s in verdict.scores if s.criterion == "S5")
    assert s5.passed and s5.score == 3.0
    assert not any("S5" in call["system"] for call in backend.calls)
    assert all(call["n_images"] == 4 for call in backend.calls)


def test_the_judge_prompt_of_a_bare_seed_is_unchanged():
    ds = load_dataset()
    criterion = ds.rubric_a.criteria[0]
    prompt = build_criterion_prompt(
        make_seed(), ds.categories["marketing"], criterion, n_frames=8
    )
    assert "You are shown 8 frames sampled evenly" in prompt
    assert "Reference" not in prompt.split("## Criterion to check")[0]


async def test_an_unreadable_reference_leaves_the_prompt_and_the_payload_agreed(tmp_path):
    """
    The header numbers the images it claims were sent. Drop one from the payload
    without dropping it from the header and "Image 2" names the wrong picture —
    which is worse than judging without either.
    """
    from video_eval_bench.bench import MockGenerator

    ds = load_dataset()
    good = make_reference(tmp_path, "hero")
    broken = make_reference(tmp_path, "set")
    broken.path.write_bytes(b"not an image at all")

    seed = make_seed([broken, good])
    backend = MockBackend()
    judge = VideoJudge(backend=backend, dataset=ds, n_frames=4)
    video = (await MockGenerator(n_frames=16)(seed, tmp_path)).video_path

    verdict = await judge.judge(seed, video)

    assert verdict.judge_error is None
    assert all(call["n_images"] == 1 + 4 for call in backend.calls)
    system = backend.calls[0]["system"]
    assert "1 reference image(s)" in system
    assert 'Image 1 — character, "Hero"' in system
    assert "Image 2" not in system
