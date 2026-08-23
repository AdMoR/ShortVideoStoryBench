"""
Benchmark runner: for each seed, get a video (via an injectable generator) and
grade it with the judge.

The generator is a callable so this module stays decoupled from any specific
video model:

    async def generate(seed: Seed, output_dir: Path) -> GenerationResult

It may come back with a video, or say it has none for this seed (`skipped`), or
raise `GenerationError` (`errored`). The three are kept apart in the report on
purpose: "nobody supplied a video for this seed" is not a generation failure, and
counting it as one would inflate the error rate of an import that never claimed
to cover the whole benchmark.

The dataset drives the loop either way — the seed selection is the benchmark, so
a partial run says so instead of quietly shrinking its own denominator.

In mock mode, `MockGenerator` writes a tiny synthetic video so the whole
pipeline runs offline.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from video_eval_bench.dataset import Dataset, load_dataset
from video_eval_bench.generator.base import GenerateFn, GenerationError, GenerationResult
from video_eval_bench.generator.manifest import MANIFEST_NAME, write_manifest
from video_eval_bench.generator.mock_generator import MockGenerator
from video_eval_bench.judge.agent import VideoJudge
from video_eval_bench.report.base import BenchReport, SeedResult

logger = logging.getLogger(__name__)


async def run_bench(
    judge: VideoJudge,
    generate: GenerateFn,
    output_dir: Path,
    run_id: str,
    dataset: Optional[Dataset] = None,
    dataset_dir: Optional[Path] = None,
    category: Optional[str] = None,
    seed_ids: Optional[List[str]] = None,
    max_seeds: Optional[int] = None,
    manifest_label: str = "",
    manifest_source: str = "",
) -> BenchReport:
    """
    Run the full benchmark: generate + judge every seed (sequentially).

    Per-seed failures are recorded on the result; the run continues.

    `category`, `seed_ids` and `max_seeds` narrow the selection, in that order.
    An agentic generator can take many minutes per seed, so ablations normally
    run on one or two seeds and only a settled configuration earns a full pass.
    """
    if dataset is None:
        dataset = load_dataset(dataset_dir)
    seeds = dataset.seeds
    if category:
        seeds = [s for s in seeds if s.category == category]
    if seed_ids:
        wanted = list(seed_ids)
        unknown = sorted(set(wanted) - {s.seed_id for s in seeds})
        if unknown:
            raise ValueError(f"Unknown seed_ids: {unknown}")
        seeds = [s for s in seeds if s.seed_id in set(wanted)]
    if max_seeds is not None:
        seeds = seeds[:max_seeds]
    if not seeds:
        raise ValueError(
            f"No seeds to run (category={category!r}, seed_ids={seed_ids!r})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = BenchReport(run_id=run_id)
    manifest_path = output_dir / MANIFEST_NAME

    for i, seed in enumerate(seeds, 1):
        result = SeedResult(seed=seed)
        started = time.monotonic()
        logger.info(f"[bench] ({i}/{len(seeds)}) {seed.seed_id} [{seed.category}]")

        # 1. Generate
        try:
            produced = await generate(seed, output_dir)
        except GenerationError as exc:
            result.status = "errored"
            result.generation_error = str(exc)
            # A run that burned an hour before timing out is the most expensive
            # seed in the run; reporting it as free hides exactly the cost worth
            # seeing, and its turn count is the only clue to why it failed.
            result.metadata = dict(exc.metadata)
            result.duration_seconds = time.monotonic() - started
            logger.error(f"[bench] generation failed for {seed.seed_id}: {exc}")
            report.results.append(result)
            _write_manifest(report, manifest_path, manifest_label, manifest_source)
            continue
        except Exception as exc:
            result.status = "errored"
            result.generation_error = str(exc)
            result.duration_seconds = time.monotonic() - started
            logger.error(f"[bench] generation failed for {seed.seed_id}: {exc}")
            report.results.append(result)
            _write_manifest(report, manifest_path, manifest_label, manifest_source)
            continue

        _check_seed(produced, seed.seed_id)
        result.status = produced.status
        result.metadata = dict(produced.metadata)
        result.video_path = produced.video_path

        if produced.status == "skipped":
            # No video means nothing to judge. Handing the judge a missing path
            # would earn a permissive default verdict — a fabricated score for a
            # video that does not exist, dragged straight into the mean.
            result.duration_seconds = time.monotonic() - started
            logger.info(f"[bench] no video for {seed.seed_id}, skipping the judge")
            report.results.append(result)
            _write_manifest(report, manifest_path, manifest_label, manifest_source)
            continue

        # 2. Judge
        result.verdict = await judge.judge(seed, result.video_path)
        # A generator that knows what it really cost overrides our stopwatch: for
        # an imported video the generation happened elsewhere, so timing this
        # process measures nothing.
        result.duration_seconds = (
            produced.duration_seconds
            if produced.duration_seconds is not None
            else time.monotonic() - started
        )
        report.results.append(result)
        _write_manifest(report, manifest_path, manifest_label, manifest_source)

    logger.info(f"[bench] done: {report.summary()}")
    return report


def _check_seed(produced: GenerationResult, seed_id: str) -> None:
    """
    The generator must hand back a result for the seed it was asked about.

    Cheap, and it catches the failure mode of a manifest-driven importer: one
    copy-pasted seed_id would otherwise grade one video against another seed's
    rubric and report a perfectly plausible score for it.
    """
    if produced.seed_id != seed_id:
        raise ValueError(
            f"generator returned a result for {produced.seed_id!r} "
            f"while generating {seed_id!r}"
        )


def _write_manifest(
    report: BenchReport, path: Path, label: str, source: str
) -> None:
    """
    Rewrite the run's video manifest after each seed.

    Per seed rather than once at the end, because the runs worth replaying are
    exactly the ones that do not finish: an agentic run killed at seed five has
    already paid for four videos, and a manifest written only on a clean exit
    would strand them. Best-effort — bookkeeping must never fail a run whose
    videos were produced fine.
    """
    try:
        write_manifest(report.results, path, label=label, source=source)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[bench] could not write {path}: {exc}")
