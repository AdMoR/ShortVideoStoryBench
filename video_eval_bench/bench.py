"""
Benchmark runner: for each seed, generate a video (via an injectable
generator) and grade it with the judge.

The generator is a callable so this module stays decoupled from any specific
video model:

    async def generate(seed: Seed) -> str  # returns local video path

In mock mode, `MockGenerator` writes a tiny synthetic video so the whole
pipeline runs offline.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from video_eval_bench.dataset import Dataset, load_dataset
from video_eval_bench.generator.base import GenerateFn
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

    for i, seed in enumerate(seeds, 1):
        result = SeedResult(seed=seed)
        started = time.monotonic()
        logger.info(f"[bench] ({i}/{len(seeds)}) {seed.seed_id} [{seed.category}]")

        # 1. Generate
        try:
            result.video_path = await generate(seed, output_dir)
        except Exception as exc:
            result.generation_error = str(exc)
            result.metadata = _generator_metadata(generate, seed.seed_id)
            logger.error(f"[bench] generation failed for {seed.seed_id}: {exc}")
            report.results.append(result)
            continue
        result.metadata = _generator_metadata(generate, seed.seed_id)

        # 2. Judge
        result.verdict = await judge.judge(seed, result.video_path)
        result.duration_seconds = time.monotonic() - started
        report.results.append(result)

    logger.info(f"[bench] done: {report.summary()}")
    return report


def _generator_metadata(generate: GenerateFn, seed_id: str) -> dict:
    """
    Whatever the generator wants to record about this seed, if it offers any.

    Deliberately best-effort: a generator's bookkeeping must never be the thing
    that fails a run whose video was produced fine.
    """
    getter = getattr(generate, "metadata_for", None)
    if getter is None:
        return {}
    try:
        return dict(getter(seed_id) or {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[bench] generator metadata for {seed_id} failed: {exc}")
        return {}
