from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from video_eval_bench.dataset.seed import Seed
from video_eval_bench.schemas import JudgeVerdict


@dataclass
class SeedResult:
    """
    Outcome of one seed in a benchmark run.

    `status` is a field rather than a key in `metadata` because it drives the
    summary counts and a `veb-compare` column, and `metadata` is generator-defined
    and best-effort — nothing that has to be counted can live in there.

    "skipped" means no video was supplied for this seed, which only an import can
    say. It is deliberately not an error: an import covering three of eight seeds
    has not failed five times.
    """

    seed: Seed
    status: Literal["completed", "skipped", "errored"] = "completed"
    video_path: Optional[str] = None
    generation_error: Optional[str] = None
    duration_seconds: float = 0.0
    verdict: Optional[JudgeVerdict] = None
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if generation succeeded and the video was judged."""
        return self.generation_error is None and self.verdict is not None


@dataclass
class BenchReport:
    """
    Aggregated results of a benchmark run.

    A report is self-describing on purpose: `variant`, `choices` and `config`
    record *which* configuration produced these numbers, so `veb-compare` can
    label and align reports from runs made days apart without being told
    anything about how they were launched.
    """

    run_id: str
    results: List[SeedResult] = field(default_factory=list)
    variant: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    note: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def by_category(self) -> Dict[str, List[SeedResult]]:
        out: Dict[str, List[SeedResult]] = {}
        for r in self.results:
            out.setdefault(r.seed.category, []).append(r)
        return out

    def summary(self) -> dict:
        per_cat = {}
        for cat, rs in self.by_category().items():
            scored = [r.verdict.total_score for r in rs if r.verdict is not None]
            per_cat[cat] = {
                "n_seeds": len(rs),
                "n_judged": len(scored),
                "n_skipped": sum(1 for r in rs if r.status == "skipped"),
                "mean_score": round(sum(scored) / len(scored), 2) if scored else None,
                "n_passed": sum(1 for r in rs if r.verdict and r.verdict.passed),
                "n_safety_vetoes": sum(
                    1 for r in rs if r.verdict and r.verdict.safety_veto
                ),
            }
        scored_all = [r.verdict.total_score for r in self.results if r.verdict is not None]
        return {
            "run_id": self.run_id,
            "variant": self.variant,
            "n_seeds": len(self.results),
            "n_ok": sum(1 for r in self.results if r.ok),
            "n_generation_errors": sum(1 for r in self.results if r.generation_error),
            "n_skipped": sum(1 for r in self.results if r.status == "skipped"),
            "n_judge_errors": sum(
                1 for r in self.results if r.verdict is not None and r.verdict.judge_error
            ),
            "mean_score": round(sum(scored_all) / len(scored_all), 2) if scored_all else None,
            "n_passed": sum(1 for r in self.results if r.verdict and r.verdict.passed),
            "n_safety_vetoes": sum(
                1 for r in self.results if r.verdict and r.verdict.safety_veto
            ),
            "total_duration_seconds": round(
                sum(r.duration_seconds for r in self.results), 2
            ),
            "per_category": per_cat,
        }

    def to_json(self) -> dict:
        return {
            "summary": self.summary(),
            "variant": self.variant,
            "choices": self.choices,
            "note": self.note,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "config": self.config,
            "results": [
                {
                    "seed_id": r.seed.seed_id,
                    "category": r.seed.category,
                    "status": r.status,
                    "video_path": r.video_path,
                    "generation_error": r.generation_error,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "verdict": r.verdict.model_dump() if r.verdict else None,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }
