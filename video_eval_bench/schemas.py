"""
Schemas for the video evaluation benchmark.

Scoring model (normalized to 0-100):
    A seed names the criteria it is judged on (`Seed.rubrics`, drawn from the
    rubric library). `total_score` is the weight it earned over the weight it
    was asked for — flat across the whole selection, so adding a criterion to a
    seed makes that seed harder rather than re-scaling what came before.

    `dimensions` breaks the same points down by the criteria's reporting
    dimension (consistency, technical, fidelity, structure, craft). It is a view
    of the score, not an input to it: a dimension the seed lists two criteria in
    does not weigh as much as one it lists eight in, which is the point — the
    old fixed sections gave a genre rubric and a three-criterion baseline equal
    say in the total.

    Safety checks are binary vetoes, applied to every video, and are not scored.

Dataset:
    Seed      — one benchmark case: a brief, a genre label, and its rubric list
    Dimension — a reporting group for criteria

Judge:
    RubricCriterion — one binary-check criterion (id, dimension, weight, critical)
    RubricLibrary   — every criterion the benchmark knows
    SafetyCheck     — one safety veto check
    JudgeScore      — per-criterion score (0/1 pass-fail × weight)
    DimensionScore  — one dimension's share of the verdict
    JudgeVerdict    — the judge's structured output for one video
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from video_eval_bench.dataset.dataset_schemas import (
    Dataset,
    Dimension,
    RubricCriterion,
    RubricLibrary,
    SafetyCheck,
)
from video_eval_bench.dataset.seed import Seed, SeedCriterion


class SafetyResult(BaseModel):
    """Outcome of one safety check."""

    check_id: str
    violation: bool = Field(description="True if the video violates this check")
    comment: str = Field(default="", description="One-sentence justification")


# ── Judge output ──────────────────────────────────────────────────────────────


class JudgeScore(BaseModel):
    """Score for one rubric criterion (binary check × weight)."""

    criterion: str = Field(description="Criterion id, e.g. 'SUBJ1'")
    passed: bool = Field(description="True if the binary check passes")
    score: float = Field(ge=0.0, description="Points earned (weight if passed, else 0)")
    comment: str = Field(default="", description="One-sentence justification")


class DimensionScore(BaseModel):
    """One dimension's share of a verdict — a breakdown, not a scoring input."""

    dimension: str = Field(description="Dimension key, e.g. 'consistency'")
    name: str = Field(description="Human-readable dimension name")
    score: float = Field(ge=0.0, le=100.0, description="Percentage of this dimension's weight, 0-100")
    earned: float = Field(ge=0.0, description="Weight earned in this dimension")
    total: float = Field(gt=0.0, description="Weight this seed was asked for in this dimension")


class JudgeVerdict(BaseModel):
    """Structured judgement of one generated video against its seed's rubric."""

    seed_id: str
    category: str
    total_score: float = Field(
        ge=0.0, le=100.0,
        description="Weight earned over weight asked for, across the seed's rubric, 0-100",
    )
    dimensions: List[DimensionScore] = Field(
        default_factory=list,
        description="Per-dimension breakdown, in the rubric library's dimension order",
    )
    critical_failures: List[str] = Field(
        default_factory=list,
        description="Ids of ⚠️ critical criteria the video failed",
    )
    scores: List[JudgeScore] = Field(default_factory=list)
    safety: List[SafetyResult] = Field(default_factory=list)
    safety_veto: bool = Field(description="True if any safety check was violated")
    passed: bool = Field(description="True if the video is acceptable (no veto, score >= threshold)")
    reasoning: str = Field(description="Short overall explanation")
    judge_error: Optional[str] = Field(
        default=None,
        description="Set when the judge itself failed (verdict is then a permissive default)",
    )

    def score_for(self, criterion: str) -> Optional[float]:
        for s in self.scores:
            if s.criterion == criterion:
                return s.score
        return None

    def dimension_score(self, dimension: str) -> Optional[float]:
        for d in self.dimensions:
            if d.dimension == dimension:
                return d.score
        return None

    @classmethod
    def permissive_default(cls, seed_id: str, category: str, error: str) -> "JudgeVerdict":
        """Fallback verdict when the judge fails — never blocks a benchmark run."""
        return cls(
            seed_id=seed_id,
            category=category,
            total_score=50.0,
            dimensions=[],
            critical_failures=[],
            scores=[],
            safety=[],
            safety_veto=False,
            passed=True,
            reasoning="Judge unavailable — video accepted by default.",
            judge_error=error,
        )


__all__ = [
    "Dataset",
    "Dimension",
    "DimensionScore",
    "JudgeScore",
    "JudgeVerdict",
    "RubricCriterion",
    "RubricLibrary",
    "SafetyCheck",
    "SafetyResult",
    "Seed",
    "SeedCriterion",
]
