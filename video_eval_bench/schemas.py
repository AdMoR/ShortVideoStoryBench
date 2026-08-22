"""
Schemas for the video evaluation benchmark.

Scoring model (normalized to 0-100):
    Section A — Universal Technical Baseline   (applied to every video)
    Section B — Semantic & Cultural Fidelity   (applied to every video)
    Section C — Genre-Specific Criteria        (one genre rubric per video)
    Section D — Safety                         (binary veto checks, not scored)

    Each section's points earned are divided by that section's total weight
    and scaled to 0-100. The overall score is the mean of the three section
    percentages. Rubrics do NOT need to sum to any fixed number.

Dataset:
    Seed      — one benchmark case: a generation prompt + genre, belonging to a
                category (genre)
    Category  — a genre: name + its Section C rubric

Judge:
    RubricCriterion — one binary-check criterion (id, weight, critical flag)
    Rubric          — a section's criteria (A, B, or one genre's C)
    SafetyCheck     — one Section D veto check
    JudgeScore      — per-criterion score (0/1 pass-fail × weight)
    JudgeVerdict    — the judge's structured output for one video
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from video_eval_bench.dataset.dataset_schemas import Category, Rubric, RubricCriterion, SafetyCheck
from video_eval_bench.dataset.seed import Seed


class SafetyResult(BaseModel):
    """Outcome of one safety check."""

    check_id: str
    violation: bool = Field(description="True if the video violates this check")
    comment: str = Field(default="", description="One-sentence justification")


# ── Judge output ──────────────────────────────────────────────────────────────


class JudgeScore(BaseModel):
    """Score for one rubric criterion (binary check × weight)."""

    criterion: str = Field(description="Criterion id, e.g. 'U1'")
    passed: bool = Field(description="True if the binary check passes")
    score: float = Field(ge=0.0, description="Points earned (weight if passed, else 0)")
    comment: str = Field(default="", description="One-sentence justification")


class JudgeVerdict(BaseModel):
    """Structured judgement of one generated video against the full rubric."""

    seed_id: str
    category: str
    # Section scores: percentage of points earned in each section (0-100)
    section_a: float = Field(ge=0.0, le=100.0, description="Section A score, 0-100")
    section_b: float = Field(ge=0.0, le=100.0, description="Section B score, 0-100")
    section_c: float = Field(ge=0.0, le=100.0, description="Section C score, 0-100")
    total_score: float = Field(
        ge=0.0, le=100.0,
        description="Mean of the three section percentages, 0-100",
    )
    scores: List[JudgeScore] = Field(default_factory=list)
    safety: List[SafetyResult] = Field(default_factory=list)
    safety_veto: bool = Field(description="True if any Section D check was violated")
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

    @classmethod
    def permissive_default(cls, seed_id: str, category: str, error: str) -> "JudgeVerdict":
        """Fallback verdict when the judge fails — never blocks a benchmark run."""
        return cls(
            seed_id=seed_id,
            category=category,
            section_a=50.0,
            section_b=50.0,
            section_c=50.0,
            total_score=50.0,
            scores=[],
            safety=[],
            safety_veto=False,
            passed=True,
            reasoning="Judge unavailable — video accepted by default.",
            judge_error=error,
        )
