"""
Dataset loading: seeds + rubrics from YAML files.

Layout:
    dataset/
        rubric_a.yaml   — Section A: Universal Technical Baseline (10 pts)
        rubric_b.yaml   — Section B: Semantic & Cultural Fidelity (10 pts)
        rubric_c.yaml   — Section C: one genre rubric per category (80 pts each)
        rubric_d.yaml   — Section D: safety veto checks
        seeds.yaml      — list of seeds, each referencing a genre key
"""

import logging
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from video_eval_bench.dataset.seed import Seed


logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


class Category(BaseModel):
    """A genre: name + its Section C (genre-specific) rubric."""

    key: str = Field(description="Genre key, e.g. 'entertainment'")
    name: str = Field(description="Human-readable name")
    rubric: "Rubric" = Field(description="Section C (genre-specific) rubric")



class RubricCriterion(BaseModel):
    """One observable (binary) check within a rubric section."""

    id: str = Field(description="Stable id, e.g. 'U1', 'E3', 'M6'")
    name: str = Field(description="Short criterion name")
    description: str = Field(description="What to check, in detail")
    weight: float = Field(gt=0.0, description="Points awarded when the check passes")
    critical: bool = Field(
        default=False,
        description="True for ⚠️ criteria — a failure here caps the section score",
    )


class Rubric(BaseModel):
    """A rubric section: a set of binary-check criteria with a point total."""

    section: str = Field(description="Section label, e.g. 'A', 'B', 'C:entertainment'")
    title: str = Field(description="Human-readable title")
    criteria: List[RubricCriterion] = Field(min_length=1)
    notes: str = Field(default="", description="Section-level judging guidance")

    @property
    def total_points(self) -> float:
        return sum(c.weight for c in self.criteria)

    def score_for(self, scores: dict[str, float]) -> float:
        """Points earned in this section (sum of passed criteria weights)."""
        return sum(s for c, s in scores.items() if c in {x.id for x in self.criteria})

    def critical_failed(self, passed: dict[str, bool]) -> List[str]:
        """Ids of ⚠️ criteria that failed."""
        return [c.id for c in self.criteria if c.critical and not passed.get(c.id, True)]


class SafetyCheck(BaseModel):
    """One Section D safety veto check (binary: violation or not)."""

    id: str = Field(description="Stable id, e.g. 'D1'")
    category: str = Field(description="Safety category, e.g. 'Violence & Gore'")
    description: str = Field(description="The question the judge must answer")
    veto: bool = Field(default=True, description="A violation vetoes the whole video")



@dataclass
class Dataset:
    """The full benchmark dataset: universal rubrics + genre rubrics + safety + seeds."""

    rubric_a: Rubric
    rubric_b: Rubric
    categories: Dict[str, Category]
    safety_checks: List[SafetyCheck]
    seeds: List[Seed]

