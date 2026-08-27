"""
Dataset package: seeds + rubrics (Sections A/B/C/D) loaded from YAML files.

Re-exports the schema types (Seed, SeedReference, Category, Rubric,
RubricCriterion, SafetyCheck, Dataset) and the load_*() loaders so callers can do:

    from video_eval_bench.dataset import load_dataset
"""

from video_eval_bench.dataset.seed import Seed, SeedReference
from video_eval_bench.dataset.dataset_schemas import (
    DEFAULT_DATASET_DIR,
    Category,
    Dataset,
    Rubric,
    RubricCriterion,
    SafetyCheck,
)
from video_eval_bench.dataset.dataset_utils import (
    load_categories,
    load_dataset,
    load_rubric_a,
    load_rubric_b,
    load_safety_checks,
    load_seeds,
)

__all__ = [
    "Seed",
    "SeedReference",
    "Category",
    "Dataset",
    "Rubric",
    "RubricCriterion",
    "SafetyCheck",
    "DEFAULT_DATASET_DIR",
    "load_categories",
    "load_dataset",
    "load_rubric_a",
    "load_rubric_b",
    "load_safety_checks",
    "load_seeds",
]
