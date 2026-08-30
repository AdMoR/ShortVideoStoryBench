"""
Dataset package: seeds + the rubric library, loaded from YAML files.

Re-exports the schema types (Seed, SeedReference, Dimension, RubricCriterion,
RubricLibrary, SafetyCheck, Dataset) and the load_*() loaders so callers can do:

    from video_eval_bench.dataset import load_dataset
"""

from video_eval_bench.dataset.seed import (
    CriterionVerification,
    Seed,
    SeedCriterion,
    SeedJudgeVerdict,
    SeedProvenance,
    SeedReference,
)
from video_eval_bench.dataset.dataset_schemas import (
    DEFAULT_DATASET_DIR,
    Dataset,
    Dimension,
    RubricCriterion,
    RubricLibrary,
    SafetyCheck,
    TagFacet,
    TagVocabulary,
    bind_description,
)
from video_eval_bench.dataset.dataset_utils import (
    load_dataset,
    load_tags,
    load_genres,
    load_rubrics,
    load_safety_checks,
    load_seeds,
)

__all__ = [
    "Seed",
    "SeedCriterion",
    "SeedJudgeVerdict",
    "CriterionVerification",
    "SeedProvenance",
    "SeedReference",
    "Dataset",
    "Dimension",
    "RubricCriterion",
    "RubricLibrary",
    "SafetyCheck",
    "TagFacet",
    "TagVocabulary",
    "bind_description",
    "DEFAULT_DATASET_DIR",
    "load_dataset",
    "load_tags",
    "load_genres",
    "load_rubrics",
    "load_safety_checks",
    "load_seeds",
]
