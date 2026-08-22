import logging
from typing import Dict, List, Optional
from pathlib import Path

import yaml

from video_eval_bench.dataset.dataset_schemas import (
    DEFAULT_DATASET_DIR,
    Dataset,
    Rubric,
    Category,
    SafetyCheck,
)
from video_eval_bench.dataset.seed import Seed

logger = logging.getLogger(__name__)


def load_dataset(dataset_dir: Optional[Path] = None) -> Dataset:
    """Load the full dataset; validates that every seed references a known genre."""
    ds = Dataset(
        rubric_a=load_rubric_a(dataset_dir),
        rubric_b=load_rubric_b(dataset_dir),
        categories=load_categories(dataset_dir),
        safety_checks=load_safety_checks(dataset_dir),
        seeds=load_seeds(dataset_dir),
    )
    unknown = {s.category for s in ds.seeds} - set(ds.categories)
    if unknown:
        raise ValueError(f"Seeds reference unknown genres: {sorted(unknown)}")
    return ds



def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_rubric(path: Path) -> Rubric:
    data = _load_yaml(path)
    return Rubric(
        section=data["section"],
        title=data.get("title", data["section"]),
        criteria=[
            {
                "id": c["id"],
                "name": c.get("name", c["id"]),
                "description": c.get("description", ""),
                "weight": c["weight"],
                "critical": c.get("critical", False),
            }
            for c in data["criteria"]
        ],
        notes=data.get("notes", ""),
    )


def load_rubric_a(dataset_dir: Optional[Path] = None) -> Rubric:
    """Section A — Universal Technical Baseline (10 points)."""
    return _load_rubric((dataset_dir or DEFAULT_DATASET_DIR) / "rubric_a.yaml")


def load_rubric_b(dataset_dir: Optional[Path] = None) -> Rubric:
    """Section B — Semantic & Cultural Fidelity (10 points)."""
    return _load_rubric((dataset_dir or DEFAULT_DATASET_DIR) / "rubric_b.yaml")


def load_safety_checks(dataset_dir: Optional[Path] = None) -> List[SafetyCheck]:
    """Section D — safety veto checks."""
    data = _load_yaml((dataset_dir or DEFAULT_DATASET_DIR) / "rubric_d.yaml")
    return [
        SafetyCheck(
            id=c["id"],
            category=c.get("category", c["id"]),
            description=c.get("description", ""),
            veto=c.get("veto", True),
        )
        for c in data.get("checks", [])
    ]


def load_categories(dataset_dir: Optional[Path] = None) -> Dict[str, Category]:
    """Section C — genre rubrics, keyed by genre key."""
    data = _load_yaml((dataset_dir or DEFAULT_DATASET_DIR) / "rubric_c.yaml")
    categories: Dict[str, Category] = {}
    for entry in data.get("genres", []):
        cat = Category(
            key=entry["key"],
            name=entry.get("name", entry["key"]),
            rubric=Rubric(
                section=f"C:{entry['key']}",
                title=entry.get("name", entry["key"]),
                criteria=[
                    {
                        "id": c["id"],
                        "name": c.get("name", c["id"]),
                        "description": c.get("description", ""),
                        "weight": c["weight"],
                        "critical": c.get("critical", False),
                    }
                    for c in entry["criteria"]
                ],
                notes=entry.get("focus", ""),
            ),
        )
        categories[cat.key] = cat
    logger.info(f"Loaded {len(categories)} genre rubrics")
    return categories


def load_seeds(
    dataset_dir: Optional[Path] = None,
    category: Optional[str] = None,
) -> List[Seed]:
    """Load seeds, optionally filtered to one genre."""
    data = _load_yaml((dataset_dir or DEFAULT_DATASET_DIR) / "seeds.yaml")
    seeds = [Seed(**s) for s in data.get("seeds", [])]
    if category:
        seeds = [s for s in seeds if s.category == category]
    logger.info(f"Loaded {len(seeds)} seeds" + (f" (category={category})" if category else ""))
    return seeds
