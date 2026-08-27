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
                "requires_references": c.get("requires_references", False),
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
                        "requires_references": c.get("requires_references", False),
                    }
                    for c in entry["criteria"]
                ],
                notes=entry.get("focus", ""),
            ),
        )
        categories[cat.key] = cat
    logger.info(f"Loaded {len(categories)} genre rubrics")
    return categories


def _resolve_references(seed: Seed, dataset_dir: Path) -> None:
    """
    Make a seed's reference paths absolute, and refuse a seed we cannot honour.

    This is the only place that knows both the seed and the dataset directory, so
    it is where relative paths become openable ones. The two checks fail the
    whole load rather than the seed, matching how `load_dataset` treats an
    unknown genre: a dataset either loads whole or fails before any generation is
    paid for.

    Duplicate ids are rejected because the id *is* the filename the generator
    stages the image under — two references called `maya` would silently become
    one image, and the brief would describe an image the agent never got.
    """
    seen: set[str] = set()
    for ref in seed.references:
        if ref.id in seen:
            raise ValueError(
                f"seed {seed.seed_id!r} has two references with id {ref.id!r}; "
                "ids are used as filenames and must be unique within a seed"
            )
        seen.add(ref.id)
        ref.path = (dataset_dir / ref.path).resolve()
        if not ref.path.is_file():
            raise FileNotFoundError(
                f"seed {seed.seed_id!r} reference {ref.id!r}: no such file: {ref.path}"
            )


def load_seeds(
    dataset_dir: Optional[Path] = None,
    category: Optional[str] = None,
) -> List[Seed]:
    """Load seeds, optionally filtered to one genre."""
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    data = _load_yaml(dataset_dir / "seeds.yaml")
    seeds = [Seed(**s) for s in data.get("seeds", [])]
    for seed in seeds:
        _resolve_references(seed, dataset_dir)
    if category:
        seeds = [s for s in seeds if s.category == category]
    logger.info(f"Loaded {len(seeds)} seeds" + (f" (category={category})" if category else ""))
    return seeds
