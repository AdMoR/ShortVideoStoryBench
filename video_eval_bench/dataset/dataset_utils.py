import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from video_eval_bench.dataset.dataset_schemas import (
    DEFAULT_DATASET_DIR,
    Dataset,
    RubricLibrary,
    SafetyCheck,
    TagVocabulary,
    bind_description,
)
from video_eval_bench.dataset.seed import Seed

logger = logging.getLogger(__name__)


def load_dataset(dataset_dir: Optional[Path] = None) -> Dataset:
    """
    Load the full dataset.

    Validates that every seed names a known genre and only criteria the library
    defines. Both fail the whole load rather than the seed: a dataset either
    loads whole or fails before any generation is paid for, and a seed silently
    dropped for a typo would shrink the benchmark without saying so.
    """
    ds = Dataset(
        rubrics=load_rubrics(dataset_dir),
        genres=load_genres(dataset_dir),
        safety_checks=load_safety_checks(dataset_dir),
        seeds=load_seeds(dataset_dir),
        tags=load_tags(dataset_dir),
    )
    unknown_genres = sorted({s.category for s in ds.seeds} - set(ds.genres))
    if unknown_genres:
        raise ValueError(f"Seeds reference unknown genres: {unknown_genres}")
    for seed in ds.seeds:
        _validate_seed_criteria(seed, ds)
        _validate_seed_tags(seed, ds)
    unscored = [s.seed_id for s in ds.seeds if not s.scored_criteria()]
    if unscored:
        # Not a load failure: a generated seed whose every criterion the policy
        # excluded is a build result worth keeping and reading, not a broken file.
        # It is dropped from the benchmark because scoring it would be 0/0 — which
        # `JudgeVerdict` would publish as a clean pass.
        logger.warning(
            f"{len(unscored)} seed(s) have no scored criteria and will not be "
            f"benchmarked: {unscored[:10]}"
        )
        ds.seeds = [s for s in ds.seeds if s.scored_criteria()]
    logger.info(
        f"Loaded {len(ds.seeds)} seeds, {len(ds.rubrics.criteria)} criteria, "
        f"{len(ds.safety_checks)} safety checks"
    )
    return ds


def _validate_seed_criteria(seed: Seed, ds: Dataset) -> None:
    """
    Every criterion a seed names exists — in the library or on the seed itself — and
    binds exactly what that criterion asks.

    Both are whole-load failures, matching how an unknown genre is treated: a
    dataset either loads whole or fails before any generation is paid for. A bad
    binding in particular must not survive to judge time, where it would surface as
    a permissive default and read as though the video had been graded.
    """
    local = seed.local_criteria_by_id()
    unknown = sorted(c.id for c in seed.rubrics if c.id not in ds.rubrics and c.id not in local)
    if unknown:
        raise ValueError(
            f"seed {seed.seed_id!r} lists criteria that are neither in the rubric "
            f"library nor among its own `local_criteria`: {unknown}"
        )
    shadowed = sorted(cid for cid in local if cid in ds.rubrics)
    if shadowed:
        raise ValueError(
            f"seed {seed.seed_id!r} defines local criteria whose ids the library "
            f"already uses: {shadowed}. A local id must be namespaced to its seed "
            f"(e.g. {seed.seed_id}.LANG1) so a report column can never mean two "
            f"different questions."
        )
    duplicates = sorted({c.id for c in seed.rubrics if seed.criterion_ids().count(c.id) > 1})
    if duplicates:
        raise ValueError(f"seed {seed.seed_id!r} lists criteria twice: {duplicates}")
    unused = sorted(set(local) - {c.id for c in seed.rubrics})
    if unused:
        raise ValueError(
            f"seed {seed.seed_id!r} defines local criteria it does not list under "
            f"`rubrics`: {unused}. A definition nothing references is never judged."
        )
    known_dimensions = {d.key for d in ds.rubrics.dimensions}
    bad = sorted(c.id for c in local.values() if c.dimension not in known_dimensions)
    if bad:
        raise ValueError(
            f"seed {seed.seed_id!r}: local criteria reference unknown dimensions: {bad}"
        )
    for entry in seed.rubrics:
        criterion = local.get(entry.id) or ds.rubrics.get(entry.id)
        undeclared = sorted(set(entry.bind) - set(criterion.binds))
        if undeclared:
            raise ValueError(
                f"seed {seed.seed_id!r} binds {undeclared} on criterion "
                f"{entry.id!r}, which declares binds {criterion.binds}"
            )
        bind_description(criterion, entry.bind)  # raises on an unbound placeholder


def _validate_seed_tags(seed: Seed, ds: Dataset) -> None:
    """Tags come from the dataset's closed vocabulary, when it declares one."""
    if ds.tags is None:
        return
    for problem in ds.tags.problems(seed.tags):
        raise ValueError(f"seed {seed.seed_id!r}: {problem}")


def load_tags(dataset_dir: Optional[Path] = None) -> Optional[TagVocabulary]:
    """
    The seed tag vocabulary, or None for a dataset that declares none.

    Optional because the hand-written dataset carries no tags; a generated one does,
    and there the closed vocabulary is the point — these are the columns every
    cross-seed analysis groups by, so an invented value must fail the build rather
    than quietly split one column into three.
    """
    path = (dataset_dir or DEFAULT_DATASET_DIR) / "tags.yaml"
    if not path.exists():
        return None
    data = _load_yaml(path)
    return TagVocabulary(
        facets=data.get("facets", []),
        multi=data.get("multi", []),
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_rubrics(dataset_dir: Optional[Path] = None) -> RubricLibrary:
    """
    The criterion library — every check the benchmark knows how to make.

    The file is a list of `sections`, generic first, plus the closed `criterion_tags`
    vocabulary. Both are analytic axes and select nothing.

    A flat `criteria:` list is refused rather than tolerated. It would load as a
    library with no categories and no tags, and every report that groups by either
    would quietly come back empty — a dataset that loads and reports nothing is worse
    than one that fails.
    """
    path = (dataset_dir or DEFAULT_DATASET_DIR) / "rubrics.yaml"
    data = _load_yaml(path)
    if "sections" not in data:
        raise ValueError(
            f"{path} has no `sections:` key. The rubric library is organised into "
            f"scoped sections (general first, then genre/tag-conditioned); a flat "
            f"`criteria:` list is the old shape and is no longer loadable."
        )
    return RubricLibrary(
        dimensions=data.get("dimensions", []),
        criterion_tags=data.get("criterion_tags", {}),
        sections=data.get("sections", []),
    )


def load_genres(dataset_dir: Optional[Path] = None) -> Dict[str, str]:
    """Genre key -> display name. A reporting label; it selects no rubric."""
    data = _load_yaml((dataset_dir or DEFAULT_DATASET_DIR) / "genres.yaml")
    return {g["key"]: g.get("name", g["key"]) for g in data.get("genres", [])}


def load_safety_checks(dataset_dir: Optional[Path] = None) -> List[SafetyCheck]:
    """The safety veto checks, applied to every video regardless of seed."""
    data = _load_yaml((dataset_dir or DEFAULT_DATASET_DIR) / "safety.yaml")
    return [SafetyCheck(**c) for c in data.get("checks", [])]


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
    """
    Load seeds, optionally filtered to one genre.

    Criterion ids are *not* validated here — that needs the library, so
    `load_dataset` does it. Loading seeds alone is for callers that only want
    the briefs.
    """
    dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
    data = _load_yaml(dataset_dir / "seeds.yaml")
    seeds = [Seed(**s) for s in data.get("seeds", [])]
    for seed in seeds:
        _resolve_references(seed, dataset_dir)
    if category:
        seeds = [s for s in seeds if s.category == category]
    logger.info(f"Loaded {len(seeds)} seeds" + (f" (category={category})" if category else ""))
    return seeds
