"""
The video manifest: a list of seed_id -> video file, with provenance.

One format, read and written by this module, used in both directions:

  * read  — `ExternalGenerator` scores videos produced outside the harness.
  * write — every run emits one beside its report, so the run can be pushed back
            through a different judge without repeating the generation. That is
            the expensive half: an agentic seed can take an hour, and re-judging
            it should not cost that again.

Because those are the same format, a run directory *is* a valid manifest, and
replaying a run is just pointing the importer at the manifest the run wrote.

Paths are relative to the manifest's own directory, never to the cwd. A run
folder has to stay movable as a unit (the HTML report already references its
videos by bare filename), and hydra runs with `chdir: false` so the cwd is not a
stable anchor. The anchor reaches the model through pydantic's validation
context, since a manifest cannot resolve its own paths without knowing where it
was read from.

    label: gx10-qwen27b-director
    videos:
      - seed_id: marketing_001
        path: marketing_001.mp4
        prompt: "..."               # the prompt actually used, if it differs
        duration_seconds: 1320      # what generation really cost, elsewhere
        source: "model=..., skills=..."
        metadata: {turns: 6, usage: {...}}
"""

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

MANIFEST_NAME = "videos.yaml"


class ManifestEntry(BaseModel):
    """One video in a manifest, with its path already resolved."""

    # As everywhere else in the bench: an unknown key is a typo, and a typo that
    # silently does nothing produces a plausible-looking result attributed to the
    # wrong thing. The only writer of this format is `write_manifest` below.
    model_config = ConfigDict(extra="forbid")

    seed_id: str
    path: Path
    prompt: Optional[str] = None
    duration_seconds: Optional[float] = None
    source: str = ""
    metadata: dict = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def _expand_user(cls, path: Path) -> Path:
        return path.expanduser()

    @field_validator("prompt")
    @classmethod
    def _blank_is_no_prompt(cls, prompt: Optional[str]) -> Optional[str]:
        """An empty prompt is no prompt — the report would show a blank brief."""
        return prompt or None

    @classmethod
    def from_result(cls, result, source: str = "") -> "ManifestEntry":
        """The manifest entry describing a SeedResult's video."""
        metadata = dict(result.metadata or {})
        # The generator's own cost, not the bench's generate+judge measure — a
        # replay should carry forward what generation actually took.
        duration = metadata.get("duration_seconds")
        if duration is None and result.duration_seconds:
            duration = round(result.duration_seconds, 2)
        return cls(
            seed_id=result.seed.seed_id,
            # The bare filename: the run directory must stay movable as a unit.
            path=Path(result.video_path).name,
            prompt=metadata.get("prompt"),
            duration_seconds=duration,
            source=source,
            metadata=metadata,
        )

    def as_metadata(self, manifest_path: Path) -> dict:
        """
        What the report should record about this video's origin.

        `prompt` is included when the entry carries one because the HTML report
        prefers it over the dataset brief — so the report shows the prompt that
        actually produced the video, not the one the bench would have used.
        """
        meta = dict(self.metadata)
        meta.update(
            self.model_dump(
                mode="json",
                exclude={"metadata", "path"},
                exclude_none=True,
                exclude_defaults=True,
            )
        )
        meta["source_path"] = str(self.path)
        meta["manifest"] = str(manifest_path)
        return meta


class Manifest(BaseModel):
    """A parsed manifest: its label and its videos, in file order."""

    model_config = ConfigDict(extra="forbid")

    label: str = ""
    videos: List[ManifestEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_seeds_and_resolved_paths(self, info: ValidationInfo) -> "Manifest":
        base = (info.context or {}).get("base")
        seen = set()
        for entry in self.videos:
            if entry.seed_id in seen:
                # Two videos claiming one seed means one of them is silently
                # ignored, and which one would depend on file order. Say so.
                raise ValueError(f"duplicate entry for seed {entry.seed_id!r}")
            seen.add(entry.seed_id)
            if base is not None and not entry.path.is_absolute():
                entry.path = Path(base) / entry.path
        return self

    def get(self, seed_id: str) -> Optional[ManifestEntry]:
        for entry in self.videos:
            if entry.seed_id == seed_id:
                return entry
        return None

    @property
    def seed_ids(self) -> List[str]:
        return [entry.seed_id for entry in self.videos]


def load_manifest(path: Path) -> Manifest:
    """Parse a manifest, resolving every relative path against its directory."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}

    try:
        manifest = Manifest.model_validate(data, context={"base": path.parent})
    except ValidationError as exc:
        # Pydantic names the field but not the file, and a manifest is usually
        # one of several — the path is half the diagnosis.
        raise ValueError(f"{path}: {_explain(exc, data)}") from exc

    logger.info(f"Loaded {len(manifest.videos)} video(s) from {path}")
    return manifest


def write_manifest(
    results: Iterable,
    path: Path,
    label: str = "",
    source: str = "",
) -> Path:
    """
    Write the videos of `results` (SeedResults) as a manifest at `path`.

    Only seeds that actually have a video are listed — a manifest describes what
    exists, so skipped and errored seeds are omitted and become skips again on a
    replay, which keeps the seed count honest rather than inventing failures.

    Written atomically: this is rewritten after every seed so that a run killed
    partway still leaves the videos it did produce replayable, and a half-written
    manifest at that moment would defeat the purpose.
    """
    path = Path(path)
    manifest = Manifest(
        label=label,
        videos=[
            ManifestEntry.from_result(result, source=source)
            for result in results
            if result.video_path
        ],
    )

    # `exclude_defaults` keeps the file to what an entry actually carries; the
    # two top-level keys are restored so the shape stays readable when empty.
    payload = manifest.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    payload.setdefault("label", "")
    payload.setdefault("videos", [])

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    os.replace(tmp, path)
    return path


def _explain(exc: ValidationError, data: dict) -> str:
    """Render a ValidationError against the manifest's own vocabulary."""
    raw_videos = data.get("videos") or []
    parts = []
    for error in exc.errors():
        loc = error["loc"]
        if len(loc) >= 3 and loc[0] == "videos" and isinstance(loc[1], int):
            raw = raw_videos[loc[1]] if loc[1] < len(raw_videos) else {}
            who = (raw or {}).get("seed_id") or f"#{loc[1]}"
            parts.append(f"entry {who!r}: {'.'.join(str(p) for p in loc[2:])}: {error['msg']}")
        else:
            parts.append(error["msg"])
    return "; ".join(parts)
