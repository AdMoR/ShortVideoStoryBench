"""
Score videos that were generated somewhere else.

This generator looks a seed up in a manifest,
checks the file is one and places it in the run directory. The

Two behaviours that matter, and are deliberately different:

  * a seed the manifest does not mention is **skipped**
  * a seed the manifest *does* mention whose file is unreadable is an **error**.
"""

import logging
import os
import shutil
from pathlib import Path

from video_eval_bench.dataset.seed import Seed
from video_eval_bench.generator.base import GenerationError, GenerationResult
from video_eval_bench.generator.manifest import Manifest, load_manifest
from video_eval_bench.judge.frames import video_frame_count

logger = logging.getLogger(__name__)

# What `submit_video` accepts and what the HTML report knows how to embed.
VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi"})


class ExternalGenerator:
    """Resolve each seed's video from a manifest instead of generating one."""

    def __init__(self, manifest: Path, copy: bool = True, label: str = ""):
        # The manifest holds only what the file said; where it was read from is
        # ours to remember, and it anchors every relative path in it.
        self.path = Path(manifest).expanduser()
        self.manifest: Manifest = load_manifest(self.path)
        self.copy = copy
        self.label = label or self.manifest.label

    async def __call__(self, seed: Seed, output_dir: Path) -> GenerationResult:
        entry = self.manifest.get(seed.seed_id)
        if entry is None:
            logger.info(f"[ExternalGenerator] {seed.seed_id}: not in the manifest, skipping")
            return GenerationResult(
                seed_id=seed.seed_id,
                status="skipped",
                metadata={
                    "skip_reason": f"no entry for {seed.seed_id!r} in {self.path}",
                    "manifest": str(self.path),
                },
            )

        source = entry.path
        metadata = entry.as_metadata(self.path)

        if not source.exists():
            raise GenerationError(
                f"{self.path}: {seed.seed_id} points at {source}, which does not exist",
                metadata=metadata,
            )

        frames = video_frame_count(str(source))
        if frames <= 0:
            raise GenerationError(
                f"{source} is not a readable video (OpenCV found no frames)",
                metadata=metadata,
            )

        destination = self._place(source, Path(output_dir), seed.seed_id)
        metadata.update({"frames": frames, "bytes": destination.stat().st_size})

        logger.info(
            f"[ExternalGenerator] {seed.seed_id}: {frames} frames from {source} "
            f"-> {destination.name}"
        )
        return GenerationResult(
            seed_id=seed.seed_id,
            video_path=str(destination),
            metadata=metadata,
            duration_seconds=entry.duration_seconds,
        )

    def _place(self, source: Path, output_dir: Path, seed_id: str) -> Path:
        """
        Put the video in the run directory as `<seed_id><suffix>`.
        """
        suffix = source.suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise GenerationError(
                f"{source} has an unsupported extension {suffix!r}; "
                f"expected one of {sorted(VIDEO_SUFFIXES)}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / f"{seed_id}{suffix}"
        if destination.exists() or destination.is_symlink():
            destination.unlink()

        if self.copy:
            shutil.copy2(source, destination)
        else:
            os.symlink(source.resolve(), destination)
        return destination
