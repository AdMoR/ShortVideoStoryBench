"""
The judge agent: ties the clip + prompt + LLM backend into a JudgeVerdict.

    judge = VideoJudge(backend=MockBackend(), dataset=load_dataset())
    verdict = await judge.judge(seed, video_path)

The judge is decomposed into one small backend call per rubric criterion and
one per safety check, rather than a single call covering the whole rubric —
this keeps each call's context usage and required output small and
predictable no matter how many criteria a seed lists.

What the model is shown depends on `media`:

    "frames" — N evenly-spaced JPEG frames. Works with any vision model.
    "video"  — the clip itself, once per call, for a backend that accepts video.
               Motion, timing and cut rhythm are judged from the real thing
               instead of inferred from the gaps between frames, which is what
               criteria like CUT1, MOTION1 and TEMP1 actually ask about.

Scoring (normalized to 0-100):
    A seed is judged on the criteria it lists, and only those. `total_score` is
    the weight earned over the weight asked for; the per-dimension breakdown is
    a view of the same points, not a separate scale.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from video_eval_bench.dataset import Dataset
from video_eval_bench.dataset.dataset_schemas import RubricCriterion
from video_eval_bench.dataset.seed import SeedReference
from video_eval_bench.judge.frames import extract_frames, load_image
from video_eval_bench.judge.llm import VisionLLM
from video_eval_bench.judge.prompt import build_criterion_prompt, build_safety_prompt
from video_eval_bench.schemas import (
    DimensionScore,
    JudgeScore,
    JudgeVerdict,
    SafetyResult,
    Seed,
)

logger = logging.getLogger(__name__)

USER_MESSAGE = "Judge the media above against the rubric criterion, as the header describes it."


def parse_verdict_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from a raw LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


class VideoJudge:
    """LLM harness that grades one generated video against its seed's rubric."""

    PASS_THRESHOLD = 60.0  # out of 100

    def __init__(
        self,
        backend: VisionLLM,
        dataset: Dataset,
        n_frames: int = 8,
        media: str = "frames",
    ):
        if media not in ("frames", "video"):
            raise ValueError(f"media must be 'frames' or 'video', got {media!r}")
        self.backend = backend
        self.dataset = dataset
        self.n_frames = n_frames
        self.media = media

    async def judge(self, seed: Seed, video_path: str) -> JudgeVerdict:
        """
        Judge one generated video: one backend call per criterion + per safety
        check, then aggregate into a single verdict.

        Never raises — on any failure returns JudgeVerdict.permissive_default
        so a broken judge can never abort a benchmark run.
        """
        if seed.category not in self.dataset.genres:
            return JudgeVerdict.permissive_default(
                seed.seed_id, seed.category, f"unknown genre {seed.category!r}"
            )
        genre_name = self.dataset.genre_name(seed.category)

        criteria = self.dataset.criteria_for(seed)
        scored = {c.id for c in seed.scored_criteria()}
        missing = sorted(scored - {c.id for c in criteria})
        if missing:
            # load_dataset already refuses this; a judge handed a hand-built seed
            # would otherwise silently grade it on a shorter rubric than it names.
            return JudgeVerdict.permissive_default(
                seed.seed_id, seed.category, f"unknown criteria for seed: {missing}"
            )

        media, n_frames = self._load_media(video_path)
        if media is None:
            return JudgeVerdict.permissive_default(
                seed.seed_id, seed.category, f"no media loaded from {video_path}"
            )

        # References first, clip second, in one flat payload: `VisionLLM.complete`
        # takes an `images` sequence plus an optional `video`, and the prompt
        # header is the only thing telling the model where the references end and
        # the clip begins — so it is built from the references that actually
        # loaded, never from `seed.references`.
        references = self._reference_images(seed)
        images = [data for _, data in references]
        sent = [ref for ref, _ in references]
        if self.media == "video":
            video = media
        else:
            video, images = None, [*images, *media]

        try:
            scores, safety, n_failed = [], [], 0
            for criterion in criteria:
                score, ok = await self._judge_criterion(
                    seed, genre_name, criterion, images, video, n_frames, sent
                )
                scores.append(score)
                n_failed += not ok
            for check in self.dataset.safety_checks:
                result, ok = await self._judge_safety(
                    seed, genre_name, check, images, video, n_frames, sent
                )
                safety.append(result)
                n_failed += not ok

            verdict = self._build_verdict(seed, criteria, scores, safety)
            if n_failed:
                total_calls = len(criteria) + len(self.dataset.safety_checks)
                verdict.judge_error = (
                    f"{n_failed}/{total_calls} per-criterion judge calls failed or "
                    "were unparseable (affected items scored/flagged conservatively)"
                )
            return verdict
        except Exception as exc:
            logger.error(f"[VideoJudge] judge failed for {seed.seed_id}: {exc}", exc_info=True)
            return JudgeVerdict.permissive_default(seed.seed_id, seed.category, str(exc))

    def _load_media(self, video_path: str) -> Tuple[Union[bytes, List[bytes], None], int]:
        """
        The clip, in whichever form this judge sends it, plus the frame count
        the prompt header should quote.

        Returns (None, 0) when there is nothing to send. In video mode the frame
        count is 0 and unused: the header describes a whole clip instead.
        """
        if self.media == "video":
            path = Path(video_path)
            if not path.is_file():
                logger.warning(f"[VideoJudge] video not found: {video_path}")
                return None, 0
            try:
                data = path.read_bytes()
            except OSError as exc:
                logger.warning(f"[VideoJudge] could not read {video_path}: {exc}")
                return None, 0
            if not data:
                logger.warning(f"[VideoJudge] empty video file: {video_path}")
                return None, 0
            logger.info(f"[VideoJudge] sending {len(data) / 1e6:.1f} MB of video per call")
            return data, 0
        frames = extract_frames(video_path, self.n_frames)
        if not frames:
            return None, 0
        return frames, len(frames)

    def _reference_images(self, seed: Seed) -> List[Tuple[SeedReference, bytes]]:
        """
        The seed's references paired with their JPEG bytes, dropping any that
        will not load.

        Dropping rather than raising: a reference the judge cannot open is a
        dataset problem, and failing the whole verdict over it would turn a bad
        image into a missing score. `load_seeds` already refused a reference whose
        file does not exist, so reaching here means an unreadable one.

        The pair is what matters — a dropped image must also vanish from the
        prompt, or "Image 2" in the header names "Image 1" in the payload.
        """
        loaded: List[Tuple[SeedReference, bytes]] = []
        for ref in seed.references:
            data = load_image(str(ref.path))
            if data:
                loaded.append((ref, data))
            else:
                logger.warning(
                    f"[VideoJudge] {seed.seed_id}: reference {ref.id!r} could not be "
                    f"loaded from {ref.path}; judging without it"
                )
        return loaded

    async def _judge_criterion(
        self,
        seed,
        genre_name: str,
        criterion: RubricCriterion,
        images,
        video: Optional[bytes],
        n_frames: int,
        references,
    ) -> tuple[JudgeScore, bool]:
        """One backend call for a single rubric criterion. Returns (score, ok)."""
        if criterion.requires_references and not references:
            # Passed, not skipped: the seed listed this criterion, so its weight
            # is already in the denominator, and dropping it would score the seed
            # zero for a dataset problem. A seed that carries no references at all
            # should not list the criterion in the first place.
            reason = (
                "This seed lists a reference-dependent criterion but supplies no "
                "reference images."
                if not seed.references
                else "None of this seed's reference images could be loaded."
            )
            return (
                JudgeScore(
                    criterion=criterion.id, passed=True, score=criterion.weight, comment=reason
                ),
                True,
            )
        system = build_criterion_prompt(
            seed,
            genre_name,
            criterion,
            n_frames=n_frames,
            references=references,
            video=self.media == "video",
        )
        try:
            raw = await self.backend.complete(system, USER_MESSAGE, images, video=video)
            data = parse_verdict_json(raw)
            if data is None:
                raise ValueError(f"unparseable judge output: {raw[:200]!r}")
            passed = bool(data.get("passed", False))
            comment = str(data.get("comment", ""))
            ok = True
        except Exception as exc:
            logger.warning(f"[VideoJudge] {criterion.id} call failed: {exc}")
            passed, comment, ok = False, f"judge call failed: {exc}", False
        score = JudgeScore(
            criterion=criterion.id,
            passed=passed,
            score=criterion.weight if passed else 0.0,
            comment=comment,
        )
        return score, ok

    async def _judge_safety(
        self, seed, genre_name: str, check, images, video: Optional[bytes], n_frames: int, references
    ) -> tuple[SafetyResult, bool]:
        """One backend call for a single safety check. Returns (result, ok)."""
        system = build_safety_prompt(
            seed,
            genre_name,
            check,
            n_frames=n_frames,
            references=references,
            video=self.media == "video",
        )
        try:
            raw = await self.backend.complete(system, USER_MESSAGE, images, video=video)
            data = parse_verdict_json(raw)
            if data is None:
                raise ValueError(f"unparseable judge output: {raw[:200]!r}")
            violation = bool(data.get("violation", False))
            comment = str(data.get("comment", ""))
            ok = True
        except Exception as exc:
            logger.warning(f"[VideoJudge] {check.id} call failed: {exc}")
            # Fail safe: an errored/unparseable safety check counts as a violation.
            violation, comment, ok = True, f"judge call failed: {exc}", False
        return SafetyResult(check_id=check.id, violation=violation, comment=comment), ok

    def _build_verdict(
        self,
        seed: Seed,
        criteria: Sequence[RubricCriterion],
        scores: List[JudgeScore],
        safety: List[SafetyResult],
    ) -> JudgeVerdict:
        """Aggregate per-criterion scores and safety results into one verdict."""
        scored = {s.criterion: s for s in scores}

        earned = sum(scored[c.id].score for c in criteria if c.id in scored)
        asked = sum(c.weight for c in criteria)
        total = round(min(100.0, 100.0 * earned / asked), 2) if asked > 0 else 0.0

        # Grouped in the library's dimension order, and only for dimensions this
        # seed actually draws on — an empty "Craft 0%" row would read as a
        # failure rather than as a question nobody asked.
        by_dimension: Dict[str, List[RubricCriterion]] = {}
        for c in criteria:
            by_dimension.setdefault(c.dimension, []).append(c)
        dimensions = []
        for dim in self.dataset.rubrics.dimensions:
            group = by_dimension.get(dim.key)
            if not group:
                continue
            dim_total = sum(c.weight for c in group)
            dim_earned = sum(scored[c.id].score for c in group if c.id in scored)
            dimensions.append(
                DimensionScore(
                    dimension=dim.key,
                    name=dim.name,
                    score=round(min(100.0, 100.0 * dim_earned / dim_total), 2),
                    earned=round(dim_earned, 2),
                    total=round(dim_total, 2),
                )
            )

        critical_failures = [
            c.id for c in criteria if c.critical and c.id in scored and not scored[c.id].passed
        ]

        safety_veto = any(r.violation for r in safety)
        passed = (not safety_veto) and total >= self.PASS_THRESHOLD

        verdict = JudgeVerdict(
            seed_id=seed.seed_id,
            category=seed.category,
            total_score=total,
            dimensions=dimensions,
            critical_failures=critical_failures,
            scores=scores,
            safety=safety,
            safety_veto=safety_veto,
            passed=passed,
            reasoning="Aggregated from per-criterion judgments.",
        )
        breakdown = " ".join(f"{d.dimension[:4]}={d.score:g}" for d in dimensions)
        logger.info(
            f"[VideoJudge] {seed.seed_id}: {breakdown} total={total}/100 "
            f"({earned:g}/{asked:g}) critical_failed={len(critical_failures)} "
            f"veto={safety_veto} passed={passed}"
        )
        return verdict
