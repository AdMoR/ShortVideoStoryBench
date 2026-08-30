"""
PiBackend — judge LLM backend that shells out to the `pi` CLI.

Design (validated against pi 0.x + amor-ms/unsloth/Qwen3.8-27B-UD-Q3_K_XL.gguf,
a local llama.cpp server registered under the "amor-ms" pi provider):

  * VideoJudge decomposes the rubric into one small call per criterion / safety
    check, so each pi run only ever needs to answer one focused binary
    question. Frames are pre-extracted once by VideoJudge and attached via
    @-mentions in the prompt — the model never needs to go looking for them
    itself, so pi is run with no tools at all (`--no-tools`). This keeps each
    run to a single turn: no risk of the model burning its budget re-reading
    already-attached images or exploring the filesystem.
  * pi CANNOT watch video directly (`read` on an .mp4 returns raw container
    bytes), which is moot now since VideoJudge never hands PiBackend a video
    path — only pre-extracted JPEG frames.

Payload discipline (videos can be large):
  * The raw video is NEVER sent to the model. Only small JPEG frames
    (default: 6 frames, 512px wide, q75 ~= 30-60 KB each).
  * Frames are written to a per-call temp dir; the prompt references them by
    absolute path.
  * `--mode json` NDJSON output is parsed for the final assistant text;
    `--no-session` keeps runs ephemeral.
"""

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from video_eval_bench.judge.llm import VisionLLM
from video_eval_bench.pi_ndjson import final_assistant_text, iter_events

logger = logging.getLogger(__name__)

DEFAULT_PI_MODEL = "amor-ms/unsloth/Qwen3.8-27B-UD-Q3_K_XL.gguf"


@dataclass
class PiConfig:
    """Configuration for the PiBackend."""

    pi_bin: str = "pi"
    model: str = DEFAULT_PI_MODEL
    provider: Optional[str] = None
    timeout_seconds: float = 120.0
    # A criterion that fails here is recorded as a score of ZERO, not as "not
    # scored" — so a slow model or a hiccup silently depresses the result rather
    # than showing up as a gap. Observed in runs/20260823-145259, where U2 timed
    # out and dragged section A to 0. Retrying is much cheaper than a wrong number.
    attempts: int = 3
    retry_backoff_seconds: float = 5.0
    extra_args: List[str] = field(default_factory=list)


class PiBackend(VisionLLM):
    """VisionLLM backend driven by the pi CLI: one focused, tool-free run per call."""

    def __init__(self, config: Optional[PiConfig] = None):
        self.config = config or PiConfig()
        if shutil.which(self.config.pi_bin) is None:
            raise RuntimeError(f"pi binary not found: {self.config.pi_bin!r}")

    async def complete(
        self,
        system: str,
        user_text: str,
        images: List[bytes],
        video: Optional[bytes] = None,
    ) -> str:
        """One pi run: system prompt + user text + attached JPEG images, no tools."""
        if video is not None:
            # `read` on an .mp4 gives pi raw container bytes, not a watchable
            # clip. Refusing beats attaching a file the model cannot decode and
            # letting it answer anyway.
            raise ValueError(
                "PiBackend cannot send video — pi attaches files for the model to "
                "read, and it cannot decode a video container. Use "
                "judge.backend=openai for judge.media=video."
            )
        with tempfile.TemporaryDirectory(prefix="pi_judge_") as tmp:
            tmp_path = Path(tmp)
            mentions = []
            for i, img in enumerate(images, 1):
                p = tmp_path / f"img_{i:02d}.jpg"
                p.write_bytes(img)
                mentions.append(str(p))
            prompt = user_text + ("\n\n" + " ".join(f"@{m}" for m in mentions) if mentions else "")
            return await self._complete_with_retries(system, prompt)

    async def _complete_with_retries(self, system: str, prompt: str) -> str:
        """
        Run one criterion, retrying transient failures.

        Every failure mode here — a timeout, a non-zero exit, an empty reply — is
        the infrastructure failing rather than the video being bad, and the caller
        cannot tell the difference: it scores the criterion zero either way. So
        the cost of one more attempt is far lower than the cost of a wrong score.
        """
        cfg = self.config
        attempts = max(1, cfg.attempts)
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                return await self._run_pi(system, prompt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # timeout, bad exit, empty text
                last = exc
                if attempt == attempts:
                    break
                delay = cfg.retry_backoff_seconds * attempt
                logger.warning(
                    f"[PiBackend] attempt {attempt}/{attempts} failed ({exc}); "
                    f"retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"all {attempts} judge attempts failed; last error: {last}")

    async def _run_pi(self, system: str, prompt: str) -> str:
        """Run pi in non-interactive, tool-free JSON mode and return the final assistant text."""
        cfg = self.config
        cmd = [
            cfg.pi_bin,
            *(["--provider", cfg.provider] if cfg.provider else []),
            "--model", cfg.model,
            "--print",
            "--mode", "json",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--offline",
            "--no-tools",
            "--system-prompt", system,
            *cfg.extra_args,
            prompt,
        ]
        logger.info(f"[PiBackend] running pi ({cfg.model}), prompt={len(prompt)} chars")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=cfg.timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"pi run timed out after {cfg.timeout_seconds}s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"pi exited with {proc.returncode}: {stderr.decode(errors='replace')[-500:]}"
            )

        text = self._parse_final_text(stdout.decode(errors="replace"))
        if not text:
            raise RuntimeError("pi produced no assistant text")
        return text

    @staticmethod
    def _parse_final_text(ndjson: str) -> str:
        """Extract the last assistant message text from pi's NDJSON stream."""
        return final_assistant_text(iter_events(ndjson))
