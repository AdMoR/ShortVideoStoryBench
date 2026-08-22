"""
LLM backends for the judge.

The judge needs a vision-capable LLM. Two backends:

    LiteLlmBackend — real calls via litellm (OpenAI-compatible, Anthropic, Google, ...)
    MockBackend    — deterministic fake for offline testing / skeleton validation

Both implement `VisionLLM`:
    async complete(system: str, user_text: str, images: list[bytes]) -> str

VideoJudge decomposes the rubric into one small `complete()` call per
criterion / safety check, so a backend only ever needs to answer one focused
question at a time — there is no video-wide entry point to implement.
"""

import abc
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LiteLlmBackend."""

    model: str = "openai/gpt-4o"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 2048
    extra: dict = field(default_factory=dict)


class VisionLLM(abc.ABC):
    """Minimal async interface: one completion with text + JPEG images."""

    @abc.abstractmethod
    async def complete(self, system: str, user_text: str, images: List[bytes]) -> str:
        """Return the raw text response of the model."""


class LiteLlmBackend(VisionLLM):
    """Real vision LLM calls via litellm (OpenAI chat-completions message format)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    async def complete(self, system: str, user_text: str, images: List[bytes]) -> str:
        import base64

        import litellm

        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                },
            }
            for img in images
        ]
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}, *image_parts],
            },
        ]

        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        kwargs.update(self.config.extra)

        logger.info(f"[LiteLlmBackend] calling {self.config.model} with {len(images)} image(s)")
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""


class MockBackend(VisionLLM):
    """
    Deterministic fake backend for offline testing.

    Each complete() call covers exactly one rubric criterion or one safety
    check (matching VideoJudge's per-criterion decomposition). Produces a
    plausible single-item verdict: pass (or fail, via fail_rate) for a
    criterion, no violation (or one, via veto) for a safety check — so the
    full pipeline (prompt building, frame extraction, JSON parsing, section
    aggregation, veto logic) can be exercised without any network or model.
    """

    def __init__(self, fail: bool = False, fail_rate: float = 0.0, veto: bool = False):
        self.fail = fail  # when True, simulate a broken judge (no JSON)
        self.fail_rate = fail_rate  # fraction of criteria that fail (deterministic: every Nth)
        self.veto = veto  # when True, flag D1 as a safety violation
        self.calls: List[dict] = []  # recorded for test assertions
        self._criterion_count = 0  # counts criterion calls, for fail_rate

    def _mock_response(self, system: str) -> str:
        # A criterion prompt names exactly one "## Criterion to check: <id>";
        # a safety prompt names exactly one "## Safety check: <id>".
        crit_match = re.search(r"^## Criterion to check: (\S+)", system, flags=re.MULTILINE)
        safety_match = re.search(r"^## Safety check: (\S+)", system, flags=re.MULTILINE)

        if safety_match:
            check_id = safety_match.group(1)
            return json.dumps(
                {"violation": self.veto and check_id == "D1", "comment": "mock"}
            )

        if crit_match:
            i = self._criterion_count
            self._criterion_count += 1
            passed = self.fail_rate <= 0 or (i % max(1, round(1 / self.fail_rate)) != 0)
            return json.dumps({"passed": passed, "comment": "mock"})

        # Unrecognized prompt shape — behave permissively rather than error.
        return json.dumps({"passed": True, "violation": False, "comment": "mock"})

    async def complete(self, system: str, user_text: str, images: List[bytes]) -> str:
        self.calls.append({"system": system, "user_text": user_text, "n_images": len(images)})
        if self.fail:
            return "I cannot evaluate this video, sorry."
        return self._mock_response(system)
