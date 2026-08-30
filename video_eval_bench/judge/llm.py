"""
LLM backends for the judge.

The judge needs a vision-capable LLM. Three backends:

    LiteLlmBackend   — hosted providers via litellm (OpenAI, Anthropic, Google, ...)
    OpenAIBackend    — a raw OpenAI chat-completions endpoint, spoken directly
    MockBackend      — deterministic fake for offline testing / skeleton validation

All implement `VisionLLM`:
    async complete(system, user_text, images, video=None) -> str

`video` is what separates the two real backends. Frames are a workaround for
models that only take stills: motion, timing and cut rhythm have to be inferred
from the gaps between them, which is exactly what criteria like CUT1, MOTION1
and TEMP1 are asking about. An endpoint that accepts a whole clip judges those
from the clip. litellm validates message content against the OpenAI schema and
rejects the video part outright, so the video path needs an endpoint spoken
directly — hence OpenAIBackend, which builds the request body itself.

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
    # Generous by default: a call carrying a whole clip is slow to upload and
    # slow to prefill compared with a handful of stills.
    timeout_seconds: float = 300.0
    extra: dict = field(default_factory=dict)


class VisionLLM(abc.ABC):
    """Minimal async interface: one completion with text + JPEG images (+ a clip)."""

    #: True for a backend that can be handed the clip itself rather than frames.
    #: `build_judge` checks this before configuring `media="video"`, so an
    #: unsupported combination fails at startup instead of once per criterion.
    supports_video: bool = False

    @abc.abstractmethod
    async def complete(
        self,
        system: str,
        user_text: str,
        images: List[bytes],
        video: Optional[bytes] = None,
    ) -> str:
        """Return the raw text response of the model."""


class LiteLlmBackend(VisionLLM):
    """Real vision LLM calls via litellm (OpenAI chat-completions message format)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    async def complete(
        self,
        system: str,
        user_text: str,
        images: List[bytes],
        video: Optional[bytes] = None,
    ) -> str:
        import base64

        import litellm

        if video is not None:
            # Not a soft fallback to frames: silently grading a "video" run on
            # stills would put a number in the report that its own config
            # contradicts.
            raise ValueError(
                "LiteLlmBackend cannot send video — litellm validates content parts "
                "against the OpenAI schema and rejects the video part. Use "
                "judge.backend=openai against an endpoint that accepts video."
            )

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


class OpenAIBackend(VisionLLM):
    """
    An OpenAI chat-completions endpoint, spoken directly over HTTP.

    Exists for one reason: it builds the request body itself, so it can send the
    clip. `POST {api_base}/chat/completions` with the standard message shape,
    plus one non-standard content part for video:

        {"type": "input_video", "input_video": {"data": "<base64 of the file>"}}

    That part is an agreed extension to the OpenAI schema rather than part of
    it. An endpoint that does not implement it answers 400 — which is why
    `preflight()` exists and why `veb` calls it before a run rather than
    discovering it on the first of ~60 criterion calls. Images use the standard
    `image_url` part with a data: URI and work against any vision endpoint.

    The whole file is sent per call, base64-encoded (a third larger than the
    file). The judge makes one call per criterion, so a seed costs
    `size × n_criteria` in upload — keep clips short, which the benchmark's are.
    """

    supports_video = True

    def __init__(self, config: LLMConfig):
        if not config.api_base:
            raise ValueError("OpenAIBackend needs judge.api_base (the endpoint's base URL)")
        self.config = config

    @property
    def _url(self) -> str:
        return self.config.api_base.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def build_messages(
        self, system: str, user_text: str, images: List[bytes], video: Optional[bytes]
    ) -> list:
        """The message list for one call. Split out so tests can read it."""
        import base64

        parts: list = [{"type": "text", "text": user_text}]
        for img in images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                    },
                }
            )
        if video is not None:
            parts.append(
                {
                    "type": "input_video",
                    "input_video": {"data": base64.b64encode(video).decode()},
                }
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": parts},
        ]

    def _props_urls(self) -> List[str]:
        """
        Where the endpoint might advertise its modalities.

        Two candidates because `/props` is served at the server ROOT, while
        `api_base` normally ends in `/v1` (that is where chat/completions lives).
        Asking only `{api_base}/props` would 404 on every standard deployment and
        the check would silently never run — worse than not having it.
        """
        from urllib.parse import urlsplit, urlunsplit

        base = self.config.api_base.rstrip("/")
        parts = urlsplit(base)
        root = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        candidates = [f"{base}/props", f"{root}/props"]
        return list(dict.fromkeys(candidates))

    async def preflight(self) -> None:
        """
        Fail before the run if the endpoint cannot do what the config asks.

        Cheap, and it turns "every criterion errored" — which the judge records
        as a score of zero, not as a gap — into one startup error naming the
        cause. An endpoint that serves no `/props` at all is not treated as a
        failure (plenty of OpenAI-compatible servers do not have it), but it is
        logged, because it means this check did not actually run.
        """
        import httpx

        modalities = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in self._props_urls():
                try:
                    response = await client.get(url, headers=self._headers())
                except Exception as exc:
                    logger.debug(f"[OpenAIBackend] {url}: {exc}")
                    continue
                if response.status_code != 200:
                    continue
                try:
                    modalities = (response.json() or {}).get("modalities")
                except ValueError:
                    continue
                if modalities is not None:
                    break

        if not modalities:
            logger.warning(
                "[OpenAIBackend] no /props modalities from "
                f"{self.config.api_base} — cannot verify it accepts video before "
                "the run; a rejection will surface on the first criterion call"
            )
            return
        logger.info(f"[OpenAIBackend] endpoint modalities: {modalities}")
        if not modalities.get("video"):
            raise RuntimeError(
                f"{self.config.api_base} reports modalities={modalities} — it "
                "cannot accept video. The server needs a multimodal projector "
                "loaded (--mmproj), a build with video support enabled, and "
                "ffmpeg/ffprobe on its PATH. Run with judge.media=frames until then."
            )

    async def complete(
        self,
        system: str,
        user_text: str,
        images: List[bytes],
        video: Optional[bytes] = None,
    ) -> str:
        import httpx

        payload = {
            "model": self.config.model,
            "messages": self.build_messages(system, user_text, images, video),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            **self.config.extra,
        }
        logger.info(
            f"[OpenAIBackend] calling {self.config.model} at {self._url} with "
            f"{len(images)} image(s)" + (f" + {len(video) / 1e6:.1f} MB video" if video else "")
        )
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(self._url, headers=self._headers(), json=payload)
        if response.status_code != 200:
            # The body is where a rejected content part is explained, and it is
            # the whole reason this backend exists — never swallow it.
            raise RuntimeError(
                f"{self._url} returned {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        return (data["choices"][0]["message"].get("content") or "").strip()


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

    supports_video = True

    async def complete(
        self,
        system: str,
        user_text: str,
        images: List[bytes],
        video: Optional[bytes] = None,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "user_text": user_text,
                "n_images": len(images),
                "n_video_bytes": len(video) if video else 0,
            }
        )
        if self.fail:
            return "I cannot evaluate this video, sorry."
        return self._mock_response(system)
