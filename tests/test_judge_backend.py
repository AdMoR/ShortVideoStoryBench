"""
Judge LLM backends: PiBackend's retry behaviour, and the OpenAI-format backend
that can carry a whole clip.

A judge call that fails is not recorded as "not scored" — VideoJudge writes the
criterion down as a zero. So an infrastructure hiccup silently lowers the score
instead of showing up as a gap, which is exactly what happened in
runs/20260823-145259: U2 timed out at 300s and took section A to 0 with it.
These tests pin the retry that stops one slow call from rewriting the result.

The OpenAIBackend tests pin the two things that make the video path work at all:
the exact content-part shape a clip is sent as, and the preflight that refuses an
endpoint which cannot accept it.
"""

import shutil
from pathlib import Path

import pytest

from video_eval_bench.judge.pi_backend import PiBackend, PiConfig


@pytest.fixture
def backend(monkeypatch):
    """A PiBackend that never shells out, with the backoff removed."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/pi")

    def build(**kwargs):
        kwargs.setdefault("retry_backoff_seconds", 0.0)
        return PiBackend(PiConfig(**kwargs))

    return build


async def test_a_transient_failure_is_retried(backend, monkeypatch):
    """Two timeouts then a verdict must produce the verdict, not a zero."""
    calls = {"n": 0}

    async def flaky(self, system, prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("pi run timed out after 300.0s")
        return "PASS: the bottle is identical across shots"

    monkeypatch.setattr(PiBackend, "_run_pi", flaky)

    out = await backend(attempts=3).complete("system", "user", [])

    assert out.startswith("PASS")
    assert calls["n"] == 3


async def test_a_successful_call_is_not_retried(backend, monkeypatch):
    calls = {"n": 0}

    async def ok(self, system, prompt):
        calls["n"] += 1
        return "FAIL: the logo text is garbled"

    monkeypatch.setattr(PiBackend, "_run_pi", ok)

    out = await backend(attempts=3).complete("system", "user", [])

    assert out.startswith("FAIL")
    assert calls["n"] == 1, "a good answer must cost exactly one call"


async def test_exhausting_attempts_reports_the_last_error(backend, monkeypatch):
    """
    When every attempt fails the criterion still scores zero — but the message
    must say the judge failed, so a zero from a broken judge stays tellable from
    a zero the video earned.
    """
    calls = {"n": 0}

    async def always_fails(self, system, prompt):
        calls["n"] += 1
        raise RuntimeError("pi produced no assistant text")

    monkeypatch.setattr(PiBackend, "_run_pi", always_fails)

    with pytest.raises(RuntimeError, match="all 3 judge attempts failed"):
        await backend(attempts=3).complete("system", "user", [])

    assert calls["n"] == 3


async def test_every_image_reaches_the_prompt_in_order(backend, monkeypatch):
    """
    References are prepended to the frames as one flat list, and the prompt
    header numbers them. This backend attaches images as @-mentioned files, so a
    dropped or reordered one would silently mislabel every image after it — the
    judge would be told "Image 2 is the cafe" while looking at a frame.
    """
    seen = {}

    async def fake_run(self, system, prompt):
        seen["prompt"] = prompt
        return '{"passed": true, "comment": "ok"}'

    monkeypatch.setattr(PiBackend, "_run_pi", fake_run)

    images = [f"image-{i}".encode() for i in range(3)]
    await backend().complete("system", "user text", images)

    mentions = [w[1:] for w in seen["prompt"].split() if w.startswith("@")]
    assert len(mentions) == 3
    assert [Path(m).name for m in mentions] == ["img_01.jpg", "img_02.jpg", "img_03.jpg"]


async def test_a_call_with_no_images_attaches_nothing(backend, monkeypatch):
    seen = {}

    async def fake_run(self, system, prompt):
        seen["prompt"] = prompt
        return '{"passed": true, "comment": "ok"}'

    monkeypatch.setattr(PiBackend, "_run_pi", fake_run)
    await backend().complete("system", "user text", [])
    assert seen["prompt"] == "user text"


# ── OpenAIBackend: the path that can carry a whole clip ───────────────────────


def _openai_backend(**overrides):
    from video_eval_bench.judge.llm import LLMConfig, OpenAIBackend

    config = LLMConfig(model="a-model", api_base="http://endpoint:8080/v1", **overrides)
    return OpenAIBackend(config)


def test_openai_backend_needs_an_api_base():
    """Without it there is nothing to POST to, and the default would be a guess."""
    from video_eval_bench.judge.llm import LLMConfig, OpenAIBackend

    with pytest.raises(ValueError, match="api_base"):
        OpenAIBackend(LLMConfig(model="a-model"))


def test_openai_backend_sends_images_as_standard_content_parts():
    import base64

    messages = _openai_backend().build_messages("sys", "user", [b"jpeg-bytes"], None)
    parts = messages[1]["content"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert parts[0] == {"type": "text", "text": "user"}
    assert parts[1]["type"] == "image_url"
    encoded = base64.b64encode(b"jpeg-bytes").decode()
    assert parts[1]["image_url"]["url"] == f"data:image/jpeg;base64,{encoded}"
    # No video part when there is no video.
    assert not any(p["type"] == "input_video" for p in parts)


def test_openai_backend_sends_video_as_an_input_video_part():
    """
    `input_video` is an agreed extension to the OpenAI schema, base64 only. It is
    the reason this backend builds its own request body: litellm validates
    content parts and rejects this one.
    """
    import base64

    messages = _openai_backend().build_messages("sys", "user", [], b"\x00mp4-bytes")
    parts = messages[1]["content"]
    assert [p["type"] for p in parts] == ["text", "input_video"]
    assert parts[1]["input_video"] == {"data": base64.b64encode(b"\x00mp4-bytes").decode()}


async def test_openai_backend_surfaces_the_endpoints_rejection(monkeypatch):
    """
    A rejected content part is explained in the response body, and a criterion
    call that fails is scored ZERO rather than reported as a gap — so swallowing
    the body would turn a config error into a bad-looking video.
    """
    import httpx

    class Response:
        status_code = 400
        text = '{"error":{"message":"unsupported content[].type"}}'

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    with pytest.raises(RuntimeError, match="unsupported content"):
        await _openai_backend().complete("sys", "user", [], b"mp4")


def _props_client(monkeypatch, responses):
    """Stub httpx.AsyncClient; `responses` maps a URL to a status/body or an exception."""
    import httpx

    asked = []

    class Response:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            return self._body

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kw):
            asked.append(url)
            answer = responses.get(url, (404, None))
            if isinstance(answer, Exception):
                raise answer
            return Response(*answer)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    return asked


def test_preflight_looks_for_props_at_the_server_root_too():
    """
    `/props` is served at the root while api_base ends in `/v1`, so asking only
    `{api_base}/props` would 404 everywhere and the check would never run.
    """
    assert _openai_backend()._props_urls() == [
        "http://endpoint:8080/v1/props",
        "http://endpoint:8080/props",
    ]


async def test_openai_preflight_refuses_an_endpoint_without_video(monkeypatch):
    """
    Better one startup error than ~60 criterion calls each scoring zero and a
    report that reads as though every video failed.
    """
    asked = _props_client(monkeypatch, {
        # The /v1 candidate 404s, as it does on a real server; the root answers.
        "http://endpoint:8080/props": (
            200, {"modalities": {"vision": True, "audio": False, "video": False}}
        ),
    })
    with pytest.raises(RuntimeError, match="cannot accept video"):
        await _openai_backend().preflight()
    assert asked == ["http://endpoint:8080/v1/props", "http://endpoint:8080/props"]


async def test_openai_preflight_accepts_an_endpoint_with_video(monkeypatch):
    _props_client(monkeypatch, {
        "http://endpoint:8080/props": (200, {"modalities": {"vision": True, "video": True}}),
    })
    await _openai_backend().preflight()  # does not raise


async def test_openai_preflight_tolerates_an_endpoint_without_props(monkeypatch):
    """Plenty of OpenAI-compatible servers simply do not serve /props."""
    import httpx

    _props_client(monkeypatch, {
        "http://endpoint:8080/v1/props": httpx.ConnectError("nope"),
        "http://endpoint:8080/props": httpx.ConnectError("nope"),
    })
    await _openai_backend().preflight()  # does not raise


async def test_litellm_backend_refuses_video():
    """
    Falling back to frames would put a number in the report that the run's own
    config contradicts.
    """
    from video_eval_bench.judge.llm import LiteLlmBackend, LLMConfig

    with pytest.raises(ValueError, match="cannot send video"):
        await LiteLlmBackend(LLMConfig()).complete("sys", "user", [], b"mp4")


async def test_pi_backend_refuses_video(backend):
    with pytest.raises(ValueError, match="cannot send video"):
        await backend().complete("sys", "user", [], b"mp4")
