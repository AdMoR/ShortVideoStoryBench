"""
PiBackend retry behaviour.

A judge call that fails is not recorded as "not scored" — VideoJudge writes the
criterion down as a zero. So an infrastructure hiccup silently lowers the score
instead of showing up as a gap, which is exactly what happened in
runs/20260823-145259: U2 timed out at 300s and took section A to 0 with it.
These tests pin the retry that stops one slow call from rewriting the result.
"""

import shutil

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
