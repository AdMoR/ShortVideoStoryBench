"""Shared fixtures: a fake `pi` binary and a ready-made PiGeneratorConfig."""

import os
import stat
import sys
from pathlib import Path

import pytest

from video_eval_bench.config import (
    DEFAULT_PROMPT_DIR,
    ModelConfig,
    PiGeneratorConfig,
    PromptConfig,
    ToolsConfig,
)

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def fake_pi(tmp_path_factory) -> Path:
    """
    An executable that stands in for `pi`.

    A launcher with `sys.executable` in its shebang, so the stub runs under the
    same interpreter as the tests and can import cv2 and the package.
    """
    path = tmp_path_factory.mktemp("bin") / "pi"
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path.insert(0, {str(TESTS_DIR)!r})\n"
        "import fake_pi\n"
        "sys.exit(fake_pi.main())\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def pi_config(fake_pi):
    """A PiGeneratorConfig pointed at the fake binary, with test-scale timings."""

    def build(**overrides) -> PiGeneratorConfig:
        defaults = dict(
            pi_bin=str(fake_pi),
            model=ModelConfig(provider="gx10", name="test-model"),
            system_prompt=PromptConfig(
                system_file=DEFAULT_PROMPT_DIR / "director_system.md",
                task_file=DEFAULT_PROMPT_DIR / "task.md",
            ),
            tools=ToolsConfig(builtin=["read", "write", "edit", "bash"]),
            timeout_seconds=30.0,
            exit_grace_seconds=1.0,
            heartbeat_seconds=0.5,
        )
        defaults.update(overrides)
        return PiGeneratorConfig(**defaults)

    return build


@pytest.fixture
def mode(monkeypatch):
    """Select the fake pi's behaviour for one test."""

    def set_mode(name: str, **env):
        monkeypatch.setenv("FAKE_PI_MODE", name)
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

    return set_mode


@pytest.fixture(autouse=True)
def _clean_fake_pi_env(monkeypatch):
    """Never let one test's mode leak into the next."""
    for key in ("FAKE_PI_MODE", "FAKE_PI_LINGER", "FAKE_PI_SILENCE"):
        monkeypatch.delenv(key, raising=False)
    yield
