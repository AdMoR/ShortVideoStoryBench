"""
Fixtures for the end-to-end tier.

These tests drive the real `pi` binary against a real model. They are opt-in
(`-m e2e`) and skip rather than fail when either is unavailable, so a laptop
without the model still gets a green offline suite.
"""

import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_video_service import MockVideoService  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = Path(__file__).resolve().parent / "skills" / "mock-video"

# The provider the e2e configs target, as registered in ~/.pi/agent/models.json.
MODEL_ENDPOINT = "http://gx10-cbc5:8081/v1/models"


def _model_available() -> bool:
    try:
        with urllib.request.urlopen(MODEL_ENDPOINT, timeout=3) as response:
            json.loads(response.read() or b"{}")
        return True
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


@pytest.fixture(scope="session")
def live_model():
    """Skip the tier unless pi is installed and the model endpoint answers."""
    if shutil.which("pi") is None:
        pytest.skip("pi is not on PATH")
    if not _model_available():
        pytest.skip(f"no model at {MODEL_ENDPOINT}")
    return MODEL_ENDPOINT


@pytest.fixture
def video_service():
    """A stand-in backend; `delay` and `fail_first` shape the scenario."""

    started = []

    def start(delay: float = 0.0, fail_first: bool = False) -> MockVideoService:
        service = MockVideoService(delay=delay, fail_first=fail_first)
        service.url = service.start()
        started.append(service)
        return service

    yield start
    for service in started:
        service.stop()


@pytest.fixture
def e2e_config(live_model, video_service):
    """
    A validated BenchConfig for the e2e tier, composed from the real config tree
    so the shipped `experiment=e2e` preset is what is under test.
    """
    from hydra import compose, initialize_config_module
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf, open_dict

    import video_eval_bench.run as run  # noqa: F401 - registers the veb_prompt resolver
    from video_eval_bench.config import BenchConfig

    def build(service, *overrides) -> BenchConfig:
        with initialize_config_module("video_eval_bench.conf", version_base="1.3"):
            cfg = compose(
                "config",
                overrides=[
                    "experiment=e2e",
                    f"paths.root={REPO_ROOT}",
                    *overrides,
                ],
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)
            with open_dict(cfg):
                cfg.pop("hydra")
            resolved = OmegaConf.to_container(cfg, resolve=True)
            resolved.pop("paths", None)
            resolved["generator"].setdefault("env", {})
            resolved["generator"]["env"]["MOCK_VIDEO_URL"] = service.url
            return BenchConfig(**resolved)

    return build
