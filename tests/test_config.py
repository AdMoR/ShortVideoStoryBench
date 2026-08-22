"""
The config system: Hydra composes the axes, pydantic refuses the impossible ones.

The validation tests are the point. An ablation arm that is silently inert — a
`skills=` variant whose skills the agent is never told about, or an override with
a typo in it — produces a plausible number attributed to the wrong configuration,
which is worse than a crash.
"""

import pytest
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from pydantic import ValidationError

import video_eval_bench.run as run  # registers the veb_prompt resolver
from video_eval_bench.config import (
    BenchConfig,
    ModelConfig,
    PiGeneratorConfig,
    PromptConfig,
    SkillsConfig,
    ToolsConfig,
    redact,
)


def build(*overrides):
    """Compose the real config tree and validate it, as `veb` does."""
    with initialize_config_module("video_eval_bench.conf", version_base="1.3"):
        cfg = compose("config", overrides=list(overrides), return_hydra_config=True)
        choices = dict(cfg.hydra.runtime.choices)
        variant = run.variant_of(cfg.hydra.runtime.choices)
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            cfg.pop("hydra")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        resolved.pop("paths", None)
        return BenchConfig(**resolved), choices, variant


# ── composition ───────────────────────────────────────────────────────────────


def test_defaults_compose_to_the_pi_agent_on_gx10():
    bench, _, _ = build()
    assert bench.generator.kind == "pi"
    assert bench.generator.model.provider == "gx10"
    assert bench.generator.model.name == "unsloth/Qwen3.8-27B-GGUF:Q8_0"
    assert bench.judge.backend == "pi"


def test_each_axis_swaps_independently():
    bench, _, _ = build("tools=no_bash", "system_prompt=minimal", "model=amor_ms_qwen27b_q3")
    assert "bash" not in bench.generator.tools.builtin
    assert bench.generator.model.provider == "amor-ms"
    # The minimal prompt is the short one; the director prompt is not.
    assert len(bench.generator.system_prompt.system()) < 500


def test_packaged_prompts_resolve_to_real_files():
    """`${veb_prompt:...}` finds the prompts shipped with the package."""
    bench, _, _ = build()
    system = bench.generator.system_prompt.system()
    assert "submit_video" in system
    # The director prompt carries the waiting rule — the expensive thing to lose.
    assert "poll" in system.lower()
    assert "{prompt}" in bench.generator.system_prompt.task()  # still a template


def test_experiment_preset_drops_inapplicable_axes():
    bench, choices, _ = build("experiment=mock")
    assert bench.generator.kind == "mock"
    assert bench.judge.backend == "mock"
    assert choices.get("model") in (None, "null")


def test_variant_is_derived_from_the_chosen_options():
    _, _, variant = build("skills=none", "tools=no_bash")
    assert "tools=no_bash" in variant
    assert "model=gx10_qwen27b_q8" in variant


def test_run_selection_overrides():
    bench, _, _ = build("run.category=marketing", "run.max_seeds=1")
    assert bench.run.category == "marketing"
    assert bench.run.max_seeds == 1


# ── validation ────────────────────────────────────────────────────────────────


def test_typo_in_an_override_is_refused():
    """A silently-ignored override would misattribute a whole run."""
    with pytest.raises(Exception):
        build("generator.timout_seconds=10")


def test_unknown_tool_name_is_refused():
    with pytest.raises(ValidationError, match="unknown pi built-in tool"):
        ToolsConfig(builtin=["read", "bassh"])


def test_skills_without_read_are_refused(tmp_path):
    """
    pi only injects the skills block when the `read` tool is enabled, so this
    config would run happily and measure nothing.
    """
    skill = tmp_path / "skill"
    skill.mkdir()
    with pytest.raises(ValidationError, match="silently ignored"):
        PiGeneratorConfig(
            model=ModelConfig(name="m"),
            system_prompt=PromptConfig(system_text="s", task_text="t"),
            tools=ToolsConfig(builtin=["write", "bash"]),
            skills=SkillsConfig(paths=[skill]),
        )


def test_skills_with_read_are_accepted(tmp_path):
    skill = tmp_path / "skill"
    skill.mkdir()
    cfg = PiGeneratorConfig(
        model=ModelConfig(name="m"),
        system_prompt=PromptConfig(system_text="s", task_text="t"),
        tools=ToolsConfig(builtin=["read", "bash"]),
        skills=SkillsConfig(paths=[skill]),
    )
    assert cfg.skills.paths == [skill]


def test_missing_referenced_file_is_refused(tmp_path):
    with pytest.raises(ValidationError, match="do not exist"):
        PiGeneratorConfig(
            model=ModelConfig(name="m"),
            system_prompt=PromptConfig(system_text="s", task_text="t"),
            extensions=[tmp_path / "nope.ts"],
        )


def test_prompt_given_twice_is_refused():
    with pytest.raises(ValidationError, match="not both"):
        PromptConfig(system_text="a", system_file="b.md", task_text="t")


def test_prompt_must_be_given():
    with pytest.raises(ValidationError, match="is required"):
        PromptConfig(system_text="a")


# ── redaction ─────────────────────────────────────────────────────────────────


def test_secrets_are_redacted_before_reaching_a_report():
    out = redact(
        {
            "judge": {"api_key": "sk-live-123", "backend": "litellm"},
            "generator": {"env": {"SERVICE_URL": "http://x", "SERVICE_API_KEY": "hunter2"}},
        }
    )
    assert out["judge"]["api_key"] == "***"
    assert out["generator"]["env"]["SERVICE_API_KEY"] == "***"
    assert out["generator"]["env"]["SERVICE_URL"] == "http://x"
