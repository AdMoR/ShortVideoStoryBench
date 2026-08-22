"""
Typed run configuration.

Hydra composes the YAML (see `video_eval_bench/conf/`); this module validates the
result and builds the live objects. The split matters: Hydra is good at assembling
a config out of interchangeable pieces, and bad at telling you that the pieces you
assembled cannot work. Pydantic is the second half.

Every model sets `extra="forbid"`. In a sweep, an override with a typo in it that
silently does nothing is worse than a crash — it produces a plausible-looking
result attributed to the wrong configuration.

    cfg = OmegaConf.to_container(hydra_cfg, resolve=True)
    bench = BenchConfig(**cfg)
    generate = build_generator(bench.generator)
    judge = build_judge(bench.judge, dataset)
"""

import logging
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# pi's built-in tools (`pi --help`, "Built-in Tool Names"). Used to reject typos in
# a `tools=` ablation arm before a run burns an hour proving the point.
PI_BUILTIN_TOOLS = frozenset({"read", "bash", "edit", "write", "grep", "find", "ls"})

# The tool the agent must call to hand back its video. Registered by the packaged
# extension, so it is always present regardless of the configured tool allowlist.
SUBMIT_TOOL = "submit_video"

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = PACKAGE_DIR / "generator" / "prompts"
BENCH_TOOLS_EXTENSION = PACKAGE_DIR / "generator" / "pi_ext" / "bench_tools.ts"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    """Which model, on which pi provider."""

    provider: Optional[str] = Field(
        default=None,
        description="pi provider name, e.g. 'gx10'. None uses pi's default provider.",
    )
    name: str = Field(description="Model id as pi knows it, e.g. 'unsloth/Qwen3.8-27B-GGUF:Q8_0'")
    thinking: Optional[str] = Field(
        default=None,
        description="pi thinking level: off, minimal, low, medium, high, xhigh, max",
    )


class ToolsConfig(_Base):
    """Which pi tools the agent may use — one of the ablation axes."""

    builtin: List[str] = Field(default_factory=lambda: ["read", "write", "edit", "bash"])
    exclude: List[str] = Field(default_factory=list)
    no_builtin: bool = Field(
        default=False,
        description="Drop every built-in tool; only extension tools remain.",
    )

    @model_validator(mode="after")
    def _known_tool_names(self) -> "ToolsConfig":
        unknown = sorted((set(self.builtin) | set(self.exclude)) - PI_BUILTIN_TOOLS)
        if unknown:
            raise ValueError(
                f"unknown pi built-in tool(s): {unknown}. "
                f"Valid names: {sorted(PI_BUILTIN_TOOLS)}"
            )
        return self

    def effective(self) -> set:
        """The tools actually available to the agent."""
        if self.no_builtin:
            return set()
        return set(self.builtin) - set(self.exclude)


class SkillsConfig(_Base):
    """Which skills the agent is given — the highest-value ablation axis."""

    paths: List[Path] = Field(default_factory=list)
    discover: bool = Field(
        default=False,
        description="Let pi discover skills from its usual roots as well.",
    )


class PromptConfig(_Base):
    """
    The system prompt and the per-seed task template.

    Each may be given inline or as a file, never both — a config that specifies a
    prompt twice is a config whose author is unsure which one is live.
    """

    system_text: Optional[str] = None
    system_file: Optional[Path] = None
    append_files: List[Path] = Field(default_factory=list)
    task_text: Optional[str] = None
    task_file: Optional[Path] = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "PromptConfig":
        for inline, file, label in (
            (self.system_text, self.system_file, "system"),
            (self.task_text, self.task_file, "task"),
        ):
            if inline is not None and file is not None:
                raise ValueError(
                    f"{label} prompt: give either {label}_text or {label}_file, not both"
                )
            if inline is None and file is None:
                raise ValueError(f"{label} prompt: one of {label}_text / {label}_file is required")
        return self

    def system(self) -> str:
        return self.system_text if self.system_text is not None else _read(self.system_file)

    def task(self) -> str:
        return self.task_text if self.task_text is not None else _read(self.task_file)

    def appended(self) -> List[str]:
        return [_read(p) for p in self.append_files]


class PiGeneratorConfig(_Base):
    """An agentic generator: one `pi` run per seed, in an isolated workspace."""

    kind: Literal["pi"] = "pi"
    pi_bin: str = "pi"
    model: ModelConfig
    system_prompt: PromptConfig
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    extensions: List[Path] = Field(
        default_factory=list,
        description="Extra pi extension files (.ts) providing custom tools.",
    )
    discover_extensions: bool = False
    context_files: bool = Field(default=False, description="Load AGENTS.md / CLAUDE.md.")
    prompt_templates: bool = False
    approve_project: bool = Field(
        default=False,
        description="Trust project-local .pi files in the workspace (only needed if copy_files stages one).",
    )
    offline: bool = Field(default=True, description="Skip pi's startup network calls.")

    # Timing. See §4a/§4b of the design: the only timer that may exist is the
    # overall budget — an inactivity timeout would kill a legitimate long
    # blocking generation command.
    timeout_seconds: float = Field(
        default=3600.0,
        gt=0,
        description="Hard per-seed budget. Must exceed the longest expected generation.",
    )
    exit_grace_seconds: float = Field(
        default=15.0,
        gt=0,
        description="How long to let pi exit on its own after submit_video before killing it.",
    )
    heartbeat_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Progress logging interval while the agent is working.",
    )

    env: dict = Field(default_factory=dict, description="Extra environment for the pi process.")
    copy_files: List[Path] = Field(
        default_factory=list, description="Files staged into each seed's workspace."
    )
    extra_args: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _skills_need_read(self) -> "PiGeneratorConfig":
        """
        pi appends its skills block to the system prompt only when the `read`
        built-in is enabled, and that block is a list of *paths* the model is told
        to open with `read`. With `read` disabled, `--skill` is silently inert:
        the agent is never told the skills exist and could not open them anyway.

        Caught here because the failure is invisible at runtime — a `skills=` arm
        of an ablation would appear to run fine and measure nothing.
        """
        if self.skills.paths and "read" not in self.tools.effective():
            raise ValueError(
                f"skills are configured ({len(self.skills.paths)} path(s)) but the 'read' tool "
                "is not enabled. pi only injects the skills block when `read` is available, so "
                "the skills would be silently ignored. Add 'read' to tools.builtin, or drop the "
                "skills."
            )
        return self

    @model_validator(mode="after")
    def _files_exist(self) -> "PiGeneratorConfig":
        missing = [
            str(p)
            for p in [
                self.system_prompt.system_file,
                self.system_prompt.task_file,
                *self.system_prompt.append_files,
                *self.skills.paths,
                *self.extensions,
                *self.copy_files,
            ]
            if p is not None and not Path(p).exists()
        ]
        if missing:
            raise ValueError(f"configured path(s) do not exist: {missing}")
        return self


class MockGeneratorConfig(_Base):
    """Synthetic videos — offline pipeline checks, no model involved."""

    kind: Literal["mock"] = "mock"
    n_frames: int = 16
    fps: int = 8
    width: int = 320
    height: int = 180


GeneratorConfig = Annotated[
    Union[PiGeneratorConfig, MockGeneratorConfig], Field(discriminator="kind")
]


class JudgeConfig(_Base):
    """Which vision LLM grades the videos."""

    backend: Literal["mock", "pi", "litellm"] = "mock"
    model: Optional[str] = None
    provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    pi_bin: str = "pi"
    timeout_seconds: float = 120.0
    n_frames: int = Field(default=8, gt=0)


class RunConfig(_Base):
    """Which seeds to run, and where the dataset comes from."""

    dataset_dir: Optional[Path] = None
    category: Optional[str] = None
    seed_ids: Optional[List[str]] = None
    max_seeds: Optional[int] = Field(
        default=None,
        gt=0,
        description="Cap the seed count. An agentic run on a local model is slow; "
        "ablations should run on one or two seeds.",
    )
    note: str = ""


class BenchConfig(_Base):
    """One benchmark run: one generator variant, one judge, one seed selection."""

    generator: GeneratorConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def _read(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return Path(path).read_text(encoding="utf-8")


# ── builders ──────────────────────────────────────────────────────────────────


def build_generator(cfg):
    """Build the generator callable for a validated generator config."""
    if isinstance(cfg, MockGeneratorConfig):
        from video_eval_bench.generator.mock_generator import MockGenerator

        return MockGenerator(n_frames=cfg.n_frames, fps=cfg.fps, size=(cfg.width, cfg.height))

    from video_eval_bench.generator.pi_generator import PiGenerator

    return PiGenerator(cfg)


def build_judge(cfg: JudgeConfig, dataset):
    """Build the VideoJudge for a validated judge config."""
    from video_eval_bench.judge.agent import VideoJudge
    from video_eval_bench.judge.llm import LLMConfig, LiteLlmBackend, MockBackend

    if cfg.backend == "mock":
        backend = MockBackend()
    elif cfg.backend == "pi":
        from video_eval_bench.judge.pi_backend import PiBackend, PiConfig

        backend = PiBackend(
            PiConfig(
                pi_bin=cfg.pi_bin,
                provider=cfg.provider,
                model=cfg.model or PiConfig.model,
                timeout_seconds=cfg.timeout_seconds,
            )
        )
    else:
        backend = LiteLlmBackend(
            LLMConfig(
                model=cfg.model or "openai/gpt-4o",
                api_base=cfg.api_base,
                api_key=cfg.api_key,
            )
        )
    return VideoJudge(backend=backend, dataset=dataset, n_frames=cfg.n_frames)


def redact(config: dict) -> dict:
    """Copy of a resolved config with secrets removed, safe to write into a report."""
    import copy

    out = copy.deepcopy(config)
    judge = out.get("judge")
    if isinstance(judge, dict) and judge.get("api_key"):
        judge["api_key"] = "***"
    generator = out.get("generator")
    if isinstance(generator, dict) and isinstance(generator.get("env"), dict):
        generator["env"] = {
            k: ("***" if _is_secret(k) else v) for k, v in generator["env"].items()
        }
    return out


def _is_secret(key: str) -> bool:
    k = key.upper()
    return any(marker in k for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
