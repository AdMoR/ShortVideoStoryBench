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
import os
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

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
PI_EXT_DIR = PACKAGE_DIR / "generator" / "pi_ext"
BENCH_TOOLS_EXTENSION = PI_EXT_DIR / "bench_tools.ts"


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


class SandboxConfig(_Base):
    """
    Where the agent's `pi` process runs — on the host, or jailed in a container.

    This is not an ablation axis; it is a correctness one. With `kind="none"` the
    agent has the operator's whole filesystem, its workspace sits four levels
    under the repo root, and a real run walked up and read `dataset/rubrics.yaml`
    before scoring itself (runs/FINDINGS.md §4). `kind="docker"` mounts the seed's
    workspace and nothing else, so there is nothing above it to walk into.

    Only the *generator's* pi is sandboxed. The judge runs `--no-tools`, so it has
    no filesystem reach worth jailing.
    """

    kind: Literal["none", "docker"] = "none"
    docker_bin: str = "docker"
    image: str = Field(
        default="veb-pi:latest", description="Built by docker/build.sh."
    )
    network: str = Field(
        default="bridge",
        description=(
            "Docker network mode. `bridge` keeps the agent off the host's loopback; "
            "`host` is the fallback if the generation endpoints turn out to be "
            "unroutable from a bridge network."
        ),
    )
    extra_hosts: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "name -> address, as --add-host. The generation hosts usually need this: "
            "Docker strips the host's loopback resolver from a container's "
            "resolv.conf, so a name that only resolves there is unreachable inside."
        ),
    )
    env_file: Optional[Path] = Field(
        default=None,
        description="Passed as --env-file. Where the LLM and generation-server keys come from.",
    )
    env_passthrough: List[str] = Field(
        default_factory=list,
        description=(
            "Names forwarded from the harness's own environment, on top of env_file. "
            "The tighter alternative to putting every key in env_file."
        ),
    )
    user: Optional[str] = Field(
        default=None,
        description=(
            "--user value. Defaults to the harness's own uid:gid, so files the agent "
            "writes into the mounted workspace come back owned by the operator."
        ),
    )
    extra_mounts: List[str] = Field(
        default_factory=list,
        description=(
            'Raw "src:dst[:ro]" bind mounts, on top of the ones the harness derives. '
            "A video_backend arm whose extension needs a data file adds it here."
        ),
    )
    docker_args: List[str] = Field(
        default_factory=list,
        description="Escape hatch appended to `docker run` — --memory, --pids-limit, ...",
    )

    def run_as(self) -> str:
        return self.user or f"{os.getuid()}:{os.getgid()}"

    @model_validator(mode="after")
    def _docker_needs_env_file(self) -> "SandboxConfig":
        """
        A sandboxed run with no env file reaches no model and no generation server.

        Caught here rather than at runtime because the symptom is a per-seed
        authentication failure many minutes in, once per seed, with the real cause
        several layers down in pi's stderr.
        """
        if self.kind == "docker" and self.env_file is None:
            raise ValueError(
                "sandbox.kind='docker' needs sandbox.env_file: the container starts "
                "with an empty environment, so the model provider and the generation "
                "server credentials have to come from somewhere. Copy .env.example to "
                ".env, or set sandbox.env_passthrough and point env_file at an empty file."
            )
        if self.env_file is not None and not Path(self.env_file).exists():
            raise ValueError(
                f"sandbox.env_file does not exist: {self.env_file}. "
                "Copy .env.example to .env and fill in the keys you need."
            )
        return self


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
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    extensions: List[Path] = Field(
        default_factory=list,
        description="Extra pi extension files (.ts) providing custom tools.",
    )
    extension_tools: List[str] = Field(
        default_factory=list,
        description=(
            "Tool names those extensions register. pi's --tools allowlist covers "
            "extension tools too, so anything not named here is filtered out and "
            "the agent never sees it."
        ),
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
        description=(
            "Unused. Submitting no longer ends a run — the agent may submit repeatedly "
            "and the last one wins — so there is no post-submission grace to enforce. "
            "Kept so existing configs and sweeps still load."
        ),
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
    def _pi_bin_is_resolvable_where_it_runs(self) -> "PiGeneratorConfig":
        """
        Sandboxed, `pi_bin` is looked up inside the image, not on the host.

        A host path here — the natural thing to write, and what the test fixtures
        use — would exec-fail once per seed with docker's own error, several
        layers away from the setting that caused it.
        """
        if self.sandbox.kind == "docker" and Path(self.pi_bin).name != self.pi_bin:
            raise ValueError(
                f"pi_bin={self.pi_bin!r} is a host path, but sandbox.kind='docker' runs pi "
                "inside the image, where that path does not exist. Use a bare name on PATH "
                "in the image (the default, 'pi')."
            )
        return self

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
    def _extension_tools_are_declared(self) -> "PiGeneratorConfig":
        """
        An extension whose tools are not in the allowlist is loaded and then
        filtered out — the agent is never told the tool exists.

        Caught here because the symptom is silent and expensive: observed on a
        real run, where the agent spent nine turns hunting for a generator that
        was loaded the whole time, then went and read the grading rubric instead.
        """
        if self.extensions and not self.tools.no_builtin and self.tools.builtin:
            if not self.extension_tools:
                raise ValueError(
                    f"{len(self.extensions)} extension(s) are configured but "
                    "extension_tools is empty. pi's --tools allowlist covers extension "
                    "tools, so their tools would be filtered out and the agent would "
                    "never see them. List the tool names the extensions register."
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


class ExternalGeneratorConfig(_Base):
    """
    Videos generated elsewhere, scored here.

    Nothing is generated: each seed's video comes from a manifest. Because every
    run also *writes* that format, this is equally the replay arm — pointing it
    at a finished run's `videos.yaml` re-judges those videos without repeating
    the generation that produced them.
    """

    kind: Literal["external"] = "external"
    manifest: Path = Field(description="YAML manifest mapping seed_id -> video file")
    # Not just `copy`: that name shadows a BaseModel attribute and pydantic warns.
    copy_videos: bool = Field(
        default=True,
        description=(
            "Copy each video into the run directory (self-contained, archivable) "
            "rather than symlinking it. Symlinks suit replaying a run that is "
            "staying put, or a batch too large to duplicate."
        ),
    )
    label: str = Field(
        default="",
        description=(
            "Names this batch in the report's choices, so veb-compare can tell two "
            "imports apart — both are `generator=external` otherwise. Defaults to "
            "the manifest's own label."
        ),
    )

    @model_validator(mode="after")
    def _manifest_exists(self) -> "ExternalGeneratorConfig":
        """A missing manifest should fail now, not after the dataset loads."""
        # Hydra's mandatory-value sentinel survives to_container(), so say what to
        # pass rather than reporting a file literally named "???" as missing.
        if str(self.manifest) == "???":
            raise ValueError(
                "this arm needs a manifest of videos to judge; pass "
                "generator.manifest=<path to videos.yaml> "
                "(a previous run's videos.yaml works — that is how replay works)"
            )
        if not Path(self.manifest).expanduser().exists():
            raise ValueError(f"manifest does not exist: {self.manifest}")
        return self


GeneratorConfig = Annotated[
    Union[PiGeneratorConfig, MockGeneratorConfig, ExternalGeneratorConfig],
    Field(discriminator="kind"),
]


class JudgeConfig(_Base):
    """Which vision LLM grades the videos."""

    backend: Literal["mock", "pi", "litellm", "openai"] = "mock"
    media: Literal["frames", "video"] = Field(
        default="frames",
        description=(
            "What the judge is shown. 'frames' samples n_frames stills and works "
            "with any vision model. 'video' sends the clip itself, so motion, "
            "timing and cut rhythm are judged rather than inferred — it needs "
            "backend=openai against an endpoint that accepts video."
        ),
    )
    model: Optional[str] = None
    provider: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    pi_bin: str = "pi"
    timeout_seconds: float = 120.0
    # Per criterion, not per video. A failed call scores the criterion zero, so
    # retrying protects the number itself, not just the run.
    attempts: int = Field(default=3, ge=1)
    retry_backoff_seconds: float = Field(default=5.0, ge=0)
    n_frames: int = Field(default=8, gt=0)

    @model_validator(mode="after")
    def _video_needs_a_backend_that_can_send_it(self) -> "JudgeConfig":
        """
        Reject media=video on a backend that cannot carry it, here rather than
        at the first criterion call.

        A backend call that raises is recorded as a failed criterion, which
        scores ZERO — so getting this wrong would not error out, it would quietly
        publish a run where every video scored 0/100.
        """
        if self.media == "video" and self.backend not in ("openai", "mock"):
            raise ValueError(
                f"judge.media=video needs a backend that can send a clip; "
                f"judge.backend={self.backend!r} can only send frames. "
                "Use judge.backend=openai (with judge.api_base set), or "
                "judge.media=frames."
            )
        if self.backend == "openai" and not self.api_base:
            raise ValueError("judge.backend=openai needs judge.api_base set")
        return self


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
    """
    Build the generator callable for a validated generator config.

    Dispatch is explicit on purpose: a fall-through default would quietly build
    the wrong generator for a kind nobody wired up, and the symptom — an agent
    run where an import was asked for — costs an hour to notice.
    """
    if isinstance(cfg, MockGeneratorConfig):
        from video_eval_bench.generator.mock_generator import MockGenerator

        return MockGenerator(n_frames=cfg.n_frames, fps=cfg.fps, size=(cfg.width, cfg.height))

    if isinstance(cfg, ExternalGeneratorConfig):
        from video_eval_bench.generator.external_generator import ExternalGenerator

        return ExternalGenerator(
            manifest=cfg.manifest, copy=cfg.copy_videos, label=cfg.label
        )

    if isinstance(cfg, PiGeneratorConfig):
        from video_eval_bench.generator.pi_generator import PiGenerator

        return PiGenerator(cfg)

    raise TypeError(f"no generator for config of type {type(cfg).__name__}")


def build_judge(cfg: JudgeConfig, dataset):
    """Build the VideoJudge for a validated judge config."""
    from video_eval_bench.judge.agent import VideoJudge
    from video_eval_bench.judge.llm import (
        LLMConfig,
        LiteLlmBackend,
        MockBackend,
        OpenAIBackend,
    )

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
                attempts=cfg.attempts,
                retry_backoff_seconds=cfg.retry_backoff_seconds,
            )
        )
    elif cfg.backend == "openai":
        backend = OpenAIBackend(
            LLMConfig(
                model=cfg.model or "",
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                timeout_seconds=cfg.timeout_seconds,
            )
        )
    else:
        backend = LiteLlmBackend(
            LLMConfig(
                model=cfg.model or "openai/gpt-4o",
                api_base=cfg.api_base,
                api_key=cfg.api_key,
                timeout_seconds=cfg.timeout_seconds,
            )
        )
    return VideoJudge(
        backend=backend, dataset=dataset, n_frames=cfg.n_frames, media=cfg.media
    )


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
