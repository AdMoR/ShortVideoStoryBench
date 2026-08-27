"""
Agentic video generator: one `pi` run per seed, in an isolated workspace.

The agent is given a system prompt, a tool allowlist, optional skills and optional
custom-tool extensions — the four axes the benchmark ablates — plus the seed brief.
It works in its own directory and hands the finished video back by calling the
`submit_video` tool from the packaged `bench_tools.ts` extension.

## How a run ends, and how the video comes back

Submitting does **not** end the run, and `submit_video` carries no `terminate`. The
agent may submit as often as it likes and the last successful submission wins, so
banking an early result costs it nothing. Python unblocks on process exit, which
`--print` mode reaches once the agent loop has no more work, or on `timeout_seconds`.

The result arrives through two channels, and both are required:

  * the bytes, at VEB_OUTPUT_PATH — a path this module pinned into the environment,
    so the agent never has to be told it or remember it;
  * the claim, as a `tool_execution_end` event for `submit_video` in the NDJSON —
    without which a stale file from an earlier attempt would read as a fresh result.

Because the extension finishes its copy before returning, and the event is emitted
after `execute()` resolves, the file is whole by the time we see the event.

A run that hits the budget having already submitted is a **success**, not a failure:
`outcome` records `timeout_after_submit` and the banked video is the result. The
earlier design — kill the agent `exit_grace_seconds` after its first submission —
forced an all-or-nothing choice between delivering and continuing to work, and a
real run spent an hour refining a finished video it never submitted at all.

## Where the agent runs

`sandbox=docker` puts the whole pi process in a throwaway container holding the
seed's workspace and nothing else, and gives the agent neutral paths — /workspace,
/out, /opt/veb — so the mount reveals nothing about the host. This is a
correctness feature, not a deployment one: on the host the workspace sits four
levels under the repo root with no path restriction on `read` or `bash`, and a
real run used that to read `dataset/rubric_c.yaml` before being scored against it
(runs/FINDINGS.md §4).

Under `sandbox=none` every path below is the host path and the argv is exactly
what it always was. `_SeedPaths` is the only place the two views differ.

## What must not be added

There is deliberately **no inactivity timeout**. pi imposes no default timeout on its
bash tool, so a legitimate generation may block silently for twenty minutes; a
timer on stdout silence would kill exactly the runs the benchmark exists to measure.
The only budget is the overall `timeout_seconds`, and progress is made visible with
heartbeat logging instead.

Two subprocess hazards are handled explicitly:

  * stderr is redirected to a file, never to a pipe we drain second. Draining pipes
    sequentially deadlocks once stderr's ~64KB buffer fills: pi blocks on write, we
    block on stdout, and the run burns its whole budget.
  * pi is spawned in its own session and every signal goes to the process *group*.
    Killing pi alone would orphan whatever its bash tool was running — a curl, an
    ffmpeg, a poll loop — and leak it into every later seed of the bench.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from video_eval_bench.config import BENCH_TOOLS_EXTENSION, SUBMIT_TOOL, PiGeneratorConfig
from video_eval_bench.dataset.seed import Seed, SeedReference
from video_eval_bench.generator.base import GenerationError, GenerationResult
from video_eval_bench.judge.frames import video_frame_count
from video_eval_bench.pi_ndjson import is_error, parse_line, tool_result_details

logger = logging.getLogger(__name__)

# stdout is read in chunks and split on newlines here rather than with
# StreamReader.readline(), whose 64KiB line limit a fat tool-result event would
# breach. A pathological run with no newline at all is capped so it cannot eat
# memory unbounded.
_READ_CHUNK = 65536
_MAX_PENDING_LINE = 64 * 1024 * 1024

# What the agent sees inside the sandbox. Neutral on purpose: the host layout is
# not something the agent under test should be able to read off its own argv.
WORKSPACE_MOUNT = "/workspace"
OUTPUT_MOUNT = "/out"
EXT_MOUNT = "/opt/veb/ext"
SKILLS_MOUNT = "/opt/veb/skills"

# Where a seed's reference images are staged. Relative to the workspace, so it
# reads the same on the host and in the container and needs no mount of its own.
REFERENCE_DIR = "references"


class PiGenerationError(GenerationError):
    """Generation failed for one seed. run_bench records it and continues."""


class _RunState:
    """What the event stream told us about a run."""

    def __init__(self) -> None:
        self.turns = 0
        self.tool_calls: Dict[str, int] = {}
        self.usage: dict = {}
        self.submitted_path: Optional[str] = None
        self.submitted_at: Optional[float] = None
        self.submit_notes: str = ""
        self.current_tool: Optional[str] = None
        self.last_event_type: Optional[str] = None
        self.outcome = "exited"

    def observe(self, event: dict) -> None:
        kind = event.get("type")
        self.last_event_type = kind

        if kind == "turn_start":
            self.turns += 1
        elif kind == "tool_execution_start":
            name = str(event.get("toolName", "?"))
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
            self.current_tool = name
        elif kind == "tool_execution_end":
            self.current_tool = None
            if event.get("toolName") == SUBMIT_TOOL and not is_error(event):
                details = tool_result_details(event)
                path = details.get("path")
                if path:
                    self.submitted_path = str(path)
                    self.submitted_at = time.monotonic()
                    self.submit_notes = str(details.get("notes", ""))

        # pi emits an all-zeros usage block on message_start, so "non-empty" is
        # not enough — accepting it resets the count and reports a long run as
        # having used no tokens at all.
        usage = event.get("usage")
        if isinstance(usage, dict) and usage.get("totalTokens"):
            self.usage = usage


class _SeedPaths:
    """
    Where each thing lives on the host, and where the agent sees it.

    Under `sandbox=none` the two are the same object and every consumer below is
    unchanged. Under `sandbox=docker` the agent side is a neutral path — the host
    layout is not something the agent under test should be able to read off its
    own command line, and /workspace has nothing above it to walk into.

    The output is the one asymmetric case. On the host the agent writes straight
    to `runs/<id>/<seed>.mp4`; mounting that directory into the container would
    hand the agent `videos.yaml` and every earlier seed's video along with it, so
    a sandboxed run writes into an empty per-seed staging directory and the
    harness moves the file afterwards.
    """

    def __init__(self, cfg, seed_dir: Path, output_path: Path):
        self.sandboxed = cfg.sandbox.kind == "docker"

        self.workspace_host = seed_dir / "workspace"
        self.output_host = output_path
        self.staging_host: Optional[Path] = None

        extensions = [BENCH_TOOLS_EXTENSION, *(Path(e).resolve() for e in cfg.extensions)]
        skills = [Path(sk).resolve() for sk in cfg.skills.paths]

        if not self.sandboxed:
            self.workspace_agent = self.workspace_host
            self.output_agent = self.output_host
            self.extensions: List[Tuple[Path, Path]] = [(e, e) for e in extensions]
            self.skills: List[Tuple[Path, Path]] = [(sk, sk) for sk in skills]
            return

        self.staging_host = seed_dir / "out"
        self.workspace_agent = Path(WORKSPACE_MOUNT)
        self.output_agent = Path(OUTPUT_MOUNT) / "video.mp4"
        self.extensions = [
            (host, Path(EXT_MOUNT) / name)
            for host, name in zip(extensions, _unique_names(e.name for e in extensions))
        ]
        self.skills = [
            (host, Path(SKILLS_MOUNT) / name)
            for host, name in zip(skills, _unique_names(sk.name for sk in skills))
        ]

    def mounts(self) -> List[str]:
        """The `-v` arguments for one seed, in `src:dst[:ro]` form."""
        binds = [
            f"{self.workspace_host}:{self.workspace_agent}",
            f"{self.staging_host}:{OUTPUT_MOUNT}",
        ]
        binds += [f"{host}:{agent}:ro" for host, agent in self.extensions]
        binds += [f"{host}:{agent}:ro" for host, agent in self.skills]
        return binds


def _unique_names(names) -> List[str]:
    """
    Basenames made unique, so two skills called `video` do not shadow each other.

    Collisions are suffixed rather than rejected: a skills= arm naming two
    directories with the same basename is a legitimate thing to want, and losing
    one of them silently would make the arm measure the wrong bundle.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{Path(name).stem}-{count}{Path(name).suffix}")
    return out


class PiGenerator:
    """Runs the pi agent once per seed and returns the video it submits."""

    def __init__(self, config: PiGeneratorConfig):
        self.config = config
        # Sandboxed, pi lives in the image, not on the host — so the binary that
        # has to exist here is docker's.
        sandboxed = config.sandbox.kind == "docker"
        label = "docker" if sandboxed else "pi"
        required = config.sandbox.docker_bin if sandboxed else config.pi_bin
        if shutil.which(required) is None and not Path(required).exists():
            raise RuntimeError(f"{label} binary not found: {required!r}")

    # ── the generator contract ────────────────────────────────────────────────

    async def __call__(self, seed: Seed, output_dir: Path) -> GenerationResult:
        cfg = self.config
        seed_dir = Path(output_dir) / seed.seed_id
        output_path = Path(output_dir) / f"{seed.seed_id}.mp4"
        paths = _SeedPaths(cfg, seed_dir, output_path)

        paths.workspace_host.mkdir(parents=True, exist_ok=True)
        if paths.staging_host is not None:
            paths.staging_host.mkdir(parents=True, exist_ok=True)
        for src in cfg.copy_files:
            _stage(Path(src), paths.workspace_host)
        _stage_references(seed, paths.workspace_host)

        transcript_path = seed_dir / "transcript.jsonl"
        stderr_path = seed_dir / "pi_stderr.log"

        argv = self.build_argv(seed, paths)
        container = f"veb-{seed.seed_id}-{uuid.uuid4().hex[:8]}" if paths.sandboxed else None
        if container is not None:
            argv = self._docker_argv(argv, seed, paths, container)
        env = self._process_env(seed, paths)

        logger.info(
            f"[PiGenerator] {seed.seed_id}: {cfg.model.name} in {paths.workspace_host} "
            f"({'container ' + container if container else 'on the host'}, "
            f"budget {cfg.timeout_seconds:g}s)"
        )

        started = time.monotonic()
        state = _RunState()
        returncode: Optional[int] = None
        error: Optional[str] = None

        with open(stderr_path, "wb") as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(paths.workspace_host),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_file,
                # Own session, so signals reach the whole process tree.
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(
                    self._pump(proc, state, transcript_path, seed),
                    timeout=cfg.timeout_seconds,
                )
                returncode = await proc.wait()
            except asyncio.TimeoutError:
                await _kill_group(proc)
                await self._remove_container(container)
                if state.submitted_path is not None:
                    # Submission no longer ends the run, so an agent that keeps
                    # working until the budget expires is normal, not a failure.
                    # It banked a video; that video is the result.
                    state.outcome = "timeout_after_submit"
                    logger.info(
                        f"[PiGenerator] {seed.seed_id}: budget expired with a submitted "
                        f"video — keeping the last submission"
                    )
                else:
                    state.outcome = "timeout"
                    error = (
                        f"pi run exceeded the {cfg.timeout_seconds:g}s budget"
                        f"{_stderr_tail(stderr_path)}"
                    )
            except BaseException:
                # Cancellation included: never leave the tree running.
                await _kill_group(proc)
                await self._remove_container(container)
                raise
            finally:
                if proc.returncode is None:
                    await _kill_group(proc)
                    await self._remove_container(container)

        elapsed = time.monotonic() - started
        _collect_output(paths)
        metadata = self._write_run_json(
            seed_dir, seed, argv, state, returncode, elapsed, transcript_path, stderr_path
        )

        if error:
            raise PiGenerationError(error, metadata=metadata)

        # _resolve_video raises from four places, and a run that burned its whole
        # budget before failing is the one whose turn count and token usage are
        # most worth reporting — so every exit carries the metadata out.
        try:
            video_path = self._resolve_video(
                state, output_path, returncode, stderr_path, elapsed
            )
        except PiGenerationError as exc:
            exc.metadata = metadata
            raise

        # duration_seconds stays None on purpose: `elapsed` is generation only,
        # while the report's duration has always meant generate+judge. pi's own
        # cost is in metadata, which is where the report already looks for it.
        return GenerationResult(
            seed_id=seed.seed_id, video_path=video_path, metadata=metadata
        )

    # ── argv / env ────────────────────────────────────────────────────────────

    def build_argv(self, seed: Seed, paths: "_SeedPaths") -> List[str]:
        """The full pi command line for one seed, in the agent's view of the world."""
        cfg = self.config
        argv: List[str] = [cfg.pi_bin]

        if cfg.model.provider:
            argv += ["--provider", cfg.model.provider]
        argv += ["--model", cfg.model.name]
        if cfg.model.thinking:
            argv += ["--thinking", cfg.model.thinking]

        argv += ["--print", "--mode", "json", "--no-session"]
        if cfg.offline:
            argv.append("--offline")

        argv += ["--system-prompt", cfg.system_prompt.system()]
        for extra in cfg.system_prompt.appended():
            argv += ["--append-system-prompt", extra]

        if not cfg.skills.discover:
            argv.append("--no-skills")
        for _, skill in paths.skills:
            argv += ["--skill", str(skill)]

        if not cfg.discover_extensions:
            argv.append("--no-extensions")
        # The handoff tool is not optional — without it the agent has no way to
        # deliver a result at all. _SeedPaths puts it first, ahead of the
        # configured extensions.
        for _, extension in paths.extensions:
            argv += ["-e", str(extension)]

        if cfg.tools.no_builtin:
            # --no-builtin-tools drops the built-ins but keeps extension tools,
            # so the handoff tool survives on its own.
            argv.append("--no-builtin-tools")
        elif cfg.tools.builtin:
            # --tools is an allowlist over built-in *and* extension tools alike,
            # so every custom tool has to be named or it is silently filtered
            # out. Both halves of this were observed on real runs: first the
            # agent produced a video with no way to deliver it, then it hunted
            # nine turns for a generator that was loaded the whole time.
            allowed = [*cfg.tools.builtin, SUBMIT_TOOL, *cfg.extension_tools]
            argv += ["--tools", ",".join(dict.fromkeys(allowed))]
        if cfg.tools.exclude:
            argv += ["--exclude-tools", ",".join(cfg.tools.exclude)]

        if not cfg.context_files:
            argv.append("--no-context-files")
        if not cfg.prompt_templates:
            argv.append("--no-prompt-templates")
        if cfg.approve_project:
            argv.append("--approve")

        argv += list(cfg.extra_args)
        argv.append(self.render_task(seed, paths))
        return argv

    def render_task(self, seed: Seed, paths: "_SeedPaths") -> str:
        """The per-seed user message, rendered from the task template."""
        fields = dict(
            seed_id=seed.seed_id,
            category=seed.category,
            prompt=seed.prompt.strip(),
            workspace=str(paths.workspace_agent),
            output_path=str(paths.output_agent),
            references=_render_references(seed),
            # The agent has no clock of its own. Without this it cannot tell a
            # cheap refinement from one that costs it the whole run, and a real
            # run ended with a finished video it never got around to submitting.
            budget_minutes=max(1, round(self.config.timeout_seconds / 60)),
        )
        try:
            return self.config.system_prompt.task().format(**fields)
        except (KeyError, IndexError) as exc:
            # A shell snippet or JSON example in a custom template will contain
            # braces that str.format tries to substitute. Say so, rather than
            # failing every seed with a bare KeyError.
            raise PiGenerationError(
                f"task template refers to {exc} — available placeholders are "
                f"{sorted(fields)}. Literal braces must be doubled: {{{{ and }}}}."
            ) from exc

    def _agent_env(self, seed: Seed, paths: "_SeedPaths") -> Dict[str, str]:
        """
        The variables the agent's process needs, wherever it ends up running.

        VEB_OUTPUT_PATH and VEB_WORKSPACE are pinned here rather than passed as
        tool arguments: no prompt can redirect the output, and the agent never has
        to carry the destination through context compaction.
        """
        env = {k: str(v) for k, v in self.config.env.items()}
        env["VEB_OUTPUT_PATH"] = str(paths.output_agent)
        env["VEB_WORKSPACE"] = str(paths.workspace_agent)
        env["VEB_SEED_ID"] = seed.seed_id
        return env

    def _process_env(self, seed: Seed, paths: "_SeedPaths") -> dict:
        """
        The environment of the process we actually spawn.

        Sandboxed that process is the docker client, which needs the harness's own
        environment (DOCKER_HOST and friends) and nothing else — the agent's
        variables travel as `-e` flags instead, so they are visible in the argv the
        report records rather than inherited invisibly.
        """
        env = dict(os.environ)
        if not paths.sandboxed:
            env.update(self._agent_env(seed, paths))
        return env

    # ── the sandbox ───────────────────────────────────────────────────────────

    def _docker_argv(
        self, argv: List[str], seed: Seed, paths: "_SeedPaths", name: str
    ) -> List[str]:
        """Wrap a pi command line in the container it should run inside."""
        sandbox = self.config.sandbox
        docker: List[str] = [
            sandbox.docker_bin,
            "run",
            "--rm",
            # tini as pid 1: pi's bash tool spawns children, and without an init
            # a killed container leaves them to be reaped by nobody.
            "--init",
            # Named so a run killed at the budget can still be cleaned up —
            # SIGKILLing the docker client does not stop the container.
            "--name",
            name,
            # No -t. stdout has to stay a clean NDJSON pipe; a tty would line-wrap
            # the event stream _pump is parsing.
            "--network",
            sandbox.network,
        ]
        for host, address in sandbox.extra_hosts.items():
            docker += ["--add-host", f"{host}:{address}"]
        docker += ["--user", sandbox.run_as()]
        if sandbox.env_file is not None:
            docker += ["--env-file", str(Path(sandbox.env_file).resolve())]
        for passthrough in sandbox.env_passthrough:
            if passthrough in os.environ:
                docker += ["-e", f"{passthrough}={os.environ[passthrough]}"]
        for key, value in self._agent_env(seed, paths).items():
            docker += ["-e", f"{key}={value}"]
        for mount in [*paths.mounts(), *sandbox.extra_mounts]:
            docker += ["-v", mount]
        docker += ["-w", str(paths.workspace_agent)]
        docker += list(sandbox.docker_args)
        docker.append(sandbox.image)
        return docker + argv

    async def _remove_container(self, name: Optional[str]) -> None:
        """
        Stop the container a killed docker client left behind.

        `_kill_group` signals the client, not the container: the daemon owns the
        latter, and it keeps running with whatever the agent's bash tool started
        inside it. With `video_backend=wangp` that is a live GPU job, so this is
        the difference between a timed-out seed and a wedged server.
        """
        if name is None:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.sandbox.docker_bin,
                "rm",
                "-f",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
        except (OSError, asyncio.TimeoutError) as exc:
            # Already gone is the common case and not worth a warning; anything
            # else is worth saying out loud but never worth failing the seed for.
            logger.warning(f"[PiGenerator] could not remove container {name}: {exc}")

    # ── the read loop ─────────────────────────────────────────────────────────

    async def _pump(
        self, proc, state: _RunState, transcript_path: Path, seed: Seed
    ) -> None:
        """
        Drain stdout to the transcript, tracking events, until pi exits — or until
        the submission grace period expires.

        The loop ticks on a short timeout purely so heartbeats and the grace check
        happen; a tick expiring is not an error and never ends the run. Chunks are
        split on newlines here rather than using readline(), whose line limit a
        large tool-result event would breach.
        """
        cfg = self.config
        started = time.monotonic()
        last_heartbeat = started
        pending = b""

        with open(transcript_path, "wb") as transcript:
            while True:
                tick = self._tick(state)
                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(_READ_CHUNK), timeout=tick)
                except asyncio.TimeoutError:
                    chunk = None

                if chunk:
                    transcript.write(chunk)
                    transcript.flush()
                    pending += chunk
                    if b"\n" in pending:
                        *lines, pending = pending.split(b"\n")
                        for raw in lines:
                            event = parse_line(raw.decode("utf-8", errors="replace"))
                            if event is not None:
                                state.observe(event)
                    elif len(pending) > _MAX_PENDING_LINE:
                        # No newline in 64MB: stop buffering, keep transcribing.
                        pending = b""
                elif chunk == b"":
                    break  # EOF: pi has closed stdout and is exiting

                now = time.monotonic()
                if now - last_heartbeat >= cfg.heartbeat_seconds:
                    last_heartbeat = now
                    logger.info(
                        f"[PiGenerator] {seed.seed_id}: {now - started:.0f}s elapsed, "
                        f"turn {state.turns}, tool={state.current_tool or '-'}, "
                        f"last event={state.last_event_type or '-'}"
                    )

            # Trailing bytes with no final newline.
            if pending:
                event = parse_line(pending.decode("utf-8", errors="replace"))
                if event is not None:
                    state.observe(event)

    def _tick(self, state: _RunState) -> float:
        """How long to wait for the next chunk before doing housekeeping."""
        return self.config.heartbeat_seconds

    # ── result ────────────────────────────────────────────────────────────────

    def _resolve_video(
        self,
        state: _RunState,
        output_path: Path,
        returncode: Optional[int],
        stderr_path: Path,
        elapsed: float,
    ) -> str:
        """Both channels must agree: a submission event, and a usable file."""
        if state.submitted_path is None:
            if returncode not in (0, None):
                raise PiGenerationError(
                    f"pi exited with {returncode} without submitting a video"
                    f"{_stderr_tail(stderr_path)}"
                )
            raise PiGenerationError(
                f"the agent finished after {elapsed:.0f}s without calling {SUBMIT_TOOL} — "
                "no video was delivered"
            )

        if not output_path.exists():
            raise PiGenerationError(
                f"{SUBMIT_TOOL} reported success but {output_path} does not exist"
            )

        frames = video_frame_count(str(output_path))
        if frames <= 0:
            raise PiGenerationError(
                f"submitted file {output_path} is not a readable video "
                "(OpenCV found no frames)"
            )

        if returncode not in (0, None) and state.outcome != "early_terminated":
            # The video exists and the agent claimed it; a bad exit code after a
            # good submission is an anomaly to record, not a reason to discard it.
            # (An early termination is us killing pi on purpose, not an anomaly.)
            logger.warning(
                f"[PiGenerator] pi exited with {returncode} after a successful submission; "
                f"keeping {output_path}"
            )
        logger.info(
            f"[PiGenerator] {output_path.name}: {frames} frames, {elapsed:.0f}s, "
            f"{state.turns} turns, outcome={state.outcome}"
        )
        return str(output_path)

    def _write_run_json(
        self,
        seed_dir: Path,
        seed: Seed,
        argv: List[str],
        state: _RunState,
        returncode: Optional[int],
        elapsed: float,
        transcript_path: Path,
        stderr_path: Path,
    ) -> dict:
        metadata = {
            "seed_id": seed.seed_id,
            "model": self.config.model.name,
            "provider": self.config.model.provider,
            "argv": argv,
            "returncode": returncode,
            "outcome": state.outcome,
            "duration_seconds": round(elapsed, 2),
            "turns": state.turns,
            "tool_calls": state.tool_calls,
            "usage": state.usage,
            "submitted_path": state.submitted_path,
            "submit_notes": state.submit_notes,
            "transcript": str(transcript_path),
            "stderr": str(stderr_path),
        }
        (seed_dir / "pi_run.json").write_text(json.dumps(metadata, indent=2))
        return metadata


# ── helpers ───────────────────────────────────────────────────────────────────


def _collect_output(paths: "_SeedPaths") -> None:
    """
    Move a sandboxed run's video out of its staging directory to where the rest
    of the harness expects it.

    A no-op on the host, where the agent wrote straight to the final path. The
    staging step exists so the container can be given an empty directory instead
    of the run directory, which holds `videos.yaml` and every earlier seed's
    video — the agent under test has no business reading either.
    """
    if paths.staging_host is None:
        return
    produced = paths.staging_host / "video.mp4"
    if produced.exists():
        paths.output_host.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(paths.output_host))
    # Leave nothing behind if it is empty; anything the agent put there that is
    # not the video stays, because it is evidence about a seed that failed.
    try:
        paths.staging_host.rmdir()
    except OSError:
        pass


def _stage(src: Path, workspace: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, workspace / src.name, dirs_exist_ok=True)
    else:
        shutil.copy2(src, workspace / src.name)


def _stage_references(seed: Seed, workspace: Path) -> None:
    """
    Copy a seed's reference images into `<workspace>/references/<id><suffix>`.

    Inside the workspace rather than beside it, because that is the one directory
    both worlds already agree on: it is bind-mounted at /workspace under
    `sandbox=docker`, so this needs no new mount, and `generate_video` refuses
    any path outside it (`inWorkspace` in wangp_tools.ts / fake_video_tools.ts).

    Renaming to the reference id does two jobs. It keeps host filenames out of
    what the agent sees, and it makes `references/maya.png` — the path the brief
    prints — the same path the tool call takes.
    """
    if not seed.references:
        return
    directory = workspace / REFERENCE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for ref in seed.references:
        shutil.copy2(ref.path, directory / _reference_name(ref))


def _reference_name(ref: SeedReference) -> str:
    return f"{ref.id}{ref.path.suffix}"


def _render_references(seed: Seed) -> str:
    """
    The brief's reference section, or "" for a seed that carries none.

    Empty rather than "no references supplied" so one task template serves both
    kinds of seed: a bare seed's brief reads exactly as it did before this
    existed, and a `skills=`/`system_prompt=` ablation measured on bare seeds is
    not silently comparing against a different prompt.

    Paths are workspace-relative, which reads identically on the host and in the
    container and leaks no host layout into the command line.
    """
    if not seed.references:
        return ""
    lines = [
        "",
        "## Reference images",
        "",
        f"{len(seed.references)} reference image(s) are in your working directory, under "
        f"`{REFERENCE_DIR}/`. The generated video must match them.",
        "",
    ]
    for ref in seed.references:
        path = f"{REFERENCE_DIR}/{_reference_name(ref)}"
        lines.append(f'- `{path}` — {ref.role}, "{ref.label}"')
        for line in ref.description.strip().splitlines():
            lines.append(f"  {line.strip()}")
    lines += [
        "",
        "Refer to each subject by its label in the prompts you write. If your video "
        "tool can be conditioned on images, use these — describing them in words "
        "instead is not the same thing.",
        # Trailing empty element, so the block ends with the blank line the
        # template's next heading expects. Without references the whole block is
        # "" and the brief is byte-for-byte what it was before seeds could carry
        # any — a prompt ablation measured on bare seeds must not shift under it.
        "",
    ]
    return "\n".join(lines)


async def _kill_group(proc, grace: float = 5.0) -> None:
    """
    SIGTERM then SIGKILL the whole process group.

    Signalling the group rather than the pid is the point: pi's bash tool may have
    a long-running child, and once we kill pi it can no longer clean up after
    itself. `start_new_session=True` makes the group id equal the pid, so the pid
    is used directly — asking for the pgid after exit would race.
    """
    if proc.returncode is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        if wait is None:
            break
        try:
            await asyncio.wait_for(proc.wait(), timeout=wait)
            return
        except asyncio.TimeoutError:
            continue
    try:
        await proc.wait()
    except ProcessLookupError:
        pass


def _stderr_tail(path: Path, limit: int = 500) -> str:
    try:
        text = path.read_text(errors="replace").strip()
    except OSError:
        return ""
    return f": {text[-limit:]}" if text else ""
