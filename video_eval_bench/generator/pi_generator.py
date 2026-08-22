"""
Agentic video generator: one `pi` run per seed, in an isolated workspace.

The agent is given a system prompt, a tool allowlist, optional skills and optional
custom-tool extensions — the four axes the benchmark ablates — plus the seed brief.
It works in its own directory and hands the finished video back by calling the
`submit_video` tool from the packaged `bench_tools.ts` extension.

## How a run ends, and how the video comes back

`terminate: true` on the submit tool does not stop the process; it only tells pi to
skip the follow-up LLM call after the tool batch. Python unblocks on process exit,
which `--print` mode reaches once the agent loop has no more work.

The result arrives through two channels, and both are required:

  * the bytes, at VEB_OUTPUT_PATH — a path this module pinned into the environment,
    so the agent never has to be told it or remember it;
  * the claim, as a `tool_execution_end` event for `submit_video` in the NDJSON —
    without which a stale file from an earlier attempt would read as a fresh result.

Because the extension finishes its copy before returning, and the event is emitted
after `execute()` resolves, the file is whole by the time we see the event. That is
what makes it safe to stop waiting early: after a submission we give pi
`exit_grace_seconds` to exit on its own, then kill it rather than let a chatty model
keep going.

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
from pathlib import Path
from typing import Dict, List, Optional

from video_eval_bench.config import BENCH_TOOLS_EXTENSION, SUBMIT_TOOL, PiGeneratorConfig
from video_eval_bench.dataset.seed import Seed
from video_eval_bench.judge.frames import video_frame_count
from video_eval_bench.pi_ndjson import is_error, parse_line, tool_result_details

logger = logging.getLogger(__name__)

# stdout is read in chunks and split on newlines here rather than with
# StreamReader.readline(), whose 64KiB line limit a fat tool-result event would
# breach. A pathological run with no newline at all is capped so it cannot eat
# memory unbounded.
_READ_CHUNK = 65536
_MAX_PENDING_LINE = 64 * 1024 * 1024


class PiGenerationError(RuntimeError):
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

        usage = event.get("usage")
        if isinstance(usage, dict) and usage:
            self.usage = usage


class PiGenerator:
    """Runs the pi agent once per seed and returns the video it submits."""

    def __init__(self, config: PiGeneratorConfig):
        self.config = config
        if shutil.which(config.pi_bin) is None and not Path(config.pi_bin).exists():
            raise RuntimeError(f"pi binary not found: {config.pi_bin!r}")
        self._metadata: Dict[str, dict] = {}

    # ── the generator contract ────────────────────────────────────────────────

    async def __call__(self, seed: Seed, output_dir: Path) -> str:
        cfg = self.config
        seed_dir = Path(output_dir) / seed.seed_id
        workspace = seed_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for src in cfg.copy_files:
            _stage(Path(src), workspace)

        output_path = Path(output_dir) / f"{seed.seed_id}.mp4"
        transcript_path = seed_dir / "transcript.jsonl"
        stderr_path = seed_dir / "pi_stderr.log"

        argv = self.build_argv(seed, workspace, output_path)
        env = self._build_env(seed, workspace, output_path)

        logger.info(
            f"[PiGenerator] {seed.seed_id}: {cfg.model.name} in {workspace} "
            f"(budget {cfg.timeout_seconds:g}s)"
        )

        started = time.monotonic()
        state = _RunState()
        returncode: Optional[int] = None
        error: Optional[str] = None

        with open(stderr_path, "wb") as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
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
                state.outcome = "timeout"
                await _kill_group(proc)
                error = (
                    f"pi run exceeded the {cfg.timeout_seconds:g}s budget"
                    f"{_stderr_tail(stderr_path)}"
                )
            except BaseException:
                # Cancellation included: never leave the tree running.
                await _kill_group(proc)
                raise
            finally:
                if proc.returncode is None:
                    await _kill_group(proc)

        elapsed = time.monotonic() - started
        metadata = self._write_run_json(
            seed_dir, seed, argv, state, returncode, elapsed, transcript_path, stderr_path
        )
        self._metadata[seed.seed_id] = metadata

        if error:
            raise PiGenerationError(error)

        return self._resolve_video(state, output_path, returncode, stderr_path, elapsed)

    def metadata_for(self, seed_id: str) -> dict:
        """Run metadata for the report (opt-in protocol used by run_bench)."""
        return self._metadata.get(seed_id, {})

    # ── argv / env ────────────────────────────────────────────────────────────

    def build_argv(self, seed: Seed, workspace: Path, output_path: Path) -> List[str]:
        """The full pi command line for one seed."""
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
        for skill in cfg.skills.paths:
            argv += ["--skill", str(Path(skill).resolve())]

        if not cfg.discover_extensions:
            argv.append("--no-extensions")
        # The handoff tool is not optional — without it the agent has no way to
        # deliver a result at all.
        argv += ["-e", str(BENCH_TOOLS_EXTENSION)]
        for extension in cfg.extensions:
            argv += ["-e", str(Path(extension).resolve())]

        if cfg.tools.no_builtin:
            # --no-builtin-tools drops the built-ins but keeps extension tools,
            # so the handoff tool survives on its own.
            argv.append("--no-builtin-tools")
        elif cfg.tools.builtin:
            # --tools is an allowlist over built-in *and* extension tools alike,
            # so the handoff tool has to be named or it is filtered out — and the
            # agent then generates a perfectly good video with no way to deliver
            # it. Observed, not theorised.
            argv += ["--tools", ",".join([*cfg.tools.builtin, SUBMIT_TOOL])]
        if cfg.tools.exclude:
            argv += ["--exclude-tools", ",".join(cfg.tools.exclude)]

        if not cfg.context_files:
            argv.append("--no-context-files")
        if not cfg.prompt_templates:
            argv.append("--no-prompt-templates")
        if cfg.approve_project:
            argv.append("--approve")

        argv += list(cfg.extra_args)
        argv.append(self.render_task(seed, workspace, output_path))
        return argv

    def render_task(self, seed: Seed, workspace: Path, output_path: Path) -> str:
        """The per-seed user message, rendered from the task template."""
        fields = dict(
            seed_id=seed.seed_id,
            category=seed.category,
            prompt=seed.prompt.strip(),
            workspace=str(workspace),
            output_path=str(output_path),
            reference_video=seed.reference_video or "",
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

    def _build_env(self, seed: Seed, workspace: Path, output_path: Path) -> dict:
        env = dict(os.environ)
        env.update({k: str(v) for k, v in self.config.env.items()})
        # Pinned, not passed as tool arguments: no prompt can redirect the output,
        # and the agent never has to carry the destination through compaction.
        env["VEB_OUTPUT_PATH"] = str(output_path)
        env["VEB_WORKSPACE"] = str(workspace)
        env["VEB_SEED_ID"] = seed.seed_id
        return env

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
                if self._grace_expired(state, now):
                    logger.info(
                        f"[PiGenerator] {seed.seed_id}: video submitted, pi still running after "
                        f"{cfg.exit_grace_seconds:g}s grace — terminating"
                    )
                    state.outcome = "early_terminated"
                    await _kill_group(proc)
                    break

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
        tick = self.config.heartbeat_seconds
        if state.submitted_at is not None:
            remaining = self.config.exit_grace_seconds - (time.monotonic() - state.submitted_at)
            tick = min(tick, max(remaining, 0.05))
        return tick

    def _grace_expired(self, state: _RunState, now: float) -> bool:
        return (
            state.submitted_at is not None
            and now - state.submitted_at >= self.config.exit_grace_seconds
        )

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


def _stage(src: Path, workspace: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, workspace / src.name, dirs_exist_ok=True)
    else:
        shutil.copy2(src, workspace / src.name)


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
