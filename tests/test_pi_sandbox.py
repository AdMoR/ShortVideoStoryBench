"""
The sandbox: what the agent's container is given, and what it is not.

These are argv tests, not integration tests — no container is started. That is
the right level, because the property under test is a property of the command
line: every path the agent can see is one the harness chose to mount, and the
host layout appears nowhere it could be read off.

Why it matters is in runs/FINDINGS.md §4. On the host, with `tools=full`, a real
run walked out of its workspace and read `dataset/rubrics.yaml` — then scored
itself against it.
"""

from pathlib import Path

import pytest

from video_eval_bench.config import (
    BENCH_TOOLS_EXTENSION,
    PiGeneratorConfig,
    SandboxConfig,
    SkillsConfig,
)
from video_eval_bench.generator.pi_generator import PiGenerator, _SeedPaths

from test_pi_generator import argv_for, make_seed, paths_for


# Distinctive on purpose: pytest names its tmp directories after the test, so a
# plausible-looking value like "secret" matches the path this very test runs in.
SECRET_VALUE = "wangp-key-must-not-appear-in-argv"


@pytest.fixture
def env_file(tmp_path) -> Path:
    path = tmp_path / ".env"
    path.write_text(f"WANGP_API_KEY={SECRET_VALUE}\n")
    return path


@pytest.fixture
def sandboxed(pi_config, env_file):
    """A config whose agent runs in a container, with everything else default."""

    def build(**overrides) -> PiGeneratorConfig:
        sandbox = overrides.pop(
            "sandbox",
            SandboxConfig(
                kind="docker",
                env_file=env_file,
                extra_hosts={"gx10-cbc5": "100.64.0.1"},
            ),
        )
        # pi_bin is resolved inside the image, so it is a bare name here — unlike
        # the host fixtures, which point at the fake binary by absolute path.
        overrides.setdefault("pi_bin", "pi")
        return pi_config(sandbox=sandbox, **overrides)

    return build


def docker_argv(config, run_dir: Path) -> list:
    paths = paths_for(config, run_dir)
    generator = PiGenerator(config)
    return generator._docker_argv(
        generator.build_argv(make_seed(), paths), make_seed(), paths, "veb-test-0000"
    )


def docker_flags(argv) -> list:
    """
    Just the `docker run` half of the command line.

    Scoped deliberately: pi has its own `-e` (load an extension), and reading the
    whole argv would parse those as environment variables.
    """
    return argv[: argv.index("veb-pi:latest")]


def mounts_of(argv) -> dict:
    """destination -> source, for every -v in the command line."""
    flags = docker_flags(argv)
    binds = [value for flag, value in zip(flags, flags[1:]) if flag == "-v"]
    return {bind.split(":")[1]: bind.split(":")[0] for bind in binds}


def env_of(argv) -> dict:
    flags = docker_flags(argv)
    return dict(
        value.split("=", 1) for flag, value in zip(flags, flags[1:]) if flag == "-e"
    )


# ── the container's shape ─────────────────────────────────────────────────────


def test_container_is_named_and_disposable(sandboxed, tmp_path):
    """
    --name is not cosmetic: _kill_group signals the docker client, and the daemon
    keeps the container running unless something removes it by name.
    """
    argv = docker_argv(sandboxed(), tmp_path)
    assert argv[:4] == ["docker", "run", "--rm", "--init"]
    assert argv[argv.index("--name") + 1] == "veb-test-0000"


def test_no_tty_is_requested(sandboxed, tmp_path):
    """stdout has to stay a clean NDJSON pipe — a tty would wrap the events."""
    argv = docker_argv(sandboxed(), tmp_path)
    assert "-t" not in argv and "--tty" not in argv and "-it" not in argv


def test_network_and_hosts_reach_the_generation_servers(sandboxed, tmp_path):
    argv = docker_argv(sandboxed(), tmp_path)
    assert argv[argv.index("--network") + 1] == "bridge"
    assert argv[argv.index("--add-host") + 1] == "gx10-cbc5:100.64.0.1"


def test_runs_as_the_invoking_user(sandboxed, tmp_path):
    """Otherwise the agent's files come back root-owned in the operator's runs/."""
    import os

    argv = docker_argv(sandboxed(), tmp_path)
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_image_comes_last_and_pi_after_it(sandboxed, tmp_path):
    argv = docker_argv(sandboxed(), tmp_path)
    image = argv.index("veb-pi:latest")
    assert argv[image + 1] == "pi"
    assert "--print" in argv[image + 1 :]


def test_a_host_path_for_pi_is_rejected(pi_config, env_file):
    """It would exec-fail once per seed, with docker's error rather than ours."""
    with pytest.raises(ValueError, match="is a host path"):
        pi_config(
            pi_bin="/home/someone/.local/bin/pi",
            sandbox=SandboxConfig(kind="docker", env_file=env_file),
        )


# ── credentials ───────────────────────────────────────────────────────────────


def test_secrets_arrive_as_a_file_not_on_the_command_line(sandboxed, env_file, tmp_path):
    argv = docker_argv(sandboxed(), tmp_path)
    assert argv[argv.index("--env-file") + 1] == str(env_file.resolve())
    assert SECRET_VALUE not in " ".join(argv)


def test_env_passthrough_forwards_named_variables(sandboxed, env_file, monkeypatch, tmp_path):
    monkeypatch.setenv("WANGP_API_KEY", "from-the-host")
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    config = sandboxed(
        sandbox=SandboxConfig(
            kind="docker",
            env_file=env_file,
            env_passthrough=["WANGP_API_KEY", "NOT_SET_ANYWHERE"],
        )
    )
    env = env_of(docker_argv(config, tmp_path))
    assert env["WANGP_API_KEY"] == "from-the-host"
    assert "NOT_SET_ANYWHERE" not in env, "an unset name must not become an empty one"


# ── what the agent can see ────────────────────────────────────────────────────


def test_the_agent_sees_only_neutral_paths(sandboxed, tmp_path):
    """
    The whole point. Nothing downstream of the image name may mention the host
    filesystem — not the workspace, not the skills, not the destination.
    """
    argv = docker_argv(sandboxed(), tmp_path)
    pi_argv = argv[argv.index("veb-pi:latest") + 1 :]
    assert str(tmp_path) not in " ".join(pi_argv)
    assert str(BENCH_TOOLS_EXTENSION.parent) not in " ".join(pi_argv)


def test_env_pins_the_agent_side_destination(sandboxed, tmp_path):
    env = env_of(docker_argv(sandboxed(), tmp_path))
    assert env["VEB_OUTPUT_PATH"] == "/out/video.mp4"
    assert env["VEB_WORKSPACE"] == "/workspace"
    assert env["VEB_SEED_ID"] == "marketing_001"


def test_only_the_workspace_and_the_tools_are_mounted(sandboxed, tmp_path):
    """
    The run directory is NOT mounted: it holds videos.yaml and every earlier
    seed's video, which is why the output goes through a staging directory.
    """
    argv = docker_argv(sandboxed(), tmp_path)
    mounts = mounts_of(argv)

    assert mounts["/workspace"] == str(tmp_path / "marketing_001" / "workspace")
    assert mounts["/out"] == str(tmp_path / "marketing_001" / "out")
    assert mounts["/opt/veb/ext/bench_tools.ts"] == str(BENCH_TOOLS_EXTENSION)
    assert str(tmp_path) not in " ".join(
        source for destination, source in mounts.items() if destination != "/workspace"
        and destination != "/out"
    )


def test_the_tools_are_mounted_read_only(sandboxed, tmp_path):
    """An agent that can rewrite submit_video can rewrite what counts as delivery."""
    flags = docker_flags(docker_argv(sandboxed(), tmp_path))
    binds = [value for flag, value in zip(flags, flags[1:]) if flag == "-v"]
    for bind in binds:
        writable = not bind.endswith(":ro")
        assert writable is (bind.split(":")[1] in ("/workspace", "/out")), bind


def test_working_directory_is_the_mounted_workspace(sandboxed, tmp_path):
    argv = docker_argv(sandboxed(), tmp_path)
    assert argv[argv.index("-w") + 1] == "/workspace"


def test_task_prompt_names_the_container_workspace(sandboxed, tmp_path):
    config = sandboxed()
    task = PiGenerator(config).render_task(make_seed(), paths_for(config, tmp_path))
    assert "/workspace" in task
    assert str(tmp_path) not in task


# ── skills ────────────────────────────────────────────────────────────────────


def test_skills_are_mounted_under_a_stable_name(sandboxed, tmp_path):
    skill = tmp_path / "bundles" / "video-h3-prompting"
    skill.mkdir(parents=True)
    argv = docker_argv(sandboxed(skills=SkillsConfig(paths=[skill])), tmp_path)

    assert mounts_of(argv)["/opt/veb/skills/video-h3-prompting"] == str(skill)
    assert argv[argv.index("--skill") + 1] == "/opt/veb/skills/video-h3-prompting"


def test_two_skills_with_the_same_basename_do_not_shadow_each_other(sandboxed, tmp_path):
    first = tmp_path / "a" / "video"
    second = tmp_path / "b" / "video"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    argv = docker_argv(sandboxed(skills=SkillsConfig(paths=[first, second])), tmp_path)

    skills = [m for m in mounts_of(argv) if m.startswith("/opt/veb/skills/")]
    assert len(skills) == len(set(skills)) == 2


def test_extra_mounts_from_a_backend_arm_are_appended(sandboxed, env_file, tmp_path):
    config = sandboxed(
        sandbox=SandboxConfig(
            kind="docker",
            env_file=env_file,
            extra_mounts=["/host/assets:/opt/veb/ext/assets:ro"],
        )
    )
    assert mounts_of(docker_argv(config, tmp_path))["/opt/veb/ext/assets"] == "/host/assets"


# ── the unsandboxed path is untouched ─────────────────────────────────────────


def test_without_a_sandbox_every_path_is_the_host_path(pi_config, tmp_path):
    """The regression guard: sandbox=none must behave exactly as it always did."""
    config = pi_config()
    paths = paths_for(config, tmp_path)

    assert not paths.sandboxed
    assert paths.staging_host is None
    assert paths.workspace_agent == tmp_path / "marketing_001" / "workspace"
    assert paths.output_agent == tmp_path / "marketing_001.mp4"
    assert paths.extensions == [(BENCH_TOOLS_EXTENSION, BENCH_TOOLS_EXTENSION)]

    argv = argv_for(config, tmp_path)
    assert argv[0] == config.pi_bin
    assert str(BENCH_TOOLS_EXTENSION) in argv


def test_a_sandboxed_config_needs_somewhere_to_get_its_keys(pi_config):
    with pytest.raises(ValueError, match="needs sandbox.env_file"):
        pi_config(sandbox=SandboxConfig(kind="docker"))


def test_a_missing_env_file_is_caught_at_startup(pi_config, tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        pi_config(sandbox=SandboxConfig(kind="docker", env_file=tmp_path / "absent"))


def test_docker_must_exist_when_sandboxed(sandboxed):
    config = sandboxed(
        sandbox=SandboxConfig(
            kind="docker",
            docker_bin="definitely-not-a-real-binary",
            env_file=Path(".env.example"),
        )
    )
    with pytest.raises(RuntimeError, match="docker binary not found"):
        PiGenerator(config)


# ── the output handoff ────────────────────────────────────────────────────────


def test_a_sandboxed_run_delivers_through_a_staging_directory(sandboxed, tmp_path):
    """
    The container is given an empty /out, not the run directory — which holds
    videos.yaml and every earlier seed's video.
    """
    paths = paths_for(sandboxed(), tmp_path)
    assert paths.staging_host == tmp_path / "marketing_001" / "out"
    assert paths.output_agent == Path("/out/video.mp4")
    assert paths.output_host == tmp_path / "marketing_001.mp4"


def test_the_video_is_moved_to_where_the_rest_of_the_harness_looks(sandboxed, tmp_path):
    from video_eval_bench.generator.pi_generator import _collect_output

    paths = paths_for(sandboxed(), tmp_path)
    paths.staging_host.mkdir(parents=True)
    (paths.staging_host / "video.mp4").write_bytes(b"not really a video")

    _collect_output(paths)

    assert paths.output_host.read_bytes() == b"not really a video"
    assert not paths.staging_host.exists(), "an empty staging directory is litter"


def test_a_run_that_never_submitted_leaves_no_video_behind(sandboxed, tmp_path):
    """
    _resolve_video's "does not exist" check has to keep meaning what it says, so
    nothing may pre-create the destination.
    """
    from video_eval_bench.generator.pi_generator import _collect_output

    paths = paths_for(sandboxed(), tmp_path)
    paths.staging_host.mkdir(parents=True)

    _collect_output(paths)

    assert not paths.output_host.exists()


def test_the_agents_own_leftovers_survive_a_failed_seed(sandboxed, tmp_path):
    """Whatever it put in /out is evidence about why the seed failed."""
    from video_eval_bench.generator.pi_generator import _collect_output

    paths = paths_for(sandboxed(), tmp_path)
    paths.staging_host.mkdir(parents=True)
    (paths.staging_host / "half-written.mp4").write_bytes(b"")

    _collect_output(paths)

    assert (paths.staging_host / "half-written.mp4").exists()


def test_references_need_no_mount_of_their_own(sandboxed, tmp_path):
    """
    A seed's reference images ride the workspace mount that already exists.

    They are staged inside the workspace precisely so this stays true: a mount
    of their own would put a second host directory in the argv, and the agent
    would have a path above /workspace to walk into again.
    """
    from video_eval_bench.dataset.seed import SeedReference
    from video_eval_bench.generator.pi_generator import REFERENCE_DIR
    from test_pi_generator import make_seed as make_bare_seed

    image = tmp_path / "assets" / "hero.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not really a png")

    seed = make_bare_seed()
    seed.references = [
        SeedReference(
            id="hero",
            role="character",
            label="Hero",
            description="The protagonist.",
            path=image,
        )
    ]

    config = sandboxed()
    paths = paths_for(config, tmp_path)
    generator = PiGenerator(config)
    argv = generator.build_argv(seed, paths)
    with_refs = generator._docker_argv(argv, seed, paths, "veb-test-0000")

    # The brief names the reference, and names it relatively.
    assert f"{REFERENCE_DIR}/hero.png" in argv[-1]
    assert str(tmp_path) not in argv[-1]
    # The mounts are exactly the ones a seed carrying nothing gets, and the
    # image's host path appears nowhere on the command line.
    assert mounts_of(with_refs) == mounts_of(docker_argv(config, tmp_path))
    assert not any(str(image) in a for a in with_refs)
