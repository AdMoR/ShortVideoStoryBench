"""
Benchmark entry point.

    veb                                     # defaults
    veb skills=video_basic tools=no_bash    # one variant
    veb experiment=mock                     # fully offline
    veb -m skills=none,video_basic          # two runs, two reports

Every run also writes `videos.yaml`, a manifest of the videos it produced, in the
format the `external` generator reads. So a run can be re-judged without being
regenerated:

    veb experiment=external generator.manifest=runs/<id>/videos.yaml judge=litellm

One invocation is one run of one agent variant, producing one report. Comparing
variants is a separate step over those reports:

    veb-compare runs/*/report.json

The Hydra output directory doubles as the run directory, so a run's report, its
per-seed artifacts and the exact config that produced it all live together.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from video_eval_bench.bench import run_bench
from video_eval_bench.config import (
    DEFAULT_PROMPT_DIR,
    PI_EXT_DIR,
    BenchConfig,
    build_generator,
    build_judge,
    redact,
)
from video_eval_bench.dataset import load_dataset
from video_eval_bench.report.html import render_run

logger = logging.getLogger(__name__)

# Lets a config say `${veb_prompt:director_system.md}` for a prompt shipped with
# the package, instead of hard-coding an install path into YAML.
OmegaConf.register_new_resolver(
    "veb_prompt", lambda name: str(DEFAULT_PROMPT_DIR / name), replace=True
)
# Same idea for the custom-tool extensions shipped with the package.
OmegaConf.register_new_resolver("veb_ext", lambda name: str(PI_EXT_DIR / name), replace=True)


def variant_of(choices) -> str:
    """
    A readable id for this configuration, derived from the chosen group options.

    Derived rather than hand-written so it cannot drift from the config it names.
    `veb-compare` uses the same choices to label and align reports.
    """
    axes = ["model", "system_prompt", "skills", "tools", "generator", "judge"]
    parts = [f"{axis}={choices[axis]}" for axis in axes if choices.get(axis) not in (None, "null")]
    return ",".join(parts)


def selected_choices(choices) -> dict:
    """The group choices worth recording — Hydra's own bookkeeping groups dropped."""
    return {
        key: str(value)
        for key, value in choices.items()
        if not key.startswith("hydra/") and value not in (None, "null")
    }


def _generator_label(generate) -> str:
    """What the generator calls itself, if it has a name of its own."""
    return str(getattr(generate, "label", "") or "")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    choices = selected_choices(hydra_cfg.runtime.choices)
    variant = variant_of(hydra_cfg.runtime.choices)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.pop("paths", None)
    try:
        bench = BenchConfig(**resolved)
    except Exception as exc:
        # A bad ablation arm should say so now, not after an hour of generation.
        logger.error(f"invalid configuration: {exc}")
        raise SystemExit(2) from exc

    logger.info(f"[run] variant: {variant or '(defaults)'}")
    logger.info(f"[run] output:  {run_dir}")

    dataset = load_dataset(bench.run.dataset_dir)
    generate = build_generator(bench.generator)
    judge = build_judge(bench.judge, dataset)

    started = datetime.now(timezone.utc)
    report = asyncio.run(
        run_bench(
            judge=judge,
            generate=generate,
            output_dir=run_dir,
            run_id=run_dir.name,
            dataset=dataset,
            category=bench.run.category,
            seed_ids=bench.run.seed_ids,
            max_seeds=bench.run.max_seeds,
            # The run writes its videos as a manifest so it can be replayed
            # through another judge later; label it with what produced them.
            manifest_label=run_dir.name,
            manifest_source=variant,
        )
    )

    report.variant = variant
    report.choices = choices
    # An import is `generator=external` whatever produced its videos, so without
    # this two imports from two different systems collapse into one
    # indistinguishable row in veb-compare, which diffs the choices.
    source = _generator_label(generate)
    if source:
        report.choices = {**choices, "source": source}
    report.config = redact(resolved)
    report.note = bench.run.note
    report.started_at = started.isoformat()
    report.finished_at = datetime.now(timezone.utc).isoformat()

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report.to_json(), indent=2))

    print(json.dumps(report.summary(), indent=2))
    print(f"\nReport: {report_path}")

    # The HTML is a convenience on top of report.json, so a failure to render it
    # must never cost the run its results.
    try:
        print(f"HTML:   {render_run(run_dir, dataset)}")
    except Exception as exc:
        logger.warning(f"could not render the HTML report: {exc}", exc_info=True)


if __name__ == "__main__":
    sys.exit(main())
