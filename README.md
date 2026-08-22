# video-story-bench : Benchmark for video short story quality.

Tiktok and Youtube have recently both invested on the short video format.
Generating content appreciated by many is a skill master by few and where AI model are particularly lacking.

This benchmark testes how well a brief is understood by AI agent and can be translated into a high quality short video story.
This benchmark requires understanding of the medium, ability to define character and orchestrate them, but also mastering video generation and editing tools. 

We believe this benchmark is a first step for a more generic film director skill evaluation.

Last but not least, this benchmark evaluates the Safety of these video agent. A few cases tests if the agent does take shortcuts and generate harmful content.

## Overview

A dataset of **seeds** across **categories**. For each seed an **agent** generates a
video, and an **LLM judge** grades it against the **category-specific rubric**.

The generating agent is the thing under test. It is assembled from four
independently swappable parts — **model**, **system prompt**, **skills**,
**tools** — so the question the benchmark answers is not just "what does the agent
score" but "which of those four is holding it back".

## Layout

```
video_eval_bench/
├── conf/               # Hydra config tree — one group per ablation axis
├── config.py           # typed schema (pydantic) + builders
├── schemas.py          # JudgeVerdict, JudgeScore, SafetyResult
├── dataset/            # Seed/Category/Rubric schemas + YAML loaders
├── generator/
│   ├── base.py         # GenerateFn: async (seed, output_dir) -> video_path
│   ├── mock_generator.py   # synthetic videos, for offline runs
│   ├── pi_generator.py     # the agentic generator (one pi run per seed)
│   ├── pi_ext/bench_tools.ts   # the submit_video handoff tool
│   └── prompts/        # packaged system prompts + task template
├── judge/
│   ├── frames.py       # video → N evenly-spaced JPEG frames (cv2)
│   ├── prompt.py       # renders one rubric criterion into a judge prompt
│   ├── llm.py          # VisionLLM backends: LiteLlmBackend / MockBackend
│   ├── pi_backend.py   # judge via the pi CLI (local vision model)
│   └── agent.py        # VideoJudge — frames + prompt + LLM → JudgeVerdict
├── bench.py            # run_bench(): generate + judge every seed
├── run.py              # `veb` — the entry point
└── compare.py          # `veb-compare` — comparative table over reports
dataset/                # seeds.yaml + rubric_{a,b,c,d}.yaml
tests/                  # offline suite; tests/e2e/ is the opt-in live tier
```

## How it works

1. **Dataset** — `seeds.yaml` lists generation briefs, each tagged with a category.
   `rubric_a/b.yaml` are universal; `rubric_c.yaml` holds one rubric per genre;
   `rubric_d.yaml` holds the safety veto checks.
2. **Generate** — `PiGenerator` runs the `pi` agent once per seed in an isolated
   workspace, with the configured system prompt, tool allowlist, skills and
   custom-tool extensions. The agent produces a video and calls `submit_video`.
3. **Judge** — `VideoJudge` samples N frames, asks the vision LLM one focused
   question per rubric criterion, and **recomputes the weighted total itself**
   (it never trusts the model's arithmetic). A broken judge never aborts a run.
4. **Report** — per-seed verdicts, per-category aggregates, and the resolved
   config, written to `runs/<run_id>/report.json`.

## Running

```bash
veb                                  # defaults: pi agent on gx10, pi judge
veb experiment=mock                  # fully offline: synthetic video, fake verdicts
veb experiment=smoke                 # one real seed, baseline agent (no skills/tools)
veb run.category=marketing run.max_seeds=1
```

`experiment=smoke` is the **baseline arm**, not a health check: the agent gets the
default prompt, bash, and nothing else, so whether it can produce a video at all is
part of what is being measured — a refusal is a legitimate result. Nothing in this
repo wires the agent to a video backend; supplying one is what the `skills` and
`extensions` config does. `tests/e2e/` contains a working example of both.

An agentic run against a local model takes minutes per seed, so `run.max_seeds`
and `run.seed_ids` exist to keep ablations cheap. Only a settled configuration
earns a full 8-seed pass.

### Varying the agent

Each config group is one axis. Swap one, keep the rest:

```bash
veb skills=e2e_mock                  # give the agent a skill
veb system_prompt=minimal            # strip the prompt back
veb tools=no_bash                    # take the shell away
veb model=amor_ms_qwen27b_q3         # smaller quant, shorter context
```

Run several in one command; each is a separate run with its own report:

```bash
veb -m skills=none,e2e_mock tools=full,no_bash
```

**Adding an arm is a new file in a group directory** — no Python change.
`conf/skills/e2e_mock.yaml` is the worked example: point `paths` at a directory
containing a `SKILL.md`, and `skills=<name>` becomes a new arm.

Available groups: `generator`, `model`, `system_prompt`, `skills`, `tools`,
`judge`, `experiment`.

### Comparing runs

Comparison is a separate step over report files, so runs made days apart compare
the same way as a sweep:

```bash
veb-compare runs/*/report.json
veb-compare runs/sweep_*/*/report.json --metric pass_rate --sort --csv out.csv
```

Only the axes that actually differ become columns:

```
skills     tools    score  marketing  pass  veto  err  runtime
---------  -------  -----  ---------  ----  ----  ---  -------
e2e_mock   full     58.7   58.7       3/4   0     0    4m12s
none       full     41.2   41.2       1/4   0     1    3m48s
none       no_bash  33.4   33.4       0/4   0     3    1m02s
```

## What a run leaves behind

The Hydra output directory *is* the run directory, so a run is reproducible from
its own folder:

```
runs/20260822-193000/
├── .hydra/config.yaml       # the exact composed config
├── .hydra/overrides.yaml    # the overrides that produced it
├── report.json              # verdicts, aggregates, variant, resolved config
├── marketing_001/
│   ├── workspace/           # the agent's working directory, kept for debugging
│   ├── transcript.jsonl     # the full pi event stream
│   ├── pi_stderr.log
│   └── pi_run.json          # argv, turns, tool counts, tokens, outcome
└── marketing_001.mp4        # the submitted video — what the judge sees
```

## Writing a generator

The contract is one async callable:

```python
async def generate(seed: Seed, output_dir: Path) -> str:
    ...  # produce a video for seed.prompt
    return "/path/to/video.mp4"
```

`PiGenerator` is the agentic implementation. Two details of it are load-bearing:

- **The handoff is a tool, not a filename.** The agent calls `submit_video`; a file
  left in the workspace is not a result. The destination is pinned into the
  environment, never passed as a tool argument, so no prompt can redirect it and
  the agent never has to remember it across context compaction.
- **There is no inactivity timeout.** pi puts no default timeout on its bash tool,
  so a real generation can block silently for many minutes. The only budget is
  `timeout_seconds`; progress is surfaced by heartbeat logging instead.

Two things that bite when configuring an agent, both enforced by the config layer:

- pi only injects the skills block when the **`read` tool is enabled** — `--skill`
  is otherwise silently inert, and a `skills=` ablation arm would measure nothing.
  Configuring skills without `read` is a startup error.
- pi's `--tools` allowlist covers **extension tools too**, so `submit_video` is
  always appended to it. Without that the agent generates a fine video and then
  finds it has no way to hand it in.

## Tests

```bash
python -m pytest tests/ -q            # offline: no pi, no model, no network
python -m pytest tests/e2e -m e2e -q  # live: real pi + real model
```

The e2e tier runs the **real** agent and the **real** judge against a live model —
no LLM call is mocked. What it fakes is the video backend
(`tests/e2e/mock_video_service.py`), which answers in milliseconds instead of
minutes, so the process lifecycle can be exercised under short timeouts. It skips
itself when `pi` or the model endpoint is unavailable.
