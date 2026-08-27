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
│   ├── base.py         # GenerateFn: async (seed, output_dir) -> GenerationResult
│   ├── manifest.py     # videos.yaml: the format runs write and imports read
│   ├── external_generator.py  # score videos made elsewhere / replay a run
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
├── report/
│   ├── base.py         # BenchReport / SeedResult + aggregation
│   ├── html.py         # the per-run HTML report
│   └── cli.py          # `veb-report` — re-render an existing run
├── bench.py            # run_bench(): generate + judge every seed
├── run.py              # `veb` — the entry point
└── compare.py          # `veb-compare` — comparative table over reports
dataset/                # seeds.yaml + rubric_{a,b,c,d}.yaml + references/
tests/                  # offline suite; tests/e2e/ is the opt-in live tier
```

## How it works

1. **Dataset** — `seeds.yaml` lists generation briefs, each tagged with a category
   and optionally carrying [reference images](#references). `rubric_a/b.yaml` are
   universal; `rubric_c.yaml` holds one rubric per genre; `rubric_d.yaml` holds
   the safety veto checks.
2. **Generate** — `PiGenerator` runs the `pi` agent once per seed in an isolated
   workspace, with the configured system prompt, tool allowlist, skills and
   custom-tool extensions, plus the seed's references staged into the workspace.
   The agent produces a video and calls `submit_video`.
3. **Judge** — `VideoJudge` samples N frames, shows them to the vision LLM behind
   the seed's reference images, asks one focused question per rubric criterion,
   and **recomputes the weighted total itself** (it never trusts the model's
   arithmetic). A broken judge never aborts a run.
4. **Report** — per-seed verdicts, per-category aggregates, and the resolved
   config, written to `runs/<run_id>/report.json`.

## Running

Keys and endpoints come from `.env`, which is read both by `veb` itself and by
the agent's sandbox. Copy the template and fill in what the arms you run need:

```bash
cp .env.example .env
```

```bash
veb                                  # defaults: pi agent on gx10, pi judge
veb experiment=mock                  # fully offline: synthetic video, fake verdicts
veb experiment=smoke                 # one real seed, stand-in generator, ~10 min
veb run.category=marketing run.max_seeds=1
```

`experiment=smoke` runs the real agent and the real judge against one seed with the
**stand-in** generation tool, so it finishes in minutes. It answers "can the agent
drive the tool and hand back a result", not "is the video good". Swap in
`video_backend=wangp` for the real generator, and budget accordingly — a single
generation takes minutes.

An agentic run against a local model takes minutes per seed, so `run.max_seeds`
and `run.seed_ids` exist to keep ablations cheap. Only a settled configuration
earns a full 8-seed pass.

### Varying the agent

Each config group is one axis. Swap one, keep the rest:

```bash
veb skills=h3_prompting              # teach it how to prompt the video model
veb system_prompt=minimal            # strip the prompt back
veb tools=no_bash                    # take the shell away
veb model=amor_ms_qwen27b_q3         # smaller quant, shorter context
veb model=openai_gpt56_luna          # hosted frontier arm (needs OPENAI_API_KEY)
veb model=anthropic_sonnet5          # hosted frontier arm (needs a pi anthropic login)
veb video_backend=wangp              # give it a real video generator
veb sandbox=docker                   # run the agent in a container (see below)
```

To judge videos you generated some other way, or to re-judge a previous run
without regenerating it, see [Judging videos made elsewhere](#judging-videos-made-elsewhere).

### The video backend

`video_backend` decides what the agent has to generate video with:

| Arm | What the agent gets |
|---|---|
| `none` | nothing — bash and whatever is on the box. The baseline. |
| `wangp` | a blocking `generate_video` tool driving MiniMax H3 on the WanGP server: `t2va` (text only) and `ref2va` (reference-conditioned) |
| `fake` | the same tool, same parameters, returning a fixed video instantly |

`fake` exists to separate two questions that cost very different amounts. Whether
the agent *drives the tool correctly* — right mode, a full brief in the prompt,
`submit_video` afterwards — is answerable in seconds. Only whether the video is
any *good* needs the GPU. Running the cheap question against the real backend
wastes minutes per seed and makes prompt iteration impractical.

The real tool blocks through submit → poll → download inside one call, on
purpose: polling from the agent loop costs a model call per check and pushes a
long run into auto-compaction (see §4b in the generator docstring).

Run several in one command; each is a separate run with its own report:

```bash
veb -m skills=none,e2e_mock tools=full,no_bash
```

**Adding an arm is a new file in a group directory** — no Python change.
`conf/skills/h3_prompting.yaml` is the worked example: point `paths` at a
directory containing a `SKILL.md`, and `skills=<name>` becomes a new arm. That
one points at `skills/video-h3-prompting/`, which teaches the production-brief
format MiniMax H3 actually wants — the contrast arm for `video_backend=wangp`,
since an agent that writes a one-line prompt gets a visibly worse video.

Available groups: `generator`, `model`, `system_prompt`, `skills`, `tools`,
`video_backend`, `judge`, `sandbox`, `experiment`.

### References

A seed can hand the agent images to hold to — a character's face, a location, a
product, a look:

```yaml
# dataset/seeds.yaml
  - seed_id: entertainment_001
    category: entertainment
    prompt: >
      A 5 seconds short story in 3 scenes: ...
    references:
      - id: maya
        role: character          # character | location | style | prop
        label: "Maya"
        description: >
          The woman the story follows. Her face, her dark curly hair and her red
          wool coat must be the same in all three scenes.
        path: references/entertainment_001/maya.png
```

`path` is relative to `dataset/`; the files live in
`dataset/references/<seed_id>/` so the dataset stays self-contained. **The images
committed today are placeholder illustrations** — see
`dataset/references/README.md` before quoting a number from them.

The order of the list is meaningful: it is the order the agent is shown them in,
and MiniMax H3 labels its references `<Picture 1>`, `<Picture 2>`, … by their
position in `image_refs`. `role` is what tells identity work from style work,
a distinction `skills/video-h3-prompting` leans on hard.

**What the agent gets.** Each image is copied into `<workspace>/references/<id>`
before `pi` starts, and the brief names them by that relative path with their
role, label and description. Nothing new is mounted under `sandbox=docker`: the
workspace is already the container's `/workspace`, so the references arrive
inside it and `generate_video`'s "must be in the working directory" check passes
unchanged. Whether the agent then *uses* `mode: "ref2va"` or just describes the
images in words is exactly the thing being measured.

**What the judge gets.** The same images, ahead of the clip's frames, described
in the prompt header — and one more criterion, `S5 — Reference Adherence`. A seed
carrying no references passes S5 without a model call: skipping it instead would
quietly cost that seed 3 of Section B's 13 points, since a section scores as a
percentage of its own weight. `requires_references: true` in the rubric YAML is
what marks a criterion that way.

References reach the judge from the dataset, not from the generator, so an
imported video (`generator=external`) is scored on S5 too.

### Sandboxing the agent

`sandbox=docker` runs the agent's `pi` in a throwaway container per seed, holding
that seed's workspace and nothing else.

```bash
docker/build.sh                      # once
veb sandbox=docker video_backend=wangp
```

This is a correctness setting, not a deployment one. On the host the agent's
workspace sits four levels under the project root and `tools=full` puts no path
restriction on `read` or `bash` — and a real run used that to walk up and read
`dataset/rubric_c.yaml`, working out from its own reasoning that it was under
evaluation, before being scored against that rubric. A number from such a run
measures rubric-reading. See `runs/FINDINGS.md` §4.

What the agent gets:

| | |
|---|---|
| `/workspace` | its own workspace, read-write, the only writable place |
| `/out` | an empty directory it delivers into; the harness moves the video out |
| `/opt/veb/ext`, `/opt/veb/skills` | its tools and skills, read-only |
| network | `bridge`, plus `--add-host` for the generation servers |

The repo, `dataset/`, the run directory and every other seed are not mounted, so
there is nothing above `/workspace` to walk up into. Paths are neutral, so the
command line does not describe the host either.

Two things about the image are part of the measurement rather than packaging.
**The pi version is pinned** — `build_argv` is written against one CLI contract.
And **the installed package list is the benchmark floor**: `video_backend=none`
means "bash and whatever is on the box", so `ffmpeg`, `curl` and `python3` being
in `docker/Dockerfile.pi` is a statement about what the baseline arm can reach.
Adding a tool there changes what every `video_backend=none` run measures.

Credentials reach the container as `--env-file .env`, which means every key in
that file, not only the ones the arm needs — the same exposure the agent has on
the host, where it inherits the whole environment. `sandbox.env_passthrough`
forwards named variables instead, if you want it tighter.

The judge is not sandboxed. It runs `--no-tools`, so it has nothing to jail.

### Reading a run

Every run writes `report.html` next to its videos — one page with the summary,
the exact config, and per seed: the brief, the video inline, what the agent
actually did, and the judge's verdict on every criterion. Seeds are collapsible,
so an eight-seed run opens as a short list you expand where it matters.

```bash
xdg-open runs/20260822-193000/report.html
veb-report runs/*/            # re-render (e.g. after the renderer changes)
```

It references the videos rather than embedding them, so the page stays in the
low hundreds of KB and opens instantly — but it belongs to its run directory.
Move the folder, not the file.

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
├── videos.yaml              # the videos this run produced — replay it (below)
├── report.html              # the same, readable, with videos and traces inline
├── marketing_001/
│   ├── workspace/           # the agent's working directory, kept for debugging
│   ├── transcript.jsonl     # the full pi event stream
│   ├── pi_stderr.log
│   └── pi_run.json          # argv, turns, tool counts, tokens, outcome
└── marketing_001.mp4        # the submitted video — what the judge sees
```

## Judging videos made elsewhere

Videos generated outside the harness are scored by pointing the `external` arm at
a manifest. Nothing is generated; each seed's video is looked up, checked, and
placed in the run directory, so it flows through the same judge, report and
`veb-compare` as an agentic run and is directly comparable with one.

```yaml
# videos.yaml — paths resolve against this file's directory
label: wangp-minimax-h3-turbo
videos:
  - seed_id: marketing_001
    path: out/marketing_001.mp4
    prompt: "..."               # optional: the prompt actually used, if it differs
    duration_seconds: 1320      # optional: what generation really cost, elsewhere
    source: "MiniMax-H3 turbo 832x480"
```

```bash
veb experiment=external generator.manifest=videos.yaml
```

**The dataset still drives the run.** A manifest covering three of eight seeds
scores three and marks five **skipped** — distinct from an error, counted
separately, and shown on the report's front page and in `veb-compare`'s `skip`
column. The benchmark does not quietly shrink to fit what you supplied.

`label` names the batch in `veb-compare`, which otherwise sees only
`generator=external` and cannot tell two imports apart.

### Replaying the judge

Every run writes its videos as `videos.yaml` in exactly that format — so a run is
its own manifest, and can be pushed back through a different judge without
repeating the generation:

```bash
veb experiment=external generator.manifest=runs/20260822-193000/videos.yaml \
    generator.copy_videos=false judge=litellm judge.model=openai/gpt-4o

veb-compare runs/20260822-193000/report.json runs/<replay>/report.json
```

That is the expensive half saved: an agentic seed takes minutes to hours, and
judging it again should not cost that again. `copy_videos=false` symlinks instead
of copying, which suits a run that is staying on disk; the default copies so the
new run directory stays self-contained.

The manifest is rewritten after **every seed**, not at the end, because the runs
worth replaying are the ones that do not finish — a run killed at seed five has
already paid for four videos.

## Writing a generator

The contract is one async callable:

```python
async def generate(seed: Seed, output_dir: Path) -> GenerationResult:
    ...  # produce a video for seed.prompt
    return GenerationResult(seed_id=seed.seed_id, video_path="/path/to/video.mp4")
```

It returns a result rather than a path because the answer is rarely just a
filename. A generator may also report:

- `metadata` — anything worth recording about how the video was made. It arrives
  bound to the video it describes, and lands in the report as-is.
- `duration_seconds` — what generation actually cost, when timing this process
  would measure nothing (an imported video was generated elsewhere, days ago).
- `status="skipped"` — *there is no video for this seed*. Only an import says
  this. It is deliberately not an error: a batch covering three of eight seeds
  has not failed five times, and the report counts the two apart.

Failure is a raised `GenerationError`, which carries `metadata` too — a run that
burned an hour before timing out is the most expensive seed in the run, and its
turn count is the only clue to why it failed.

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
