# video-story-bench : Benchmark for video short story quality.

Tiktok and Youtube have recently both invested on the short video format.
Generating content appreciated by many is a skill master by few and where AI model are particularly lacking.

This benchmark testes how well a brief is understood by AI agent and can be translated into a high quality short video story.
This benchmark requires understanding of the medium, ability to define character and orchestrate them, but also mastering video generation and editing tools. 

We believe this benchmark is a first step for a more generic film director skill evaluation.

Last but not least, this benchmark evaluates the Safety of these video agent. A few cases tests if the agent does take shortcuts and generate harmful content.

## Overview

A dataset of **seeds** across **categories**. For each seed an **agent** generates a
video, and an **LLM judge** grades it against **the criteria that seed names** —
drawn from one shared rubric library, so no seed is scored on a question its
brief gives it no way to fail.

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
├── dataset/            # Seed/RubricLibrary schemas + YAML loaders
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
│   ├── llm.py          # VisionLLM backends: LiteLlm / OpenAI (video) / Mock
│   ├── pi_backend.py   # judge via the pi CLI (local vision model)
│   └── agent.py        # VideoJudge — clip + prompt + LLM → JudgeVerdict
├── report/
│   ├── base.py         # BenchReport / SeedResult + aggregation
│   ├── html.py         # the per-run HTML report
│   ├── atlas.py        # the rubric library as one filterable page
│   ├── site.py         # the three-page GitHub Pages site + the run snapshot
│   └── cli.py          # `veb-report` — re-render an existing run
├── seedbuilder/        # FineVideo metadata -> seeds + rubrics (`veb-seedbuild`)
│   ├── digest.py       # one metadata JSON -> a bounded, deterministic digest
│   ├── stages.py       # synthesize the brief, select the rubric, run both judges
│   ├── probe.py        # the ffprobe validator, for `container` criteria
│   ├── policy.py       # which generated criteria the bench grades — applied last
│   └── report.py       # `veb-seedreport` — the build report
├── bench.py            # run_bench(): generate + judge every seed
├── run.py              # `veb` — the entry point
└── compare.py          # `veb-compare` — comparative table over reports
dataset/                # seeds.yaml + rubrics.yaml + genres.yaml + safety.yaml + references/
                        # RUBRICS.md — the standard a criterion is held to
site/data/runs.json     # the published run snapshot — the site's only committed data
.github/workflows/      # pages.yml — builds and deploys the site
docker/                 # the agent jail: Dockerfile.pi + build.sh + entrypoint
tests/                  # offline suite; tests/e2e/ is the opt-in live tier
Makefile                # build the jail, run the benchmark — `make help`
```

## How it works

1. **Dataset** — `seeds.yaml` lists generation briefs, each tagged with a genre,
   naming [the criteria it is judged on](#rubrics), and optionally carrying
   [reference images](#references). `rubrics.yaml` is the criterion library every
   seed draws from; `genres.yaml` is the genre vocabulary; `safety.yaml` holds
   the veto checks, which apply to every seed.
2. **Generate** — `PiGenerator` runs the `pi` agent once per seed in an isolated
   workspace, with the configured system prompt, tool allowlist, skills and
   custom-tool extensions, plus the seed's references staged into the workspace.
   The agent produces a video and calls `submit_video`.
3. **Judge** — `VideoJudge` shows the vision LLM the clip (as N sampled frames, or
   [as the whole video](#judging-the-whole-clip)) behind the seed's reference
   images, asks one focused question per criterion the seed lists, and
   **recomputes the weighted total itself** (it never trusts the model's
   arithmetic). A broken judge never aborts a run.
4. **Report** — per-seed verdicts, per-category aggregates, and the resolved
   config, written to `runs/<run_id>/report.json`.

## Running

```bash
make install                         # sync the venv from uv.lock
make run                             # build the agent jail, then evaluate
```

`make run` is the whole path: it builds the `veb-pi` image when `docker/` has
changed since the last build, writes `.env` from `.env.example` if it is
missing, and evaluates the agent **at full strength** — the real `generate_video`
tool and every skill in `skills/`, sandboxed. Keys and endpoints come from that
`.env` — read both by `veb` itself and by the agent's container — so fill in what
the arms you run need before the first real run.

That is `video_backend=wangp skills=all`, which is *not* the config default:
bare `veb` composes the unaided baseline (`video_backend=none skills=none`),
because a floor is what an ablation measures from. The Makefile runs the ceiling
and leaves the floor one variable away:

```bash
make run BACKEND=none SKILLS=none    # the baseline arm
make run ARGS="-m skills=none,all"   # both, one report each
```

A real generation takes minutes per seed and the default selection is all eight,
so an unqualified `make run` is an hours-long job. While iterating, cap the seeds
or take the GPU out of the loop:

```bash
make run ARGS="run.max_seeds=1"      # one seed instead of eight
make run BACKEND=fake                # stand-in generator, answers instantly
```

```bash
make mock                            # fully offline: synthetic video, fake verdicts
make smoke                           # one real seed, stand-in generator, ~10 min
make build                           # rebuild the agent jail image
make report                          # re-render the latest run's HTML
make compare                         # comparative table over every run
make test                            # offline test suite
make help                            # all of the above, with variables
```

Any other Hydra override goes through `ARGS` the same way:

```bash
make run ARGS="run.category=marketing"
```

`make run` also passes `sandbox=docker`, a third deliberate departure from the
config default — see [Sandboxing the agent](#sandboxing-the-agent) for why, and
use `make run SANDBOX=none` to put the agent back on the host. The rest of this
README is written as the underlying `veb …` command: run those in the synced
venv (`uv run veb …`), or hand them to `make run ARGS="…"`.

`make smoke` (`experiment=smoke`) runs the real agent and the real judge against
one seed with the **stand-in** generation tool, so it finishes in minutes. It
answers "can the agent drive the tool and hand back a result", not "is the video
good". It also uses
`system_prompt=tooled`, which names `generate_video` and `submit_video` outright
rather than leaving the agent to discover them — discovery is what the default
`director` arm measures, and it is not what a smoke run is for. Swap in
`video_backend=wangp` for the real generator, and budget accordingly — a single
generation takes minutes.

`run.max_seeds` and `run.seed_ids` are what keep ablations cheap — a settled
configuration is the only one that earns a full 8-seed pass.

### Varying the agent

Each config group is one axis. Swap one, keep the rest:

```bash
veb skills=all                       # every skill in skills/ — what `make run` uses
veb skills=h3_prompting              # one bundle: how to prompt the video model
veb system_prompt=minimal            # strip the prompt back
veb system_prompt=tooled             # name generate_video/submit_video outright
veb tools=no_bash                    # take the shell away
veb model=amor_ms_qwen27b_q3         # smaller quant, shorter context
veb model=openai_gpt56_luna          # hosted frontier arm (needs OPENAI_API_KEY)
veb model=anthropic_sonnet5          # hosted frontier arm (needs a pi anthropic login)
veb video_backend=wangp              # give it a real video generator
veb sandbox=docker                   # run the agent in a container (see below)
```

Through the Makefile that is `make run ARGS="skills=h3_prompting"`, and a sweep
is `make run ARGS="-m skills=none,h3_prompting"`.

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

`skills=all` is the full listing of `skills/`, and the arm `make run` uses. It
lists its bundles explicitly rather than turning on pi's own discovery, which
reads the operator's skill roots — those are not in the repo, so a discovered
run would depend on whose box it ran on. **A new bundle in `skills/` has to be
added to `conf/skills/all.yaml` too**, or the default run will not see it.

Available groups: `generator`, `model`, `system_prompt`, `skills`, `tools`,
`video_backend`, `judge`, `sandbox`, `experiment`.

### Rubrics

There is one **rubric library** — `dataset/rubrics.yaml` — holding every criterion
the benchmark knows how to check. Nothing in it is applied automatically. Each
seed names the ids that apply to it, and the judge asks about exactly those:

```yaml
# dataset/rubrics.yaml — the library
criteria:
  - id: ENV1
    dimension: consistency
    name: Environment Consistency
    description: >
      Wherever two shots depict the same place, it is the same place. ...
    weight: 3
    critical: true

# dataset/seeds.yaml — what this seed is judged on, and only this
  - seed_id: social_media_001
    category: social_media
    rubrics: [SUBJ1, ENV1, TEMP1, ART1, TEXT1, AR1, PROD1, COPY1, TRIG1,
              HOOK1, SETUP1, TENS1, REVEAL1, CTA1, CUT1, FOCUS1]
```

`make atlas` renders the whole library as one page — every criterion with its gate,
evidence class, weight and tags, filterable by tag, plus what a built corpus actually
reached for (`video_eval_bench/report/atlas.py`). Re-render it when the library
changes; a stale atlas is worse than none, since its whole claim is to be the library.

**How a criterion is written** is its own standard, and it is documented in
[`dataset/RUBRICS.md`](dataset/RUBRICS.md): when a criterion must state the kind of
brief it applies to, when two criteria are the same question twice, why the ones a
reviewer calls vague are exactly the ones that end up binding a value from the seed,
and when a check belongs on one seed instead of in the library. Read it before adding
or merging one.

**Why per-seed.** A fixed universal section asked every video about geographical
accuracy, historical period and cultural symbolism. For a kitchen-cleaning ad
those are unanswerable, the judge was told to pass what it thought inapplicable,
and the seed banked the weight — three criteria of free score, concentrated on
whichever seeds the fixed rubric fit worst. A criterion a brief cannot fail now
simply is not on its list, so it costs no model call and contributes no weight.

**Scoring is flat.** `total_score` is the weight earned over the weight the seed
was asked for. Adding a criterion to a seed makes that seed harder; it never
rescales what was already there. `dimension` groups criteria for the report
(`consistency`, `technical`, `fidelity`, `structure`, `craft`) and is a view of
the score, not an input to it — the old fixed sections gave a three-criterion
baseline the same say in the total as an eight-criterion genre rubric.

**`critical: true`** marks the criteria a video must not fail. They are reported
as such (called out at the top of the seed's judge block, flagged ⚠️ in the
table) and are not otherwise weighted differently.

`genres.yaml` still tags each seed, but only as a reporting label: it groups the
summary table and `veb-compare` columns and names the kind of brief to the judge.
It selects no rubric. Genres and criteria are both validated at load: a typo in
either fails the whole load, before any generation is paid for.

Adding a criterion is a new entry in `rubrics.yaml` plus the seeds that should
carry it — no Python change. Some criteria ship listed by no seed: everything in
the `audio` section needs a judge that is given sound, and none is yet.

The library is organised into sections, generic first — `general`, `people`,
`continuity`, `light`, `attention`, `references`, `audio`, `narrative`,
`transformation`, `instructional`, `promotional`, `interactive`, `world`. Sections
group criteria for reading and reporting and select nothing; the later ones hold the
criteria that only mean something for a kind of brief and say so in their text.

`criterion_tags` is the analytic axis that cuts across them, in four groups:
`subject` (what the check looks at), `sub_theme` (a niche inside a subject — `brand`,
`diagram`, `interface` — kept separate so a report does not list them as peers of
`people`),
`span` (how much of the clip it needs at once) and `failure` (what kind of defect it
catches). The vocabulary is closed: a tag outside it fails the load.

**`evidence`** says what a validator must be able to *see* to settle a criterion —
`description`, `pixels`, `motion`, `audio` or `container`. The benchmark judge is
shown the clip and so can answer any of them; the field exists for
[the seed builder](#building-seeds-from-finevideo), which validates generated
criteria against source material it sometimes only has a written description of.
Asking a text judge whether a video has generation artifacts gets a confident answer
that means nothing, and `evidence` is what stops the question being asked.

**`binds`** lets one shared criterion ask about a particular seed's subject. A
criterion whose description contains `{subject}` declares `binds: [subject]`, and a
seed supplies the value:

```yaml
    rubrics:
      - id: SUBJ1
        bind: {subject: "the red kettle"}
      - CUT1                     # a bare id is shorthand when there is nothing to bind
```

The id, the weight and the dimension stay shared, so two seeds carrying `SUBJ1` are
still the same column in every report — the question is just asked in their own
terms. Binding is validated at load: a bind the criterion does not declare, or a
`{placeholder}` a seed leaves unfilled, fails the whole load rather than reaching the
judge as a literal brace.

### Judging the whole clip

By default the judge sees `n_frames` evenly-spaced stills. That is a workaround:
motion, timing and cut rhythm have to be inferred from the gaps between frames,
which is exactly what `MOTION1`, `CUT1`, `TEMP1` and `ANGLE1` are asking about.

`judge=video` sends the clip itself instead:

```bash
veb judge=video judge.api_base=http://your-server:8080/v1 judge.model=<model>
```

It speaks plain OpenAI chat-completions over HTTP, carrying the clip as a
content part alongside the reference images:

```json
{"type": "input_video", "input_video": {"data": "<base64 of the file>"}}
```

That part is an agreed extension to the OpenAI schema, not part of it, and it
accepts base64 only. litellm cannot be used for this: it validates content parts
against the schema and rejects the video part outright, which is why
`OpenAIBackend` builds its request body itself.

The endpoint needs a vision model with its multimodal projector loaded, video
decoding compiled into the server, and `ffmpeg`/`ffprobe` on the *server's* PATH
— frames are extracted server-side by shelling out to ffmpeg, at 4 fps by
default. `veb` checks `GET /props` for `modalities.video` before the run — both under
`api_base` and at the server root, since that is where llama.cpp-style servers
serve it while `api_base` ends in `/v1`. A misconfigured endpoint fails at
startup rather than erroring on every criterion, which would score each one zero
and publish a report that reads as though every video was terrible. An endpoint
that serves no `/props` is allowed through with a warning.

Cost: the clip is re-sent on every criterion call, base64-encoded at 4/3 the file
size. Keep clips short. `judge.n_frames` is ignored in this mode.

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

**What the judge gets.** The same images, ahead of the clip, described in the
prompt header — and the `references` section of the library to draw from: `REF1 —
Reference Adherence` (is what was shown the same as the images) and `REFCOV1 —
Reference Coverage` (was every reference used at all, or did the clip quietly drop
the one it found hard). Only a seed carrying references should list them; one that
lists one and whose images fail to load passes it without a model call, because
its weight is already in the denominator and a dataset problem must not read as a video
defect. `requires_references: true` in `rubrics.yaml` — carried by every criterion
in the `references` section and by nothing outside it — is what marks a criterion
that way.

References reach the judge from the dataset, not from the generator, so an
imported video (`generator=external`) is scored on them too.

### Sandboxing the agent

`sandbox=docker` runs the agent's `pi` in a throwaway container per seed, holding
that seed's workspace and nothing else.

```bash
make run ARGS="video_backend=wangp"  # builds the image first if it needs to
```

`make build` forces a rebuild — after bumping `PI_VERSION`, say, which changes
the CLI contract `PiGenerator.build_argv` is written against:

```bash
make build PI_VERSION=0.84.3 && make test
```

Without the Makefile it is `docker/build.sh` once, then `veb sandbox=docker`.

This is a correctness setting, not a deployment one. On the host the agent's
workspace sits four levels under the project root and `tools=full` puts no path
restriction on `read` or `bash` — and a real run used that to walk up and read
the genre rubric it was about to be scored against, working out from its own
reasoning that it was under evaluation. A number from such a run
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
make report                   # re-render the most recent run only
```

It references the videos rather than embedding them, so the page stays in the
low hundreds of KB and opens instantly — but it belongs to its run directory.
Move the folder, not the file.

### Comparing runs

Comparison is a separate step over report files, so runs made days apart compare
the same way as a sweep:

```bash
veb-compare runs/*/report.json                 # or: make compare
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

### The published site

Three static pages, built from the repository and deployed to GitHub Pages by
`.github/workflows/pages.yml` on every push to `master` that touches the dataset, the
report code or the published run snapshot:

| Page | What it is |
| --- | --- |
| `index.html` | What the benchmark is, and how one score is produced. |
| `atlas.html` | The rubric library (`report/atlas.py`) — every criterion, filterable by tag, with section shortcuts. |
| `performance.html` | Every published run, the criteria models fail most often, and the judge's written reason for each verdict. |

```bash
make site           # build into _site/, open _site/index.html
make site-data      # re-export runs/*/report.json -> site/data/runs.json
```

**The snapshot is the only thing CI cannot regenerate.** `runs/` is gitignored — a run
directory holds the mp4s and is hundreds of megabytes, and the Action has no GPU and no
keys. So `make site-data` exports the numbers (scores, verdicts, the judge's comments,
the arm's config; no media) into `site/data/runs.json`, and that file is committed. The
build reads it and never touches `runs/`, which is what keeps a deploy to a few seconds
of Python that can only fail on a dataset that no longer loads.

Two things the export decides on your behalf, both to stop the page telling a
comfortable lie: runs whose generator or judge was a **mock are dropped** (the mock
judge passes everything, so publishing one would put a 100.0 at the top of the table),
and the headline "best score" is taken only from runs that **covered at least five
seeds** — a 100 on one seed is a smoke test. `--include-mock` overrides the first.

Runs made before the rubric review name criteria that have since merged. The
performance page maps them through the same table the atlas uses (`PROP1` → `SEC1`,
`CLEAN1` → `FOCUS1`, …), drops `GEO1`, which has no successor, and says so under the
table rather than quietly recounting old verdicts as new ones.

One-time setup on the repository: **Settings → Pages → Source: GitHub Actions**.

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

## Building seeds from FineVideo

Fourteen hand-written seeds is too few to say anything statistical about which of the
four agent axes is holding an agent back, and hand-writing more does not scale.
`veb-seedbuild` builds seeds and their rubrics from
[FineVideo](https://huggingface.co/datasets/HuggingFaceFV/finevideo) — 18,791 real
YouTube videos, each with LLM-written metadata describing its scenes, characters,
props, editing and mood. Each seed is one source video **condensed**: a few seconds,
a handful of shots, carrying what made the original the kind of video it is.

```bash
make seeds-pilot                     # 10 videos
make seeds-report                    # read what it did
make seeds                           # the full pass (SEEDS=200)
```

A video costs roughly `2 + 1.4 × n_criteria` calls — about 22 for a 14-criterion
seed, so ~220 for the pilot and ~4,400 for two hundred. The calls are lopsided:
`select` sends the whole criterion library (~6k tokens) once per video, while the
seed judge sends only the brief (~1k) and runs a dozen times. Both judge prompts put
their bulk first and the criterion last, so consecutive calls for one video share a
long prefix a caching server can reuse — worth preserving if you edit them.

Wall time is dominated by prefill, so it is a property of your endpoint, not of the
call count. On the local Q8 27B a judge call measures around a minute, which puts the
ten-video pilot in the low hours; a hosted model (`builder_llm=litellm`) turns the
same pilot into minutes. Run it in the background — it is resumable, so a kill costs
one call.

### The builder's model

`builder_llm` is its own config group, deliberately **not** the benchmark's `judge`
group. Both builder judges are text-in, text-out — they read a brief and a criterion
and never see a frame — so `seedbuilder/client.py` is a plain
`complete(system, user) -> str` wrapper over chat-completions, with no frame
sampling, no base64 clips, and no `pi` subprocess.

```bash
veb-seedbuild builder_llm=gx10                 # the local model over HTTP (default)
veb-seedbuild builder_llm=litellm builder_llm.model=anthropic/claude-sonnet-5
veb-seedbuild builder_llm=mock                 # offline: checks the plumbing, costs nothing
```

Keeping it separate is not tidiness. `judge=pi` spawns a process per call — ~46s
against the same endpoint that answers in ~5s over HTTP, which behind a video
generation that takes minutes is invisible and in front of 4,400 text calls is
crippling. And a builder judge that *could* be handed frames is one that eventually
would be, which is exactly what the evidence routing below exists to prevent.

**The first output is not a dataset, it is a report.** A generated rubric is easy to
produce and hard to trust, so start on ten videos, read the report, edit a prompt, and
run it again — the build is resumable and a prompt edit re-runs only what that prompt
fed, so the loop is minutes.

### The two judges

A criterion generated from a video's metadata can fail in two different ways, so two
independent judges check it, each with its own prompt in `seedbuilder/prompts/`:

**The seed judge** (`judge_seed.md`) sees the generated brief and the criterion, and
deliberately *not* the source. Its question is whether the criterion is a fair thing
to ask of a video made from that brief alone: `grounded`, `ungrounded` (it needs
something the brief never says), or `unfailable` (nothing plausible fails it). The
case it exists for: "the last two seconds are silent" may be perfectly true of a music
video and perfectly checkable, but if the brief never asks for it, no agent could know
to do it — the criterion measures luck. It runs on **every** criterion, visual ones
included: a brief that says "warm morning light through a window" grounds a lighting
criterion, and one that never mentions light does not.

**The metadata judge** (`judge_metadata.md`) sees a digest of the source video and
asks whether the original itself satisfies the criterion — a bar the real video does
not clear is not a fair bar for a generated one.

### Why criteria are classified by evidence

The metadata judge reads a *description*, not pixels. It can settle "do the shots
progress in the order the brief asked for". It cannot settle "are there generation
artifacts" — an annotator writes down artifacts only when they noticed them, which is
never for a clean video, so the judge would answer confidently and meaninglessly, and
a perfectly good criterion would come back looking contradicted by its own source.

So [`evidence`](#rubrics) routes each criterion to a validator that can actually see
what it needs:

| `evidence` | Validator | Outcome |
|---|---|---|
| `description` | the metadata judge | `verified` / `contradicted` / `undetermined` |
| `container` | `ffprobe` on the source file — exact, no model | `verified` / `contradicted` |
| `pixels`, `motion`, `audio` | none in this pass | `unchecked` |

`unchecked` is a first-class outcome, not a failure. A criterion nobody could check is
not a criterion anything found fault with, and recording it as one is exactly the bug
this classification prevents. The `container` validator earns its place cheaply here:
FineVideo is 640x360 landscape throughout, so a brief that picks up a vertical-format
criterion is contradicted by ffprobe alone, with no judgement call involved.

### Nothing is dropped

**No stage deletes a criterion.** Every one the builder ever proposed survives into
the record, into the emitted `seeds.yaml` and into the report, carrying the verdicts
that were passed on it:

```yaml
  - id: SUBJ1
    bind: {subject: "the legendary shotgun"}
    scored: true
    seed_judge:   {status: grounded, reason: "the brief names the weapon as the subject"}
    verification: {status: unchecked, by: none, reason: "evidence: pixels"}
  - id: AV1
    scored: false
    seed_judge:   {status: ungrounded, reason: "the brief says nothing about audio timing"}
    verification: {status: unchecked, by: none, reason: "evidence: audio"}
```

That is what makes a build a learning artifact rather than only a result. A rejection
that was deleted teaches nothing; two hundred of them grouped by criterion tell you
whether the selection prompt over-proposes, a library entry is written too tightly, or
a judge prompt is simply wrong.

Whether a criterion is actually **scored** is therefore a separate, late-bound
decision — `seedbuilder/policy.py`, applied at emit. `permissive` scores everything
and is the default while the prompts are being tuned; `grounded` drops what the seed
judge could not ground; `strict` also drops what the source contradicted. Changing it
costs nothing:

```bash
veb-seedbuild seedbuild.emit_only=true seedbuild.policy=strict
```

### The build report

`veb-seedreport` renders the records into `build_report.html`. Its centrepiece is the
**criterion health table** — one row per criterion, with how often it was proposed and
what each judge said about it. A criterion proposed eighty times and called ungrounded
sixty of them is a selection-prompt bug; one contradicted by its own source material
nine times in ten is a library entry written too tightly. From the outside those look
identical, and the table is what tells them apart. Alongside it: rejections grouped by
criterion with the judge's reasons, the full per-seed detail, the proposal clusters
(including the sub-threshold ones the mint step did not adopt), the tag distribution,
and **the sha256 of every prompt file the build used** — a report is attributable to a
prompt revision or it is not evidence.

```bash
veb-seedreport --compare seedbuild-v1 seedbuild-v2
```

turns "did that prompt edit help" into one delta per criterion.

### The dataset it produces

`dataset_finevideo/` is a complete dataset directory — `seeds.yaml`, `rubrics.yaml`
(the curated library plus anything minted), `genres.yaml`, `tags.yaml`, `safety.yaml`
— which loads under exactly the validation `dataset/` does:

```bash
veb run.dataset_dir=dataset_finevideo
```

Seeds carry `tags` from a closed vocabulary (`seedbuilder/tags.yaml`: editing style,
pacing, whether a speaker faces camera, shot scale, …) so a corpus can be sliced
analytically, and `provenance` naming the source sample, its YouTube id, the digest
hash and the prompt revisions that produced it. The vocabulary is closed on purpose: a
free-text tagger writes "talking head", "talking-head" and "talking_head" across one
corpus and reports them as three populations.

**Linking metadata to videos.** `metadata/` and `videos/` are flat and correspond 1:1
by stem, so there is nothing to load — `veb-seedbuild` scans the metadata once into
`seedbuild/index.jsonl` and every later stage selects from that. The `data/`
directory (the raw HF parquet shards) is deliberately never read: it is 357 GB, it is
an incomplete mirror, and its `json` column is the metadata already sitting decoded
next door.

**Resuming.** Every stage writes its result to `seedbuild/records/<sample>.json`
immediately, so a kill costs at most one model call, and each result stores the hash
of the prompt that produced it — a stage is reused only if that still matches.
`--shard i/n` splits one selection across processes sharing a record store. Selection
is deterministic, so the ten videos of a pilot are the first ten of the two hundred:
scaling up reuses the pilot's records rather than rebuilding them.

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
make test                             # offline: no pi, no model, no network
python -m pytest tests/e2e -m e2e -q  # live: real pi + real model
```

The e2e tier runs the **real** agent and the **real** judge against a live model —
no LLM call is mocked. What it fakes is the video backend
(`tests/e2e/mock_video_service.py`), which answers in milliseconds instead of
minutes, so the process lifecycle can be exercised under short timeouts. It skips
itself when `pi` or the model endpoint is unavailable.

`tests/test_seedbuilder.py` covers the seed builder the same way — the model is
faked at the backend boundary (`seedbuilder/mock.py`) and everything else is the
real code. Two of its tests are the ones to keep an eye on, because they guard the
two properties the design turns on: that a `pixels` criterion reaches the seed judge
and **never** the metadata judge (asserted on the calls that were *not* made), and
that a criterion the seed judge rejects still appears in the emitted `seeds.yaml`
with its verdict attached. It builds against a miniature FineVideo dump in
`tmp_path`; the handful of tests that read the real corpus skip themselves when it
is not on the machine.
