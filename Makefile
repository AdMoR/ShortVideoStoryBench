# Build the agent jail, then run the benchmark.
#
# The two commands the README spells out — `docker/build.sh` once, then `veb
# sandbox=docker` — with the ordering enforced, so `make run` cannot score a run
# against an image that was never built or is older than the Dockerfile.
#
# Everything runs through `uv run`, not a hand-activated venv: the lockfile is
# what makes a score reproducible, and `uv run` resyncs from it before each
# invocation.
#
# Hydra overrides go in ARGS, quoted:
#
#   make run ARGS="run.max_seeds=1"  # one seed instead of eight
#   make run BACKEND=fake            # stand-in generator, no GPU minutes
#   make run SANDBOX=none            # agent on the host — see `make help`
#   make build PI_VERSION=0.84.3     # a different pi CLI

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

UV ?= uv
VEB ?= $(UV) run veb
IMAGE ?= veb-pi:latest
ARGS ?=

# The seed builder. SEEDS is the full pass's size; the pilot is fixed at 10.
SEEDBUILD ?= $(UV) run veb-seedbuild
SEEDS ?= 200
SEEDWORK ?= seedbuild-pilot

# Sandboxed by default. This is a correctness setting, not packaging: on the
# host, `tools=full` lets the agent walk up out of its workspace and read the
# rubric it is about to be scored against (runs/FINDINGS.md §4).
SANDBOX ?= docker

# The full-capability arm: the real generation tool, and every skill in skills/.
# The config default is the opposite end — `video_backend=none skills=none`, the
# unaided baseline — because that is the floor an ablation measures from. What
# you want to *run* is the agent at full strength; what you want to compare it
# against is the floor:
#
#   make run BACKEND=none SKILLS=none          # the baseline arm
#   make run ARGS="-m skills=none,all"         # both, one report each
#
# A wangp generation takes minutes per seed and the default is all 8 seeds, so
# an unqualified `make run` is an hours-long job. Use ARGS="run.max_seeds=1"
# while iterating.
BACKEND ?= wangp
SKILLS ?= all

# Marks the last successful build, so `run` builds when the Dockerfile or the
# entrypoint changed and skips it when they did not. Worth the bookkeeping: even
# a fully cached `docker build` re-exports the image, ~75s on this repo.
STAMP := .make/image-$(subst :,-,$(subst /,-,$(IMAGE)))

.DEFAULT_GOAL := help
.PHONY: help install build run smoke mock test report compare \
        seeds seeds-pilot seeds-report atlas site site-data clean

help: ## Show this help
	@echo "video-eval-bench"
	@echo
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-13s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Variables: ARGS (Hydra overrides), BACKEND=$(BACKEND), SKILLS=$(SKILLS),"
	@echo "             SANDBOX=$(SANDBOX), IMAGE=$(IMAGE), SEEDS=$(SEEDS)"

install: ## Sync the venv from uv.lock, dev extras included
	$(UV) sync --extra dev

build: ## Build the agent jail image (unconditionally)
	VEB_PI_IMAGE=$(IMAGE) docker/build.sh
	@mkdir -p $(dir $(STAMP)) && touch $(STAMP)

# The build `run` depends on. A fully cached rebuild still costs a minute of
# export, so this compares the image's own creation time against the files it
# was built from and only builds when it is actually behind — which also covers
# the case where the image exists but this stamp does not (a fresh checkout).
$(STAMP): docker/Dockerfile.pi docker/pi-entrypoint.sh
	@built=$$(docker image inspect -f '{{.Created}}' $(IMAGE) 2>/dev/null || true); \
	newest=$$(stat -c %Y $^ | sort -n | tail -1); \
	if [ -n "$$built" ] && [ "$$(date -d "$$built" +%s)" -gt "$$newest" ]; then \
		echo "$(IMAGE) is newer than docker/ — skipping build"; \
	else \
		VEB_PI_IMAGE=$(IMAGE) docker/build.sh; \
	fi
	@mkdir -p $(dir $@) && touch $@

# Keys and endpoints, for both the host and the agent's container. Created from
# the template rather than left missing, since --env-file hard-fails on it.
.env: .env.example
	@if [ ! -f $@ ]; then \
		cp $< $@; \
		echo "created .env from .env.example — fill in the keys your arms need"; \
	else \
		echo "note: .env.example changed since .env was written — check for new keys"; \
	fi

run: .env $(STAMP) ## Full-capability evaluation: real video tool + all skills, sandboxed
	$(VEB) sandbox=$(SANDBOX) video_backend=$(BACKEND) skills=$(SKILLS) $(ARGS)

smoke: .env $(STAMP) ## One real seed, stand-in generator (~10 min)
	$(VEB) experiment=smoke sandbox=$(SANDBOX) $(ARGS)

mock: ## Fully offline: synthetic video, fake verdicts — needs no image or keys
	$(VEB) experiment=mock $(ARGS)

test: ## Offline test suite (the e2e tier stays opt-in)
	$(UV) run pytest -q

report: ## Re-render the HTML report of the most recent run
	$(UV) run veb-report "$$(ls -d runs/*/ | tail -1)"

compare: ## Comparative table over every run's report.json
	$(UV) run veb-compare runs/*/report.json

# Re-render whenever dataset/rubrics.yaml changes: the atlas's whole claim is to
# be the library, so a stale one is worse than none. The pilot is optional — it
# only fills the coverage and built-seed sections.
atlas: ## Render the rubric library as one page (out/atlas.html)
	$(UV) run python -m video_eval_bench.report.atlas out/atlas.html

# The public site — the same three pages GitHub Actions publishes to Pages, built
# locally so you can look before you push. `site` reads the committed snapshot;
# `site-data` refreshes that snapshot from the runs on this machine, and its output
# is the one thing here that has to be committed for the deployed site to change.
site: ## Build the GitHub Pages site into _site/ (open _site/index.html)
	$(UV) run python -m video_eval_bench.report.site build --out _site

site-data: ## Re-export runs/*/report.json into site/data/runs.json (commit it)
	$(UV) run python -m video_eval_bench.report.site export runs/*/report.json

# ── the seed builder ─────────────────────────────────────────────────────────
# Build benchmark seeds and their rubrics from FineVideo. Start with the pilot:
# ten videos, a few minutes, and a report to read before you spend on two hundred.
#
#   make seeds-pilot            # 10 videos -> seedbuild-pilot + dataset_finevideo_pilot
#   make seeds-report           # open what it produced
#   make seeds                  # the full pass (SEEDS=200 by default)
#   make seeds ARGS="seedbuild.emit_only=true seedbuild.policy=strict"

seeds-pilot: ## Build 10 seeds from FineVideo and report on them
	$(SEEDBUILD) seedbuild=pilot $(ARGS)
	@echo "report: seedbuild-pilot/build_report.html"

seeds: ## Build SEEDS seeds from FineVideo (resumable; re-run to continue)
	$(SEEDBUILD) seedbuild.limit=$(SEEDS) $(ARGS)

seeds-report: ## Re-render the pilot's build report
	$(UV) run veb-seedreport $(SEEDWORK)

clean: ## Remove build/test caches (runs/ is data — left alone)
	rm -rf .make .pytest_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
