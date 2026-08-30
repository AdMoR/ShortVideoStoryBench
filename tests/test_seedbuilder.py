"""
Tests for the FineVideo seed builder.

Faked at the boundary, like the rest of the suite: `MockBuilderClient` stands in for
the model and everything else — prompt rendering, routing, record writing, policy,
emission — is the real code. `MockBuilderClient.calls` records every call, which is
what lets the routing tests assert on questions that were *not* asked.

The builder's model is a plain `TextLLM` (`seedbuilder/client.py`), not the
benchmark's `VisionLLM`, so nothing here needs frames, a video, or the `pi` binary.
"""

import json
from pathlib import Path

import pytest
import yaml

from video_eval_bench.dataset import load_dataset
from video_eval_bench.dataset.dataset_schemas import (
    DEFAULT_DATASET_DIR,
    RubricCriterion,
    bind_description,
)
from video_eval_bench.dataset.seed import Seed, SeedCriterion
from video_eval_bench.seedbuilder import emit as emit_module
from video_eval_bench.seedbuilder import mint as mint_module
from video_eval_bench.seedbuilder import pipeline, report, stages
from video_eval_bench.seedbuilder.digest import (
    MAX_TRANSCRIPT_LINES,
    build_digest,
    digest_sha256,
    load_metadata,
)
from video_eval_bench.seedbuilder.index import (
    IndexEntry,
    SelectionFilter,
    build_index,
    select_samples,
    shard,
)
from video_eval_bench.seedbuilder.llm import PromptRunner, parse_json_object, render
from video_eval_bench.seedbuilder.client import ClientConfig, OpenAIChatClient, build_client
from video_eval_bench.seedbuilder.mock import MockBuilderClient
from video_eval_bench.seedbuilder.policy import get_policy
from video_eval_bench.seedbuilder.probe import verify_container
from video_eval_bench.seedbuilder.records import RecordStore

FINEVIDEO = Path("/media/amor/data/finevideo")
needs_corpus = pytest.mark.skipif(
    not (FINEVIDEO / "metadata").is_dir(), reason="the FineVideo dump is not on this machine"
)


@pytest.fixture
def dataset():
    return load_dataset()


@pytest.fixture
def store(tmp_path):
    return RecordStore(tmp_path / "build")


def make_metadata(sample: str = "s", scenes: int = 2, transcript: int = 3) -> dict:
    return {
        "content_parent_category": "Education",
        "content_fine_category": "Explainers",
        "duration_seconds": 120,
        "resolution": "640x360",
        "original_video_filename": f"{sample}yt.mp4",
        "youtube_title": "How a kettle works",
        "youtube_tags": ["kettle", "science"],
        "youtube_age_limit": 0,
        "text_to_speech_word_count": 400,
        "timecoded_text_to_speech": [
            {"start": f"00:00:{i:02d}.000", "end": f"00:00:{i + 1:02d}.000", "text": f"line {i}"}
            for i in range(transcript)
        ],
        "content_metadata": {
            "title": "How a kettle works",
            "description": "An explainer about kettles.",
            "fps": 30.0,
            "characterList": [{"characterId": "1", "name": "Host", "description": "A presenter."}],
            "storylines": {"description": "Explainer", "climax": {"description": "It boils",
                                                                  "timestamp": "00:01:00.000"}},
            "scenes": [
                {
                    # sceneId deliberately 0-based, as sample_1000 is: the digest must
                    # number by position and never trust this field.
                    "sceneId": i,
                    "title": f"Scene {i}",
                    "timestamps": {"start_timestamp": "00:00:00.000",
                                   "end_timestamp": "00:00:30.000"},
                    "cast": ["Host"],
                    "mood": {"description": "Curious", "keyMoments": []},
                    "thematicElements": "Science",
                    "dynamismScore": 0.4,
                    "audioVisualCorrelation": 1.0,
                    "props": [{"name": "Kettle",
                               "timestamp": {"start_timestamp": "0", "end_timestamp": "1"}}],
                    "videoEditingDetails": [
                        {"description": "Close-up of the kettle",
                         "timestamps": {"start_timestamp": "0", "end_timestamp": "5"}}
                    ],
                    "narrativeProgression": [
                        # Flat string timestamp here, dict above — the asymmetry the
                        # digest exists to absorb.
                        {"description": "The host introduces the kettle",
                         "timestamp": "00:00:00.000"}
                    ],
                    "characterInteraction": [],
                }
                for i in range(scenes)
            ],
            "qAndA": [],
            "trimmingSuggestions": [],
        },
    }


@pytest.fixture
def corpus(tmp_path):
    """A miniature FineVideo dump: three metadata files, two with a video beside them."""
    root = tmp_path / "finevideo"
    (root / "metadata").mkdir(parents=True)
    (root / "videos").mkdir()
    for i, category in enumerate(["Education", "Entertainment", "Sports"]):
        meta = make_metadata(f"sample_{i}")
        meta["content_parent_category"] = category
        (root / "metadata" / f"sample_{i}.json").write_text(json.dumps(meta))
        if i < 2:
            (root / "videos" / f"sample_{i}.mp4").write_bytes(b"not really a video")
    return root


# ── digest ────────────────────────────────────────────────────────────────────


def test_digest_is_deterministic():
    """The digest's hash gates every stage below it, so it must not drift."""
    meta = make_metadata()
    assert digest_sha256(build_digest(meta)) == digest_sha256(build_digest(meta))


def test_digest_numbers_scenes_by_position_not_by_sceneId():
    """`sceneId` is 0-based in some FineVideo files and 1-based in others."""
    digest = build_digest(make_metadata(scenes=3))
    assert "### Scene 1:" in digest and "### Scene 3:" in digest
    assert "### Scene 0:" not in digest


def test_digest_reads_both_timestamp_spellings():
    """`timestamps` on shots, a flat string on beats — both must survive."""
    digest = build_digest(make_metadata())
    assert "Close-up of the kettle" in digest       # videoEditingDetails[].timestamps
    assert "The host introduces the kettle" in digest  # narrativeProgression[].timestamp


def test_digest_downsamples_a_long_transcript_across_the_whole_video():
    """
    Truncating to the first N would describe only the opening, which is exactly the
    bias that makes a condensed brief cover nothing but the intro.
    """
    digest = build_digest(make_metadata(transcript=500))
    narration = [line for line in digest.splitlines() if line.startswith("[00:")]
    assert len(narration) <= MAX_TRANSCRIPT_LINES
    assert "line 0" in digest and "line 499" in digest


def test_digest_stays_bounded_for_a_huge_source():
    small = build_digest(make_metadata(scenes=1, transcript=3))
    huge = build_digest(make_metadata(scenes=25, transcript=500))
    assert len(huge) < 12 * len(small)


@needs_corpus
def test_digest_handles_the_real_corpus_extremes():
    sizes = {}
    for sample in ("sample_1000", "sample_8660", "sample_13376"):
        digest = build_digest(load_metadata(FINEVIDEO / "metadata" / f"{sample}.json"))
        assert digest.startswith("# Source video")
        sizes[sample] = len(digest)
    assert max(sizes.values()) < 20_000, sizes


# ── index and selection ───────────────────────────────────────────────────────


def test_index_pairs_metadata_with_video_and_skips_the_unpaired(corpus):
    entries = build_index(corpus)
    assert [e.sample_id for e in entries] == ["sample_0", "sample_1"]
    assert entries[0].youtube_id == "sample_0yt"  # stem of original_video_filename
    assert entries[0].n_scenes == 2


def test_index_round_trips_through_jsonl(corpus, tmp_path):
    from video_eval_bench.seedbuilder.index import load_index

    out = tmp_path / "index.jsonl"
    written = build_index(corpus, out=out)
    assert load_index(out) == written


def test_selection_is_stratified_across_categories():
    entries = [
        IndexEntry(
            sample_id=f"s{i}", video=f"v{i}", youtube_id="", parent_category=category,
            fine_category="", duration_seconds=100, n_scenes=3, resolution="640x360",
            fps=30.0, word_count=100, age_limit=0, n_characters=1, has_transcript=True,
        )
        for category, count in (("Education", 20), ("Sports", 2))
        for i in range(count)
    ]
    for i, entry in enumerate(entries):
        entry.sample_id = f"{entry.parent_category}_{i}"
    picked = select_samples(entries, limit=4)
    categories = [e.parent_category for e in picked]
    assert categories.count("Sports") == 2, "the small category must not be swamped"


def test_a_pilot_selection_is_a_prefix_of_the_full_one():
    """Tuning on 10 videos must not mean rebuilding them when you scale to 200."""
    entries = [
        IndexEntry(
            sample_id=f"s{i}", video=f"v{i}", youtube_id="", parent_category=f"c{i % 4}",
            fine_category="", duration_seconds=100, n_scenes=3, resolution="640x360",
            fps=30.0, word_count=100, age_limit=0, n_characters=1, has_transcript=True,
        )
        for i in range(80)
    ]
    small = [e.sample_id for e in select_samples(entries, limit=8, seed=7)]
    large = [e.sample_id for e in select_samples(entries, limit=40, seed=7)]
    assert large[: len(small)] == small


def test_selection_filters_reject_the_unusable():
    def entry(**overrides):
        base = dict(
            sample_id="s", video="v", youtube_id="", parent_category="Education",
            fine_category="", duration_seconds=100, n_scenes=3, resolution="640x360",
            fps=30.0, word_count=100, age_limit=0, n_characters=1, has_transcript=True,
        )
        return IndexEntry(**{**base, **overrides})

    filters = SelectionFilter()
    assert filters.admits(entry())
    assert not filters.admits(entry(duration_seconds=5))
    assert not filters.admits(entry(duration_seconds=5000))
    assert not filters.admits(entry(n_scenes=1))
    assert not filters.admits(entry(age_limit=18))
    assert not filters.admits(entry(has_transcript=False))


def test_shards_are_disjoint_and_cover_everything():
    entries = [
        IndexEntry(
            sample_id=f"s{i}", video="", youtube_id="", parent_category="c",
            fine_category="", duration_seconds=100, n_scenes=2, resolution="",
            fps=30.0, word_count=0, age_limit=0, n_characters=0, has_transcript=True,
        )
        for i in range(60)
    ]
    parts = [{e.sample_id for e in shard(entries, f"{i}/4")} for i in range(4)]
    assert set.union(*parts) == {e.sample_id for e in entries}
    assert sum(len(p) for p in parts) == 60


# ── prompts ───────────────────────────────────────────────────────────────────


def test_every_prompt_slot_is_filled(dataset):
    """An unfilled slot must raise at render time, not reach the model as `{{X}}`."""
    render("synthesize", digest="d", tag_vocabulary="t")
    render("select", digest="d", prompt="p", library="l", tags="t")
    render("judge_seed", prompt="p", criterion_id="X1", criterion_name="n",
           criterion_description="d")
    render("judge_metadata", digest="d", prompt="p", criterion_id="X1",
           criterion_name="n", criterion_description="d")
    render("mint", library="l", sections="s", tags="t", count=1, n_samples=1, proposals="p")
    with pytest.raises(KeyError, match="DIGEST"):
        render("synthesize", tag_vocabulary="t")


def test_the_seed_judge_is_never_shown_the_source_material():
    """
    The whole point of that judge is to test the brief in isolation. If the digest
    leaked into its prompt it would ground every criterion from the source and the
    safeguard would silently stop working.
    """
    rendered = render("judge_seed", prompt="a brief", criterion_id="X1",
                      criterion_name="n", criterion_description="d")
    assert "{{DIGEST}}" not in rendered
    assert "digest" not in rendered.lower().replace("judge_metadata", "")


def test_json_parsing_tolerates_fences_and_chatter():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure!\n{"a": 1}\nHope that helps.') == {"a": 1}
    assert parse_json_object("not json at all") is None


# ── routing: the correction this design exists for ────────────────────────────


async def test_a_pixels_criterion_never_reaches_the_metadata_judge(dataset):
    """
    The metadata judge reads a description, which mentions artifacts only when an
    annotator noticed them — never for a clean video. Asking it anyway returns a
    confident verdict that means nothing, and marks a good criterion as contradicted
    by its own source. `evidence` exists to stop the question being asked.
    """
    backend = MockBuilderClient()
    runner = PromptRunner(backend)
    criteria = [{"id": cid, "bind": {}} for cid in ("SEQ1", "ART1", "CUT1", "AV1")]

    await stages.judge_seed_pass(runner, "a brief", criteria, dataset)
    result = await stages.judge_metadata_pass(
        runner, "DIGEST", "a brief", criteria, dataset
    )

    asked = {c["criterion_id"] for c in backend.calls if c["kind"] == "judge_metadata"}
    assert asked == {"SEQ1"}, "only the description-class criterion may be asked"

    seen = {c["criterion_id"] for c in backend.calls if c["kind"] == "judge_seed"}
    assert seen == {"SEQ1", "ART1", "CUT1", "AV1"}, "the seed judge runs on all of them"

    by_id = {c["id"]: c for c in result.data["criteria"]}
    assert by_id["ART1"]["verification"]["status"] == "unchecked"
    assert by_id["ART1"]["verification"]["by"] == "none"
    assert "pixels" in by_id["ART1"]["verification"]["reason"]


async def test_unchecked_is_not_a_failure(dataset):
    """A criterion nobody looked at must never be recorded as one anything faulted."""
    runner = PromptRunner(MockBuilderClient())
    result = await stages.judge_metadata_pass(
        runner, "DIGEST", "a brief", [{"id": "ART1", "bind": {}}], dataset
    )
    verification = result.data["criteria"][0]["verification"]
    assert verification["status"] == "unchecked"
    assert verification["status"] != "contradicted"


def test_the_container_validator_settles_orientation_without_a_model(dataset):
    ar1 = dataset.rubrics.get("AR1")
    landscape = {"width": 640, "height": 360, "orientation": "landscape"}

    verdict = verify_container(ar1, landscape, prompt="A 9-second vertical (9:16) clip")
    assert verdict["status"] == "contradicted"
    assert verdict["by"] == "container"
    assert "640x360" in verdict["comment"]

    verdict = verify_container(ar1, landscape, prompt="A 10-second 16:9 tourism spot")
    assert verdict["status"] == "verified"

    verdict = verify_container(ar1, landscape, prompt="A short clip of a kettle")
    assert verdict["status"] == "undetermined", "no orientation named — never guess"

    verdict = verify_container(ar1, None, prompt="a vertical clip")
    assert verdict["status"] == "unchecked", "ffprobe missing is not a contradiction"


# ── retention: nothing is dropped ─────────────────────────────────────────────


async def test_a_rejected_criterion_survives_into_the_emitted_dataset(
    dataset, store, tmp_path, corpus
):
    """
    The regression test for the principle. A criterion the seed judge rejects must
    still appear in the record, in `seeds.yaml` and in the report, carrying the
    verdict that rejected it — only `scored` changes. A rejection that was deleted
    teaches nothing, and reading rejections is how the prompts get tuned.
    """
    backend = MockBuilderClient(criteria=["SEQ1", "AV1"])  # AV1 comes back ungrounded
    runner = PromptRunner(backend)
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, runner, dataset, corpus / "metadata")

    record = store.load(entries[0].sample_id)
    judged = {c["id"]: c for c in record.judged_criteria()}
    assert judged["AV1"]["seed_judge"]["status"] == "ungrounded"

    out = tmp_path / "ds"
    emit_module.emit([record], out, get_policy("grounded"), dataset)
    seeds = yaml.safe_load((out / "seeds.yaml").read_text())["seeds"]
    criteria = {c["id"]: c for c in seeds[0]["rubrics"]}

    assert {"SEQ1", "AV1"} <= set(criteria), "the rejected criterion is still on disk"
    assert criteria["AV1"]["scored"] is False
    assert criteria["AV1"]["seed_judge"]["status"] == "ungrounded"
    assert criteria["AV1"]["seed_judge"]["reason"]
    assert criteria["SEQ1"]["scored"] is True


async def test_policy_changes_scored_flags_without_any_model_call(
    dataset, store, tmp_path, corpus
):
    """Re-policying must be free — that is what makes it a knob rather than a commitment."""
    backend = MockBuilderClient(criteria=["SEQ1", "AV1", "HOOK1"])
    runner = PromptRunner(backend)
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, runner, dataset, corpus / "metadata")
    record = store.load(entries[0].sample_id)

    spent = len(backend.calls)
    scored = {}
    for name in ("permissive", "grounded", "strict"):
        out = tmp_path / name
        emit_module.emit([record], out, get_policy(name), dataset)
        seeds = yaml.safe_load((out / "seeds.yaml").read_text())["seeds"]
        scored[name] = {c["id"] for c in seeds[0]["rubrics"] if c["scored"]}

    assert len(backend.calls) == spent, "emission must not call the model"
    # Stated as differences rather than exact sets, so the test is about the policy
    # rather than about which criteria the mock happens to pick.
    assert {"SEQ1", "AV1", "HOOK1"} <= scored["permissive"]
    assert scored["permissive"] - scored["grounded"] == {"AV1"}    # ungrounded
    assert scored["grounded"] - scored["strict"] == {"HOOK1"}      # contradicted


def test_sub_threshold_proposals_are_kept_not_discarded(tmp_path):
    from video_eval_bench.seedbuilder.records import BuildRecord, StageResult

    records = []
    for i in range(2):
        record = BuildRecord(sample_id=f"sample_{i}")
        record.put(
            StageResult(
                stage="select",
                data={"criteria": [], "proposals": [
                    {"name": "Caption Persistence",
                     "description": "A caption stays on screen for its shot.",
                     "dimension": "technical", "evidence": "pixels", "why": ""}
                ]},
            )
        )
        records.append(record)

    clusters = mint_module.cluster_proposals(records)
    assert len(clusters) == 1 and clusters[0]["n_samples"] == 2

    path = tmp_path / "proposals.jsonl"
    mint_module.write_proposals(clusters, path)
    assert len(mint_module.load_proposals(path)) == 1


# ── checkpointing ─────────────────────────────────────────────────────────────


async def test_a_second_run_spends_nothing(dataset, store, corpus):
    backend = MockBuilderClient()
    entries = build_index(corpus)

    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    first = len(backend.calls)
    assert first > 0

    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    assert len(backend.calls) == first, "a fresh record must issue no calls"


async def test_editing_a_prompt_reruns_only_what_it_fed(dataset, store, corpus, monkeypatch):
    """
    Editing `judge_seed.md` must re-judge without re-synthesizing. Without the prompt
    hash, a build spanning an edit would be half one version and half the other, and
    its report would average the two.
    """
    from video_eval_bench.seedbuilder import llm

    backend = MockBuilderClient()
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    backend.calls.clear()

    real = llm.prompt_sha
    monkeypatch.setattr(
        llm, "prompt_sha", lambda name: "changed" if name == "judge_seed" else real(name)
    )
    monkeypatch.setattr(stages, "prompt_sha", llm.prompt_sha)
    monkeypatch.setattr(pipeline, "prompt_sha", llm.prompt_sha)

    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    kinds = {c["kind"] for c in backend.calls}
    assert "judge_seed" in kinds
    assert "synthesize" not in kinds and "select" not in kinds


async def test_a_changed_digest_invalidates_everything_below_it(dataset, store, corpus):
    backend = MockBuilderClient()
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    backend.calls.clear()

    meta_path = corpus / "metadata" / f"{entries[0].sample_id}.json"
    meta = json.loads(meta_path.read_text())
    meta["content_metadata"]["title"] = "A completely different video"
    meta_path.write_text(json.dumps(meta))

    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    assert {"synthesize", "select", "judge_seed"} <= {c["kind"] for c in backend.calls}


def test_an_unreadable_record_is_rebuilt_rather_than_fatal(store):
    store.records_dir.mkdir(parents=True)
    (store.records_dir / "sample_9.json").write_text("{ truncated")
    record = store.load("sample_9")
    assert record.sample_id == "sample_9" and not record.stages


# ── schema ────────────────────────────────────────────────────────────────────


def test_a_criterion_must_declare_its_evidence_class():
    with pytest.raises(Exception):
        RubricCriterion(id="X1", dimension="craft", name="n", description="d", weight=1)


def test_binds_must_match_the_placeholders_the_description_uses():
    with pytest.raises(ValueError, match="binds"):
        RubricCriterion(id="X1", dimension="craft", evidence="pixels", name="n",
                        description="The {subject} stays the same.", weight=1, binds=[])
    with pytest.raises(ValueError, match="binds"):
        RubricCriterion(id="X1", dimension="craft", evidence="pixels", name="n",
                        description="Nothing to bind.", weight=1, binds=["subject"])


def test_binding_substitutes_into_the_question_the_judge_is_asked():
    criterion = RubricCriterion(
        id="X1", dimension="craft", evidence="pixels", name="n", weight=1,
        description="The {subject} keeps its colour.", binds=["subject"],
    )
    assert bind_description(criterion, {"subject": "red kettle"}) == (
        "The red kettle keeps its colour."
    )
    with pytest.raises(ValueError, match="unbound"):
        bind_description(criterion, {})


def test_a_seed_may_still_write_a_bare_criterion_id():
    seed = Seed(seed_id="s", category="c", rubrics=["SUBJ1"], prompt="p")
    assert seed.rubrics == [SeedCriterion(id="SUBJ1")]


def test_load_rejects_a_bind_the_criterion_does_not_declare(tmp_path):
    import shutil

    shutil.copytree(DEFAULT_DATASET_DIR, tmp_path / "dataset")
    seeds = tmp_path / "dataset" / "seeds.yaml"
    seeds.write_text(
        seeds.read_text().replace(
            "      - SUBJ1\n", "      - id: SUBJ1\n        bind: {nonesuch: x}\n", 1
        )
    )
    with pytest.raises(ValueError, match="nonesuch"):
        load_dataset(tmp_path / "dataset")


def test_load_rejects_a_tag_outside_the_vocabulary(tmp_path, dataset, store, corpus):
    out = tmp_path / "ds"
    from video_eval_bench.seedbuilder.records import BuildRecord, StageResult

    record = BuildRecord(sample_id="sample_0")
    record.source = {"parent_category": "Education", "duration_seconds": 10}
    record.put(StageResult(stage="digest", data={"digest": "d", "sha256": "x"}))
    record.put(StageResult(stage="synthesize", data={"prompt": "p", "tags": {"pacing": ["blazing"]}}))
    record.put(StageResult(stage="judge_seed", data={"criteria": [
        {"id": "SEQ1", "bind": {}, "seed_judge": {"status": "grounded", "reason": ""}}
    ]}))
    record.put(StageResult(stage="judge_metadata", data={"criteria": [
        {"id": "SEQ1", "verification": {"status": "verified", "by": "metadata"}}
    ]}))
    emit_module.emit([record], out, get_policy("permissive"), dataset)

    with pytest.raises(ValueError, match="blazing"):
        load_dataset(out)


# ── emission and report ───────────────────────────────────────────────────────


async def test_the_emitted_dataset_loads_under_the_existing_validation(
    dataset, store, tmp_path, corpus
):
    runner = PromptRunner(MockBuilderClient())
    entries = build_index(corpus)
    records = await pipeline.build_all(entries, store, runner, dataset, corpus / "metadata")

    out = tmp_path / "ds"
    emit_module.emit(records, out, get_policy("permissive"), dataset)
    built = load_dataset(out)

    assert len(built.seeds) == len(entries)
    seed = built.seeds[0]
    assert seed.provenance and seed.provenance.source == "finevideo"
    assert seed.provenance.prompt_hashes["judge_seed"]
    assert seed.tags["subject_kind"] == ["object"]
    assert built.criteria_for(seed), "the judge must find criteria to grade"


async def test_criteria_for_returns_only_scored_criteria(dataset, store, tmp_path, corpus):
    runner = PromptRunner(MockBuilderClient(criteria=["SEQ1", "AV1"]))
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, runner, dataset, corpus / "metadata")

    out = tmp_path / "ds"
    emit_module.emit([store.load(entries[0].sample_id)], out, get_policy("grounded"), dataset)
    built = load_dataset(out)
    seed = built.seeds[0]

    assert {"SEQ1", "AV1"} <= set(seed.criterion_ids())
    graded = [c.id for c in built.criteria_for(seed)]
    assert "SEQ1" in graded
    assert "AV1" not in graded, "the ungrounded criterion is on the seed but not graded"


async def test_the_report_renders_and_counts_what_each_judge_said(
    dataset, store, corpus
):
    runner = PromptRunner(MockBuilderClient(criteria=["SEQ1", "ART1", "AV1", "HOOK1"]))
    entries = build_index(corpus)
    await pipeline.build_all(entries, store, runner, dataset, corpus / "metadata")

    data = report.report_data(store, dataset)
    health = data["health"]
    assert health["AV1"]["seed_judge"]["ungrounded"] == len(entries)
    assert health["ART1"]["verification"]["unchecked"] == len(entries)
    assert health["HOOK1"]["verification"]["contradicted"] == len(entries)
    assert health["ART1"]["evidence"] == "pixels"
    assert data["summary"]["prompt_hashes"]["judge_seed"]

    out = report.render(store, dataset)
    html = out.read_text()
    assert "Criterion health" in html and "AV1" in html
    assert out.with_suffix(".json").exists()


async def test_the_report_html_is_well_formed(dataset, store, corpus):
    from test_html_report import _Wellformed

    runner = PromptRunner(MockBuilderClient())
    entries = build_index(corpus)
    await pipeline.build_all(entries, store, runner, dataset, corpus / "metadata")

    parser = _Wellformed()
    parser.feed(report.render(store, dataset).read_text())
    parser.close()
    assert not parser.errors, parser.errors


async def test_comparing_two_builds_shows_the_delta(dataset, tmp_path, corpus):
    entries = build_index(corpus)
    stores = []
    for name, criteria in (("a", ["SEQ1", "AV1"]), ("b", ["SEQ1"])):
        store = RecordStore(tmp_path / name)
        await pipeline.build_all(
            entries, store, PromptRunner(MockBuilderClient(criteria=criteria)),
            dataset, corpus / "metadata",
        )
        stores.append(store)

    rows = report.compare(*stores, dataset)
    by_id = {row["id"]: row for row in rows}
    assert by_id["AV1"]["left"]["proposed"] == len(entries)
    assert by_id["AV1"]["right"]["proposed"] == 0
    assert "grounded" in report.format_comparison(rows)


# ── the model client ──────────────────────────────────────────────────────────


def test_the_builder_client_is_text_only():
    """
    No frames, no clip, no subprocess. The signature is the guarantee: a builder
    judge that could be handed media is one that eventually would be, and the
    `evidence` routing exists precisely to keep the metadata judge away from
    questions its evidence cannot answer.
    """
    import inspect

    from video_eval_bench.seedbuilder.client import TextLLM

    parameters = list(inspect.signature(TextLLM.complete).parameters)
    assert parameters == ["self", "system", "user"]


def test_the_chat_client_builds_a_plain_completions_request():
    client = OpenAIChatClient(
        ClientConfig(kind="openai", model="m", api_base="http://host/v1", api_key="k")
    )
    assert client._url() == "http://host/v1/chat/completions"
    assert client._headers()["Authorization"] == "Bearer k"

    body = client.build_body("SYS", "USER")
    assert body["model"] == "m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == "SYS"
    assert "images" not in body and "input_video" not in json.dumps(body)


def test_the_chat_client_reads_the_shapes_servers_actually_return():
    from video_eval_bench.seedbuilder.client import _first_message

    assert _first_message({"choices": [{"message": {"content": "hi"}}]}) == "hi"
    assert _first_message({"choices": [{"text": "hi"}]}) == "hi"
    # A reasoning model may return content as a list of parts.
    assert _first_message(
        {"choices": [{"message": {"content": [{"text": "hi"}, {"text": "!"}]}}]}
    ) == "hi!"
    with pytest.raises(ValueError, match="no choices"):
        _first_message({"choices": []})


def test_an_openai_client_without_an_endpoint_fails_at_construction():
    """Not on the first call, three stages into a build."""
    with pytest.raises(ValueError, match="api_base"):
        build_client(ClientConfig(kind="openai", model="m"))
    with pytest.raises(ValueError, match="unknown builder llm kind"):
        build_client(ClientConfig(kind="pi", model="m"))


async def test_a_non_json_reply_is_retried_then_recorded():
    class Flaky:
        def __init__(self):
            self.n = 0

        async def complete(self, system, user):
            self.n += 1
            return '{"status": "grounded"}' if self.n >= 3 else "I think it is fine."

    client = Flaky()
    runner = PromptRunner(client, attempts=3, backoff_seconds=0)
    result = await runner.run("judge_seed", prompt="p", criterion_id="X1",
                              criterion_name="n", criterion_description="d")
    assert result.ok and result.data["status"] == "grounded"
    assert client.n == 3 and runner.calls == 3

    runner = PromptRunner(MockBuilderClient(fail=True), attempts=2, backoff_seconds=0)
    result = await runner.run("judge_seed", prompt="p", criterion_id="X1",
                              criterion_name="n", criterion_description="d")
    assert not result.ok and "not a JSON object" in result.error


def test_diagnose_names_the_failure_mode():
    """
    A bare "not a JSON object" cannot tell a truncated reply from a refusal, and
    those want different fixes — one is a token budget, the other is a prompt. The
    parse check comes first because brace counting misreads a `{` inside a string.
    """
    from video_eval_bench.seedbuilder.llm import diagnose

    assert "parses" in diagnose('{"a": "a { brace in a string"}')
    assert "truncated" in diagnose('{"a": 1, "b": [2')
    assert "prose" in diagnose("I think the answer is yes.")
    assert "reasoning block only" in diagnose("<think>weighing it up</think>")
    assert "empty" in diagnose("")


def test_a_reasoning_block_is_stripped_rather_than_parsed_around():
    """
    A scratchpad routinely contains *draft* JSON. Taking the first `{` in the string
    would return the draft instead of the answer — silently, and with the wrong
    verdict attached to a criterion.
    """
    reply = '<think>maybe {"status": "ungrounded"} — no, wait</think>{"status": "grounded"}'
    assert parse_json_object(reply) == {"status": "grounded"}


async def test_a_failed_stage_keeps_what_the_model_actually_said(dataset, store, corpus):
    """
    Without the raw reply a failure is unattributable, and the whole build is meant
    to be readable after the fact.
    """
    runner = PromptRunner(MockBuilderClient(fail=True), attempts=1, backoff_seconds=0)
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, runner, dataset, corpus / "metadata")

    stage = store.load(entries[0].sample_id).get("synthesize")
    assert not stage.ok
    assert stage.raw == "sorry, no."
    assert "prose, no object" in stage.error


async def test_each_judge_prompt_invalidates_only_its_own_calls(
    dataset, store, corpus, monkeypatch
):
    """
    The two judges are separate stages because they are keyed on separate prompts.
    They were one stage until editing `judge_metadata.md` cost ~180 seed-judge calls
    to change 18 metadata verdicts — a tuning loop that expensive is one nobody runs.
    """
    from video_eval_bench.seedbuilder import llm

    backend = MockBuilderClient()
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    backend.calls.clear()

    real = llm.prompt_sha
    monkeypatch.setattr(
        llm, "prompt_sha", lambda n: "changed" if n == "judge_metadata" else real(n)
    )
    monkeypatch.setattr(stages, "prompt_sha", llm.prompt_sha)
    monkeypatch.setattr(pipeline, "prompt_sha", llm.prompt_sha)

    await pipeline.build_one(entries[0], store, PromptRunner(backend), dataset,
                             corpus / "metadata")
    kinds = [c["kind"] for c in backend.calls]
    assert "judge_metadata" in kinds
    assert "judge_seed" not in kinds, "the other judge must not be re-run"
    assert "synthesize" not in kinds and "select" not in kinds


async def test_the_metadata_judge_is_shown_the_brief(dataset, store, corpus):
    """
    Criteria are written against the brief — "the setting the brief described", "the
    order the brief asked for". Without it the judge answered `undetermined` to all
    of them and said so: "the brief itself is not provided, so I cannot verify".
    """
    backend = MockBuilderClient(criteria=["SEQ1"])
    runner = PromptRunner(backend)
    brief = "A 6 second clip in 3 shots: (1) a kettle; (2) steam; (3) a hand."
    await stages.judge_metadata_pass(runner, "DIGEST", brief, [{"id": "SEQ1"}], dataset)

    sent = next(c["system"] for c in backend.calls if c["kind"] == "judge_metadata")
    assert brief in sent
    assert "DIGEST" in sent, "and still the source description it is judging"


async def test_a_half_judged_record_reports_unchecked_rather_than_inventing(
    dataset, store, corpus
):
    """
    The metadata pass can be missing — still running, or errored. `verification` is
    then absent rather than invented, and the policy reads that as `unchecked`: the
    honest answer for a criterion nothing looked at.
    """
    runner = PromptRunner(MockBuilderClient(criteria=["SEQ1", "ART1"]))
    entries = build_index(corpus)
    record = store.load(entries[0].sample_id)
    result = await stages.judge_seed_pass(
        runner, "a brief", [{"id": "SEQ1"}, {"id": "ART1"}], dataset
    )
    record.put(result)

    merged = record.judged_criteria()
    assert {c["id"] for c in merged} == {"SEQ1", "ART1"}
    assert all(c.get("verification") is None for c in merged)
    assert get_policy("strict").apply(merged)[0]["scored"] is True


# ── the rubric library: categories and tags ───────────────────────────────────


def test_sections_and_tags_decide_nothing(dataset):
    """
    The load-bearing property. Nothing in the library attaches a criterion to a seed
    — no `applies_when`, no `applicable()`, no per-section bundle. A category applied
    wholesale grades a video on questions its brief never asked, which is the failure
    the genre-selected rubrics were removed for.
    """
    assert not hasattr(dataset.rubrics, "applicable")
    for section in dataset.rubrics.sections:
        assert not hasattr(section, "applies_when")


def test_every_criterion_is_tagged_from_the_closed_vocabulary(dataset):
    """An untagged criterion is in the library and invisible to every grouping."""
    vocabulary = set(dataset.rubrics.tag_vocabulary)
    assert vocabulary, "the library declares a vocabulary"
    for criterion in dataset.rubrics.criteria:
        assert criterion.tags, f"{criterion.id} carries no tags"
        assert set(criterion.tags) <= vocabulary, f"{criterion.id} has tags off-vocabulary"


def test_a_tag_outside_the_vocabulary_fails_the_load():
    """Closed for the same reason the seed tag vocabulary is: drift splits columns."""
    from video_eval_bench.dataset.dataset_schemas import RubricLibrary

    with pytest.raises(ValueError, match="outside `criterion_tags`"):
        RubricLibrary(
            dimensions=[{"key": "craft", "name": "Craft"}],
            criterion_tags={"subject": ["people"]},
            sections=[{"key": "general", "name": "G", "criteria": [{
                "id": "X1", "dimension": "craft", "evidence": "pixels",
                "tags": ["people", "invented"], "name": "X", "description": "x", "weight": 1,
            }]}],
        )


def test_a_tag_declared_in_two_groups_fails_the_load():
    """The namespace is flat, so `group_of` must have one answer."""
    from video_eval_bench.dataset.dataset_schemas import RubricLibrary

    with pytest.raises(ValueError, match="declared in two groups"):
        RubricLibrary(
            dimensions=[{"key": "craft", "name": "Craft"}],
            criterion_tags={"subject": ["people"], "span": ["people"]},
            sections=[{"key": "general", "name": "G", "criteria": [{
                "id": "X1", "dimension": "craft", "evidence": "pixels",
                "tags": ["people"], "name": "X", "description": "x", "weight": 1,
            }]}],
        )


def test_the_music_criteria_are_findable_by_tag(dataset):
    """
    The gap this work was for: a musical-continuity check that reports can group.

    Speech and music share one `audio` tag and one section — they were split in two
    and neither half answered "how much of this benchmark is about sound" on its own.
    """
    audio = {c.id for c in dataset.rubrics.with_tag("audio")}
    assert {"MUSCONT1", "SPEECH1", "LIPSYNC1"} <= audio
    assert dataset.rubrics.section_of("MUSCONT1") == "audio"
    assert dataset.rubrics.group_of("audio") == "subject"
    assert "cross_shot" in dataset.rubrics.get("MUSCONT1").tags


def test_sub_themes_are_their_own_tag_group(dataset):
    """
    `brand` and `diagram` are niches inside other subjects, not peers of them. The
    group is what keeps a report from listing them beside `people` and `environment`
    as though the library were a third about branding.
    """
    lib = dataset.rubrics
    assert lib.group_of("brand") == lib.group_of("diagram") == "sub_theme"
    assert lib.group_of("people") == "subject"
    # Every sub-theme criterion also carries a broad subject, so filtering by the
    # broad axis never loses it.
    subjects = set(lib.criterion_tags["subject"])
    for tag in lib.criterion_tags["sub_theme"]:
        for c in lib.with_tag(tag):
            assert subjects & set(c.tags), c.id


def test_a_criterion_defined_twice_fails_the_load():
    """Two definitions of one id would give a report column two meanings."""
    from video_eval_bench.dataset.dataset_schemas import RubricLibrary

    criterion = {
        "id": "X1", "dimension": "craft", "evidence": "pixels",
        "name": "X", "description": "x", "weight": 1,
    }
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        RubricLibrary(
            dimensions=[{"key": "craft", "name": "Craft"}],
            sections=[
                {"key": "general", "name": "G", "criteria": [criterion]},
                {"key": "music", "name": "M", "criteria": [criterion]},
            ],
        )


def test_a_flat_library_is_refused(tmp_path):
    """The old shape must fail loudly rather than load as a library with no axes."""
    from video_eval_bench.dataset.dataset_utils import load_rubrics

    (tmp_path / "rubrics.yaml").write_text(
        yaml.safe_dump({"dimensions": [{"key": "craft", "name": "Craft"}], "criteria": []})
    )
    with pytest.raises(ValueError, match="no `sections:` key"):
        load_rubrics(tmp_path)


def test_a_seed_local_id_colliding_with_the_library_fails_the_load(tmp_path):
    """A local id shadowing a library id would silently redefine a shared column."""
    import shutil

    for name in ("rubrics.yaml", "genres.yaml", "safety.yaml"):
        shutil.copyfile(DEFAULT_DATASET_DIR / name, tmp_path / name)
    (tmp_path / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [{
        "seed_id": "s1", "category": "entertainment", "prompt": "a short.",
        "rubrics": ["SUBJ1"],
        "local_criteria": [{
            "id": "SUBJ1", "dimension": "craft", "evidence": "pixels",
            "name": "Shadow", "description": "x", "weight": 1,
        }],
    }]}))
    with pytest.raises(ValueError, match="ids the library already uses"):
        load_dataset(tmp_path)


def test_a_seed_local_criterion_is_graded_like_any_other(tmp_path):
    """A criterion only one brief needs still reaches the judge by the same path."""
    import shutil

    for name in ("rubrics.yaml", "genres.yaml", "safety.yaml"):
        shutil.copyfile(DEFAULT_DATASET_DIR / name, tmp_path / name)
    (tmp_path / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [{
        "seed_id": "s1", "category": "entertainment", "prompt": "a short.",
        "rubrics": ["SUBJ1", {"id": "s1.LANG1"}],
        "local_criteria": [{
            "id": "s1.LANG1", "dimension": "fidelity", "evidence": "audio",
            "name": "Speech Language Match", "description": "The narration is in Romanian.",
            "weight": 1,
        }],
    }]}))
    ds = load_dataset(tmp_path)
    graded = {c.id: c for c in ds.criteria_for(ds.seeds[0])}
    assert "s1.LANG1" in graded
    assert graded["s1.LANG1"].description == "The narration is in Romanian."
    assert [c.id for c in ds.criteria_for(ds.seeds[0])][-1] == "s1.LANG1", "local ones trail"


def test_a_local_criterion_nothing_lists_fails_the_load(tmp_path):
    """A definition no `rubrics` entry names would never be judged, and never say so."""
    import shutil

    for name in ("rubrics.yaml", "genres.yaml", "safety.yaml"):
        shutil.copyfile(DEFAULT_DATASET_DIR / name, tmp_path / name)
    (tmp_path / "seeds.yaml").write_text(yaml.safe_dump({"seeds": [{
        "seed_id": "s1", "category": "entertainment", "prompt": "a short.",
        "rubrics": ["SUBJ1"],
        "local_criteria": [{
            "id": "s1.X1", "dimension": "craft", "evidence": "pixels",
            "name": "X", "description": "x", "weight": 1,
        }],
    }]}))
    with pytest.raises(ValueError, match="does not list under"):
        load_dataset(tmp_path)


async def test_a_seed_carries_only_what_selection_picked(dataset, store, corpus):
    """
    No bundle arrives with it. The model names one criterion; the seed carries that
    criterion and nothing else, however many sit beside it in the same section.
    """
    runner = PromptRunner(MockBuilderClient(criteria=["SEQ1"]))
    entries = build_index(corpus)
    await pipeline.build_one(entries[0], store, runner, dataset, corpus / "metadata")

    record = store.load(entries[0].sample_id)
    assert [c["id"] for c in record.data("select")["criteria"]] == ["SEQ1"]

    sibling = next(
        c.id for s in dataset.rubrics.sections if s.key == dataset.rubrics.section_of("SEQ1")
        for c in s.criteria if c.id != "SEQ1"
    )
    assert sibling not in {c["id"] for c in record.judged_criteria()}


def test_the_coverage_report_separates_supply_from_use(dataset):
    """
    Two different holes, and they want different fixes: a tag no criterion carries
    means the library cannot ask at all; a tag nothing selected means it can and did
    not.
    """
    cov = report.coverage(dataset, [])
    assert cov["unsupplied"] == [], "every declared tag has at least one criterion"
    # No records, so nothing can have been used.
    assert set(cov["unused"]) == set(dataset.rubrics.tag_vocabulary)
    assert cov["untagged"] == []
