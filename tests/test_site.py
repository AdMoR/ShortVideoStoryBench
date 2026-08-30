"""
The published site: it builds, it says true things, and its links resolve.

CI deploys whatever this produces without a human looking at it, so the tests worth
having are the ones a broken deploy would trip: a page that fails to render at all, a
number invented from nothing, a mock run quietly presented as a result, and the
cross-page links between the performance table and the atlas.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from video_eval_bench.dataset import load_dataset
from video_eval_bench.report import atlas, site

VOID = {"br", "img", "link", "meta", "input", "hr", "base", "source"}


class _Nesting(HTMLParser):
    """Enough of a parser to catch an unclosed <div> in an f-string."""

    def __init__(self):
        super().__init__()
        self.stack, self.bad = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.bad.append(tag)


def _report(run_id: str, variant: str, choices: dict, score, seeds) -> dict:
    return {
        "variant": variant,
        "choices": choices,
        "started_at": f"2026-08-{run_id[-2:]}T00:00:00+00:00",
        "summary": {
            "run_id": run_id,
            "n_seeds": len(seeds),
            "n_ok": len(seeds),
            "mean_score": score,
            "n_passed": 0,
            "n_generation_errors": 0,
            "n_judge_errors": 0,
            "n_safety_vetoes": 0,
            "total_duration_seconds": 60,
            "per_category": {},
        },
        "results": [
            {
                "seed_id": sid,
                "category": "entertainment",
                "status": "ok",
                "verdict": {
                    "total_score": score,
                    "passed": False,
                    "dimensions": [],
                    "critical_failures": [],
                    "scores": scores,
                },
            }
            for sid, scores in seeds.items()
        ],
    }


def _write_reports(tmp_path) -> list:
    """One real run naming a superseded criterion, and one mock run to be dropped."""
    real = _report(
        "20260828-000001",
        "generator=pi,judge=video",
        {"generator": "pi", "judge": "video"},
        42.5,
        {
            "entertainment_001": [
                {"criterion": "SUBJ1", "passed": False, "comment": "she changes"},
                {"criterion": "PROP1", "passed": True, "comment": "letter holds"},
                {"criterion": "GEO1", "passed": True, "comment": "no landmark"},
            ]
        },
    )
    mock = _report(
        "20260829-000002",
        "generator=mock,judge=mock",
        {"generator": "mock", "judge": "mock"},
        100.0,
        {"entertainment_001": [{"criterion": "SUBJ1", "passed": True, "comment": "ok"}]},
    )
    paths = []
    for data in (real, mock):
        path = tmp_path / data["summary"]["run_id"] / "report.json"
        path.parent.mkdir()
        path.write_text(json.dumps(data))
        paths.append(path)
    return paths


@pytest.fixture
def runs_file(tmp_path) -> Path:
    """The published snapshot those reports export to."""
    out = tmp_path / "runs.json"
    out.write_text(json.dumps(site.snapshot(_write_reports(tmp_path))))
    return out


def test_mock_runs_are_not_published(runs_file):
    """
    The mock judge passes everything. Publishing one would put a 100.0 at the top of
    the table and mean nothing by it — the same lie the fixed rubric used to tell.
    """
    data = json.loads(runs_file.read_text())
    assert [r["run_id"] for r in data["runs"]] == ["20260828-000001"]


def test_include_mock_is_available_for_a_harness_check(tmp_path):
    reports = _write_reports(tmp_path)
    assert len(site.snapshot(reports, include_mock=True)["runs"]) == 2


def test_the_headline_run_must_have_covered_the_dataset():
    """A 100 on one seed is a smoke test, not a result."""
    one = {"run_id": "small", "summary": {"mean_score": 100.0, "n_ok": 1}, "seeds": []}
    many = {"run_id": "broad", "summary": {"mean_score": 40.0, "n_ok": 12}, "seeds": []}
    assert site.headline([one, many])["run_id"] == "broad"
    # ...but a corpus of only small runs still gets a headline rather than none.
    assert site.headline([one])["run_id"] == "small"


def test_site_builds_three_well_formed_pages(tmp_path, runs_file):
    pages = site.build(tmp_path / "out", data_file=runs_file, pilot=Path("/nonexistent"))
    assert [p.name for p in pages] == ["index.html", "atlas.html", "performance.html"]
    for page in pages:
        text = page.read_text()
        assert text.startswith("<title>")
        parser = _Nesting()
        parser.feed(text)
        assert not parser.bad and not parser.stack, page.name
    # Jekyll would eat a leading-underscore path; the site is already static.
    assert (tmp_path / "out" / ".nojekyll").exists()


def test_every_tab_link_resolves(tmp_path, runs_file):
    pages = site.build(tmp_path / "out", data_file=runs_file, pilot=Path("/nonexistent"))
    names = {p.name for p in pages}
    for href, _ in site.TABS:
        assert href in names
        for page in pages:
            assert f'href="{href}"' in page.read_text(), page.name


def test_performance_maps_superseded_ids_onto_the_library(tmp_path, runs_file):
    """
    A run older than the rubric review names PROP1 and GEO1. PROP1's verdict belongs
    to SEC1, which absorbed it; GEO1 has no successor and must be dropped rather than
    counted as something else.
    """
    site.build(tmp_path / "out", data_file=runs_file, pilot=Path("/nonexistent"))
    page = (tmp_path / "out" / "performance.html").read_text()
    assert 'href="atlas.html#c-SEC1"' in page
    # The mapping is stated in the page's own words, so PROP1 appears in the note —
    # what must not appear is a row of its own.
    assert "#c-PROP1" not in page
    assert "#c-GEO1" not in page

    lib = load_dataset().rubrics
    assert site._live("PROP1", lib) == "SEC1"
    assert site._live("GEO1", lib) is None
    assert site._live("SUBJ1", lib) == "SUBJ1"


def test_criterion_anchors_exist_in_the_atlas(tmp_path, runs_file):
    """The performance page deep-links to them, so they have to be there."""
    site.build(tmp_path / "out", data_file=runs_file, pilot=Path("/nonexistent"))
    page = (tmp_path / "out" / "atlas.html").read_text()
    lib = load_dataset().rubrics
    for criterion in lib.criteria:
        assert f'id="c-{criterion.id}"' in page


def test_a_site_with_no_published_runs_still_builds(tmp_path):
    """The first push happens before anyone has exported a run."""
    pages = site.build(
        tmp_path / "out", data_file=tmp_path / "absent.json", pilot=Path("/nonexistent")
    )
    performance = (tmp_path / "out" / "performance.html").read_text()
    assert "No run has been published yet" in performance
    assert len(pages) == 3


def test_the_atlas_names_the_corpus_it_counted(tmp_path, runs_file):
    """
    Coverage counts are meaningless without saying what they are over, and the answer
    differs between a local build (the pilot) and CI (the dataset's own seeds).
    """
    site.build(tmp_path / "out", data_file=runs_file, pilot=Path("/nonexistent"))
    assert "the benchmark's dataset" in (tmp_path / "out" / "atlas.html").read_text()

    lib = load_dataset().rubrics
    standalone = atlas.render(lib)
    assert "<nav class=\"tabs\">" not in standalone


# ── the worked example ────────────────────────────────────────────────────────


def _example(seeds) -> dict:
    """An exported example bundle, of the shape `site.build` expects to find."""
    return {
        "run_id": "20260825-072222",
        "variant": "generator=pi,judge=pi",
        "started_at": "2026-08-25T07:22:22+00:00",
        "note": "",
        "params": {
            "choices": {"system_prompt": "director"},
            "generator": {
                "agent": "pi", "model": "qwen", "provider": "gx10",
                "system_prompt": "director", "skills": [], "tools": ["bash"],
                "sandbox": "none", "offline": True, "timeout_seconds": 7200,
            },
            "judge": {
                "backend": "pi", "model": "qwen", "provider": "gx10",
                "n_frames": 8, "timeout_seconds": 360,
            },
        },
        "summary": {
            "n_seeds": 8, "n_ok": 7, "n_passed": 6, "mean_score": 64.31,
            "n_generation_errors": 1, "total_duration_seconds": 59870.0,
        },
        "cost": {"turns": 450, "tokens": 542376, "tool_calls": {"bash": 400}},
        "seeds": seeds,
    }


def _example_seed(seed_id="educational_001", criterion="SUBJ1") -> dict:
    return {
        "seed_id": seed_id,
        "category": "educational",
        "brief": "An explainer video showing how a car engine works.",
        "score": 77.04,
        "passed": True,
        "media": {
            "video": f"{seed_id}.mp4", "poster": f"{seed_id}.jpg",
            "bytes": 51761, "source_bytes": 214159,
            "width": 640, "height": 360, "duration": 5.0, "has_audio": False,
        },
        "generation": {
            "turns": 56, "tokens": 75988, "tool_calls": {"bash": 44},
            "duration_seconds": 5700.0, "outcome": "exited",
            "submit_notes": "Final 5s explainer.",
        },
        "scores": [
            {"criterion": criterion, "passed": False, "comment": "she changes"}
        ],
        "safety_violations": [],
        "trace": [
            {"kind": "usage", "turn": 1, "tokens": 2750},
            {"kind": "thinking", "turn": 1, "text": "Let me look around."},
            {"kind": "tool", "turn": 1, "name": "bash", "head": "ls -la",
             "args": '{"command": "ls -la"}', "output": "workspace/", "error": False},
        ],
    }


@pytest.fixture
def example_file(tmp_path) -> Path:
    out = tmp_path / "example.json"
    out.write_text(json.dumps(_example([_example_seed()])))
    return out


def _build_with_example(tmp_path, runs_file, example_file, media=None):
    return site.build(
        tmp_path / "out",
        data_file=runs_file,
        pilot=Path("/nonexistent"),
        example_file=example_file,
        media_dir=media or (tmp_path / "media"),
    )


def test_the_example_renders_its_clips_traces_and_verdicts(
    tmp_path, runs_file, example_file
):
    _build_with_example(tmp_path, runs_file, example_file)
    page = (tmp_path / "out" / "performance.html").read_text()
    assert 'src="media/educational_001.mp4"' in page
    assert 'poster="media/educational_001.jpg"' in page
    # The clip must not be fetched until someone asks for it: three autoloading
    # videos is most of the page's weight spent before anyone presses play.
    assert 'preload="none"' in page
    assert "An explainer video showing how a car engine works." in page
    assert "Let me look around." in page          # the trace
    assert "she changes" in page                  # the judge's reason
    assert "director" in page                     # the arm


def test_the_example_is_collapsed_and_stays_well_formed(
    tmp_path, runs_file, example_file
):
    """
    Three videos and three traces at the top of the page would bury the tables. The
    seed, its trace and its verdict are all <details>, and none of them opens itself.
    """
    _build_with_example(tmp_path, runs_file, example_file)
    page = (tmp_path / "out" / "performance.html").read_text()
    start = page.index('id="example"')
    end = page.index('id="runs"')
    section = page[start:end]
    assert "<details" in section and "<details open" not in section
    parser = _Nesting()
    parser.feed(page)
    assert not parser.bad and not parser.stack


def test_the_example_media_is_copied_into_the_built_site(
    tmp_path, runs_file, example_file
):
    """A <video src> pointing outside the deploy artifact resolves to nothing."""
    media = tmp_path / "media"
    media.mkdir()
    (media / "educational_001.mp4").write_bytes(b"not really an mp4")
    (media / "educational_001.jpg").write_bytes(b"not really a jpeg")
    _build_with_example(tmp_path, runs_file, example_file, media=media)
    assert (tmp_path / "out" / "media" / "educational_001.mp4").exists()
    assert (tmp_path / "out" / "media" / "educational_001.jpg").exists()


def test_a_live_criterion_links_into_the_atlas(tmp_path, runs_file, example_file):
    _build_with_example(tmp_path, runs_file, example_file)
    page = (tmp_path / "out" / "performance.html").read_text()
    assert 'href="atlas.html#c-SUBJ1"' in page


def test_a_retired_criterion_says_so_instead_of_linking_nowhere(tmp_path, runs_file):
    """
    Runs older than the rubric review were judged on per-category rubrics whose ids
    the library never had. Rendering those as a named criterion with a dead atlas
    link would read as a bug; the page says what they are.
    """
    example_file = tmp_path / "example.json"
    example_file.write_text(json.dumps(_example([_example_seed(criterion="ED4")])))
    _build_with_example(tmp_path, runs_file, example_file)
    page = (tmp_path / "out" / "performance.html").read_text()
    assert "#c-ED4" not in page
    assert "predates the review" in page


def test_the_page_builds_without_an_example(tmp_path, runs_file):
    """The example is optional — the first deploy happened before there was one."""
    site.build(
        tmp_path / "out",
        data_file=runs_file,
        pilot=Path("/nonexistent"),
        example_file=tmp_path / "absent.json",
    )
    page = (tmp_path / "out" / "performance.html").read_text()
    assert 'id="example"' not in page
    assert 'id="runs"' in page
