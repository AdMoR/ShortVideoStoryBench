"""
veb-compare: turn a pile of report files into one comparative table.

It reads reports and nothing else, so these tests build reports as plain dicts —
which is also the guarantee that runs made days apart still compare.
"""

import csv
import json

from video_eval_bench.compare import (
    build_table,
    load_reports,
    main,
    render_markdown,
    varying_axes,
)


def write_report(tmp_path, name, choices, score, *, passed=1, seeds=1, errors=0, vetoes=0):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "variant": ",".join(f"{k}={v}" for k, v in choices.items()),
                "choices": choices,
                "summary": {
                    "run_id": name.split("/")[0],
                    "n_seeds": seeds,
                    "n_ok": seeds - errors,
                    "n_generation_errors": errors,
                    "n_passed": passed,
                    "n_safety_vetoes": vetoes,
                    "mean_score": score,
                    "total_duration_seconds": 90,
                    "per_category": {
                        "marketing": {
                            "n_seeds": seeds,
                            "mean_score": score,
                            "n_passed": passed,
                            "n_safety_vetoes": vetoes,
                        }
                    },
                },
                "results": [],
            }
        )
    )
    return str(path)


def ablation(tmp_path):
    """Three runs differing on one axis only — the normal ablation shape."""
    common = {"model": "gx10_qwen27b_q8", "tools": "full", "judge": "pi"}
    return [
        write_report(tmp_path, "a/report.json", {**common, "skills": "none"}, 41.2),
        write_report(tmp_path, "b/report.json", {**common, "skills": "video_basic"}, 58.7),
        write_report(tmp_path, "c/report.json", {**common, "skills": "e2e_mock"}, 33.4),
    ]


def test_only_the_axis_that_varies_becomes_a_column(tmp_path):
    reports = load_reports(ablation(tmp_path))
    assert varying_axes(reports) == ["skills"]

    header, rows = build_table(reports, "mean_score")
    assert header[0] == "skills"
    assert "model" not in header  # constant across the set: noise
    assert [row[0] for row in rows] == ["none", "video_basic", "e2e_mock"]


def test_scores_land_in_the_table(tmp_path):
    header, rows = build_table(load_reports(ablation(tmp_path)), "mean_score")
    assert rows[1][header.index("score")] == "58.7"
    assert rows[1][header.index("marketing")] == "58.7"


def test_pass_rate_metric(tmp_path):
    paths = [
        write_report(tmp_path, "a/report.json", {"skills": "none"}, 40.0, passed=1, seeds=4),
        write_report(tmp_path, "b/report.json", {"skills": "video_basic"}, 60.0, passed=3, seeds=4),
    ]
    header, rows = build_table(load_reports(paths), "pass_rate")
    assert rows[0][header.index("pass%")] == "25.0"
    assert rows[1][header.index("pass%")] == "75.0"


def test_identical_choices_fall_back_to_the_variant_column(tmp_path):
    paths = [
        write_report(tmp_path, "a/report.json", {"skills": "none"}, 41.0),
        write_report(tmp_path, "b/report.json", {"skills": "none"}, 44.0),
    ]
    header, _ = build_table(load_reports(paths), "mean_score")
    assert header[0] == "variant"


def test_generation_errors_are_visible(tmp_path):
    paths = [write_report(tmp_path, "a/report.json", {"tools": "read_only"}, 0.0, errors=3, seeds=3)]
    header, rows = build_table(load_reports(paths), "mean_score")
    assert rows[0][header.index("err")] == "3"


def test_unreadable_files_are_skipped_not_fatal(tmp_path):
    good = write_report(tmp_path, "a/report.json", {"skills": "none"}, 41.0)
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    assert len(load_reports([good, str(bad), str(tmp_path / "missing.json")])) == 1


def test_markdown_output(tmp_path):
    header, rows = build_table(load_reports(ablation(tmp_path)), "mean_score")
    md = render_markdown(header, rows)
    assert md.startswith("| skills |")
    assert "| ---" in md or "|---" in md


def test_cli_writes_csv_and_sorts(tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert main([*ablation(tmp_path), "--csv", str(out), "--sort"]) == 0

    with open(out) as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "skills"
    assert [r[0] for r in rows[1:]] == ["video_basic", "none", "e2e_mock"]  # best first


def test_cli_reports_when_nothing_is_readable(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
