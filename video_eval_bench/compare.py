"""
Compare benchmark runs.

    veb-compare runs/*/report.json
    veb-compare runs/sweep_*/*/report.json --metric pass_rate --format md
    veb-compare runs/*/report.json --csv ablation.csv

Reads report files and nothing else — it never imports the generator or the
judge, so reports from a sweep and reports from runs made a week apart compare
the same way.

The label columns are chosen by diffing the runs' group choices, so a four-run
ablation over one axis shows one label column rather than six identical ones.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# metric key -> (column header, how to read it out of a summary block)
METRICS = {
    "mean_score": ("score", lambda s: s.get("mean_score")),
    "pass_rate": ("pass%", lambda s: _pct(s.get("n_passed"), s.get("n_seeds"))),
    "safety_vetoes": ("vetoes", lambda s: s.get("n_safety_vetoes")),
    "gen_errors": ("gen-err", lambda s: s.get("n_generation_errors")),
    "skipped": ("skip", lambda s: s.get("n_skipped")),
}


class Report:
    """One run's report, reduced to what a comparison needs."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.summary = data.get("summary", {})
        self.choices: Dict[str, str] = data.get("choices", {}) or {}
        self.variant: str = data.get("variant", "") or self.summary.get("variant", "")
        self.note: str = data.get("note", "") or ""
        self.run_id: str = self.summary.get("run_id", path.parent.name)

    @property
    def categories(self) -> Dict[str, dict]:
        return self.summary.get("per_category", {}) or {}

    def label(self, axes: List[str]) -> List[str]:
        """The values of the axes that vary across the comparison set."""
        if not axes:
            return [self.variant or self.run_id]
        return [self.choices.get(axis, "-") for axis in axes]


def load_reports(paths: List[str]) -> List[Report]:
    reports = []
    for raw in paths:
        path = Path(raw)
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        if "summary" not in data:
            print(f"skipping {path}: not a benchmark report", file=sys.stderr)
            continue
        reports.append(Report(path, data))
    return reports


def varying_axes(reports: List[Report]) -> List[str]:
    """
    The group choices that actually differ across these runs.

    Holding every axis constant except one is the normal shape of an ablation,
    so showing the constant ones would be six columns of noise.
    """
    keys = sorted({key for r in reports for key in r.choices})
    return [key for key in keys if len({r.choices.get(key) for r in reports}) > 1]


def build_table(reports: List[Report], metric: str) -> tuple:
    """(header, rows) — label columns, overall, then one column per category."""
    header_name, read = METRICS[metric]
    axes = varying_axes(reports)
    categories = sorted({c for r in reports for c in r.categories})

    header = [
        *(axes or ["variant"]), header_name, *categories,
        "pass", "veto", "err", "skip", "runtime",
    ]
    rows = []
    for report in reports:
        summary = report.summary
        row = [*report.label(axes), _fmt(read(summary))]
        for category in categories:
            block = report.categories.get(category)
            row.append(_fmt(read(block)) if block else "-")
        row += [
            f"{summary.get('n_passed', 0)}/{summary.get('n_seeds', 0)}",
            str(summary.get("n_safety_vetoes", 0)),
            str(summary.get("n_generation_errors", 0)),
            # Older reports predate the field; "-" is honest, 0 would not be.
            _fmt(summary.get("n_skipped")),
            _duration(summary.get("total_duration_seconds")),
        ]
        rows.append(row)
    return header, rows


def render_table(header: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(lines)


def render_markdown(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _pct(part, whole) -> Optional[float]:
    if not whole:
        return None
    return round(100.0 * (part or 0) / whole, 1)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _duration(seconds) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare video-eval-bench runs from their report files."
    )
    parser.add_argument("reports", nargs="+", help="report.json files")
    parser.add_argument(
        "--metric", choices=sorted(METRICS), default="mean_score",
        help="Value shown in the score columns (default: mean_score)",
    )
    parser.add_argument("--format", choices=["table", "md"], default="table")
    parser.add_argument("--csv", default=None, help="Also write the table to this CSV file")
    parser.add_argument(
        "--sort", action="store_true", help="Sort rows by the metric, best first"
    )
    args = parser.parse_args(argv)

    reports = load_reports(args.reports)
    if not reports:
        print("no readable reports", file=sys.stderr)
        return 1

    if args.sort:
        _, read = METRICS[args.metric]
        reports.sort(key=lambda r: (read(r.summary) is None, -(read(r.summary) or 0)))

    header, rows = build_table(reports, args.metric)
    print(render_markdown(header, rows) if args.format == "md" else render_table(header, rows))

    if args.csv:
        write_csv(Path(args.csv), header, rows)
        print(f"\nCSV: {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
