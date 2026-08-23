"""
Re-render the HTML report for a run that already exists.

    veb-report runs/20260822-193000
    veb-report runs/*/            # several at once

`veb` writes report.html automatically at the end of every run. This exists for
runs made before the renderer changed, and for iterating on the renderer itself
without paying for a fresh benchmark run.
"""

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render report.html for a run directory")
    parser.add_argument("run_dirs", nargs="+", help="run directories containing report.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    from video_eval_bench.report.html import render_run

    failures = 0
    for raw in args.run_dirs:
        run_dir = Path(raw)
        try:
            print(render_run(run_dir))
        except Exception as exc:
            print(f"{run_dir}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
