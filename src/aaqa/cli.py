"""Command-line interface: `aaqa <step>` or `python -m aaqa <step>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from aaqa.pipeline import STEPS, run

ALL = ["fetch", "process", "calibrate", "report", "train", "forecast"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aaqa", description=__doc__)
    parser.add_argument("steps", nargs="+", choices=[*STEPS, "all"],
                        help="step(s) to run, in order; 'all' = full pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    steps = ALL if "all" in args.steps else args.steps
    result = run(steps)
    if steps[-1] == "forecast":
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
