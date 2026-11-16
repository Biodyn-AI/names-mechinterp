"""Run the fixed sequence only; never provision resources or choose experiments."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parent
    runner = directory / "confirmation_run.py"

    # A failed stage stops the sequence. Partial observations never select a new
    # component, replacement world, relaxed test, or additional compute session.
    stages = [
        ["check"],
        ["behavior", "--model", "qwen", "--deadline", arguments.deadline],
        ["audit", "--model", "qwen"],
        ["behavior", "--model", "phi", "--deadline", arguments.deadline],
        ["audit", "--model", "phi"],
        ["repair", "--deadline", arguments.deadline],
        ["audit-repair"],
    ]
    for stage in stages:
        subprocess.run(
            [sys.executable, str(runner), *stage, "--output", str(arguments.output)],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(directory / "literal_audit.py"),
            "--study-root",
            str(arguments.output),
            "--repo-root",
            str(directory.parents[1]),
            "--output",
            str(arguments.output / "LITERAL_AUDIT.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
