"""Opt-in archived CPU unit tests, isolated per original directory and offline."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = """
import socket, sys
def blocked(*args, **kwargs):
    raise RuntimeError('Network access is forbidden in archived CPU tests')
socket.create_connection = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider']))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "archived-tests.json")
    args = parser.parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    reports = []
    directories = sorted(p for p in (ROOT / "experiments").iterdir() if p.is_dir() and list(p.glob("test_*.py")))
    for directory in directories:
        completed = subprocess.run([args.python, "-c", BOOTSTRAP], cwd=directory, env=env, text=True, capture_output=True, timeout=240)
        report = {"directory": directory.relative_to(ROOT).as_posix(), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        reports.append(report)
        print(json.dumps({"directory": report["directory"], "returncode": report["returncode"], "summary": completed.stdout.strip().splitlines()[-1:] or completed.stderr.strip().splitlines()[-1:]}), flush=True)
    result = {"schema": "archived-offline-cpu-tests-v1", "python": args.python, "network_blocked": True, "gpu_visible": False, "all_passed": all(r["returncode"] == 0 for r in reports), "directories": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
