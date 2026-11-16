"""The detached launcher must follow the frozen sequence and fail closed."""

import subprocess
import sys

import pytest

import execute_registered as launcher


def test_launcher_runs_both_models_then_the_fixed_repair_and_independent_audit(
    monkeypatch, tmp_path
):
    calls = []

    def record(arguments, *, check):
        assert check is True
        calls.append(arguments)

    monkeypatch.setattr(launcher.subprocess, "run", record)
    monkeypatch.setattr(
        sys,
        "argv",
        ["execute_registered.py", "--output", str(tmp_path), "--deadline", "fixed"],
    )
    launcher.main()
    assert [arguments[2] for arguments in calls[:-1]] == [
        "check", "behavior", "audit", "behavior", "audit", "repair", "audit-repair"
    ]
    assert calls[1][4] == "qwen"
    assert calls[3][4] == "phi"
    assert calls[-1][1].endswith("literal_audit.py")
    assert calls[-1][-1] == str(tmp_path / "LITERAL_AUDIT.json")
    assert all(arguments[0] == sys.executable for arguments in calls)


def test_launcher_never_continues_after_a_failed_stage(monkeypatch, tmp_path):
    calls = []

    def fail_behavior(arguments, *, check):
        calls.append(arguments)
        if arguments[2] == "behavior":
            raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(launcher.subprocess, "run", fail_behavior)
    monkeypatch.setattr(
        sys,
        "argv",
        ["execute_registered.py", "--output", str(tmp_path), "--deadline", "fixed"],
    )
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
    assert len(calls) == 2
