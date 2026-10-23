import json
from pathlib import Path
from types import SimpleNamespace

import bootstrap


def test_free_threaded_build_flag_is_detected(monkeypatch):
    monkeypatch.setattr(
        bootstrap.sysconfig,
        "get_config_var",
        lambda name: 1 if name == "Py_GIL_DISABLED" else None,
    )
    assert bootstrap._is_free_threaded_python()


def test_runtime_gil_state_is_used_when_build_flag_is_missing(monkeypatch):
    monkeypatch.setattr(bootstrap.sysconfig, "get_config_var", lambda name: None)
    monkeypatch.setattr(bootstrap.sys, "_is_gil_enabled", lambda: False, raising=False)
    assert bootstrap._is_free_threaded_python()


def test_python_probe_parses_interpreter_metadata(monkeypatch):
    expected = {
        "executable": "/usr/bin/python3.12",
        "free_threaded": False,
        "version": [3, 12, 8],
    }
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="launcher banner\n" + json.dumps(expected) + "\n", stderr=""
        ),
    )
    assert bootstrap._probe_python(["python"]) == expected


def test_standard_python_discovery_skips_free_threaded_build(monkeypatch):
    paths = {"python": "/fake/python", "python3": "/fake/python3"}
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: paths.get(name))

    def probe(command):
        if command[0] == "/fake/python":
            return {
                "executable": "/fake/python",
                "free_threaded": True,
                "version": [3, 13, 0],
            }
        if command[0] == "/fake/python3":
            return {
                "executable": "/fake/python3",
                "free_threaded": False,
                "version": [3, 12, 7],
            }
        return None

    monkeypatch.setattr(bootstrap, "_probe_python", probe)
    assert bootstrap._find_standard_python() == Path("/fake/python3")
