import unittest
import importlib
import sys
import types
from unittest.mock import patch


def _load_runner_module():
    qtcore = types.ModuleType("PySide6.QtCore")

    class _DummyQThread:
        def __init__(self, parent=None):
            self.parent = parent

    class _DummySignal:
        def __init__(self, *args, **kwargs):
            pass

    qtcore.QThread = _DummyQThread
    qtcore.Signal = _DummySignal
    pyside6 = types.ModuleType("PySide6")
    pyside6.QtCore = qtcore
    sys.modules.setdefault("PySide6", pyside6)
    sys.modules.setdefault("PySide6.QtCore", qtcore)

    return importlib.import_module("src.GUI.runner")


runner = _load_runner_module()


def _env(path: str, cuda_path: str | None) -> dict[str, str]:
    env = {"PATH": path}
    if cuda_path is not None:
        env["CUDA_PATH"] = cuda_path
    return env


class TestRunnerEnv(unittest.TestCase):
    def test_non_windows_env_unchanged(self):
        base = _env("/usr/bin:/bin", "/opt/cuda")
        with patch.object(runner.platform, "system", return_value="Linux"):
            out = runner._build_subprocess_env(base)
        self.assertEqual(out["PATH"], base["PATH"])

    def test_windows_adds_cuda_bin_to_path(self):
        base = _env(r"C:\Windows\System32", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")
        with patch.object(runner.platform, "system", return_value="Windows"):
            out = runner._build_subprocess_env(base)
        self.assertTrue(out["PATH"].startswith(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;"))

    def test_windows_keeps_existing_cuda_bin_once(self):
        base = _env(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Windows\System32",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
        )
        with patch.object(runner.platform, "system", return_value="Windows"):
            out = runner._build_subprocess_env(base)
        self.assertEqual(out["PATH"], base["PATH"])

    def test_windows_without_cuda_path_unchanged(self):
        base = _env(r"C:\Windows\System32", None)
        with patch.object(runner.platform, "system", return_value="Windows"):
            out = runner._build_subprocess_env(base)
        self.assertEqual(out["PATH"], base["PATH"])


if __name__ == "__main__":
    unittest.main()
