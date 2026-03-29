"""
runner.py — SubprocessRunner

A QThread that launches the dso_stacker binary as a subprocess, streams
stdout+stderr line-by-line to the GUI log panel via signals, and allows
the user to abort mid-run with a clean process termination.

Usage
-----
    runner = SubprocessRunner(argv)
    runner.line_ready.connect(log_panel.appendPlainText)
    runner.finished.connect(on_finished)
    runner.start()
    # later, to cancel:
    runner.abort()
    runner.wait()
"""

from __future__ import annotations

import ntpath
import os
import platform
import subprocess
from typing import Optional

from PySide6.QtCore import QThread, Signal


def _build_subprocess_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Build subprocess environment with Windows CUDA runtime PATH compatibility."""
    env = dict(base_env if base_env is not None else os.environ)

    if platform.system() != "Windows":
        return env

    cuda_path = env.get("CUDA_PATH")
    if not cuda_path:
        return env

    path_sep = ";"
    cuda_bin = ntpath.join(cuda_path, "bin")
    path_value = env.get("PATH", "")
    path_parts = [p for p in path_value.split(path_sep) if p]
    normalized = {ntpath.normcase(ntpath.normpath(p)) for p in path_parts}
    cuda_bin_norm = ntpath.normcase(ntpath.normpath(cuda_bin))

    if cuda_bin_norm not in normalized:
        env["PATH"] = cuda_bin if not path_value else f"{cuda_bin}{path_sep}{path_value}"

    return env


class SubprocessRunner(QThread):
    """Run dso_stacker in a background thread and stream its output."""

    line_ready = Signal(str)   # one output line at a time
    finished   = Signal(int)   # process exit code

    def __init__(self, argv: list[str], parent=None) -> None:
        super().__init__(parent)
        self._argv    = argv
        self._proc: Optional[subprocess.Popen] = None
        self._aborted = False

    # ------------------------------------------------------------------ #
    # QThread entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,          # line-buffered
                env=_build_subprocess_env(),
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self.line_ready.emit(line.rstrip("\n"))
            self._proc.wait()
            code = self._proc.returncode
        except FileNotFoundError as exc:
            self.line_ready.emit(f"ERROR: {exc}")
            code = -1
        except Exception as exc:
            self.line_ready.emit(f"ERROR: {exc}")
            code = -1
        self.finished.emit(code)

    # ------------------------------------------------------------------ #
    # Abort                                                                #
    # ------------------------------------------------------------------ #

    def abort(self) -> None:
        """Request process termination.  Call wait() afterwards."""
        self._aborted = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
