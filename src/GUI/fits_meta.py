"""
fits_meta.py — Asynchronous FITS header metadata reader.

Reads NAXIS1 (width), NAXIS2 (height), and the optional BAYERPAT keyword
from a FITS file using a minimal pure-Python parser — no external libraries
required. Only the primary HDU header is read; pixel data is never loaded.

Runs in a QThreadPool worker so the UI thread is never blocked.

FITS header format (FITS Standard 4.0 §4):
  - Cards are 80 bytes each: "KEYWORD = VALUE / comment"
  - Packed 36 cards per 2880-byte block
  - Header ends at the first card whose keyword is "END"
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

_BLOCK = 2880
_CARD  = 80


def _read_fits_keywords(path: str, want: set[str]) -> dict[str, str]:
    """Return a dict of {keyword: raw_value_string} for the requested keys.

    Reads only the primary HDU header blocks; stops at END or EOF.
    Returns an empty dict on any I/O or format error.
    """
    found: dict[str, str] = {}
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(_BLOCK)
                if len(block) < _BLOCK:
                    break
                for i in range(36):
                    card = block[i * _CARD:(i + 1) * _CARD].decode("ascii", errors="replace")
                    key = card[:8].rstrip()
                    if key == "END":
                        return found
                    if key in want and "=" in card[8:10]:
                        # Value field is card[10:80]; strip whitespace and quotes
                        raw = card[10:80].split("/")[0].strip().strip("'").strip()
                        found[key] = raw
                        if len(found) == len(want):
                            return found
    except Exception:
        pass
    return found


class _MetaSignals(QObject):
    """Carrier object for signals, since QRunnable doesn't inherit QObject."""
    result = pyqtSignal(int, dict)   # (row, meta_dict)


class FitsMetaWorker(QRunnable):
    """Read FITS primary HDU header in a thread-pool worker.

    On completion emits ``signals.result(row, meta)`` where *meta* is::

        {"width": int|None, "height": int|None, "bayer": str|None}

    The signal is queued automatically because the receiver lives on
    the main thread (Qt's default cross-thread connection type).
    """

    def __init__(self, row: int, path: str) -> None:
        super().__init__()
        self.row = row
        self.path = path
        self.signals = _MetaSignals()

    @pyqtSlot()
    def run(self) -> None:
        meta: dict = {"width": None, "height": None, "bayer": None}
        kw = _read_fits_keywords(self.path, {"NAXIS1", "NAXIS2", "BAYERPAT"})
        try:
            meta["width"]  = int(kw["NAXIS1"])
        except (KeyError, ValueError):
            pass
        try:
            meta["height"] = int(kw["NAXIS2"])
        except (KeyError, ValueError):
            pass
        if "BAYERPAT" in kw:
            meta["bayer"] = kw["BAYERPAT"].upper()
        self.signals.result.emit(self.row, meta)
