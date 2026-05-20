from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class LatencyLogger:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8", newline="")
        self._writer: csv.DictWriter[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> LatencyLogger:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
