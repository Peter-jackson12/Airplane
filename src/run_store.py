"""Schema/experiment-checked, atomic phase checkpoints (single writer)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path: Path, fields: list[str], run_id: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"CSV schema mismatch: {path}; use a new output prefix")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row: {path}")
            if row["run_id"] != run_id:
                raise ValueError(f"Experiment mismatch: {path}; use a new output prefix")
            if "run_metadata" in row:
                if digest(json.loads(row["run_metadata"])) != run_id:
                    raise ValueError(f"Experiment metadata mismatch: {path}")
            if row["phase_key"] in rows:
                raise ValueError(f"Duplicate phase: {row['phase_key']}")
            rows[row["phase_key"]] = row
    return rows


def upsert_row(path: Path, fields: list[str], run_id: str, row: dict) -> None:
    """Validate before writing; force reruns replace one phase, not its header."""
    if set(row) != set(fields) or row.get("run_id") != run_id:
        raise ValueError("Row schema or experiment mismatch")
    rows = read_rows(path, fields, run_id)
    rows[row["phase_key"]] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows.values())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
