"""Schema/experiment-checked, atomic phase checkpoints (single writer)."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
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


def save_oof(rows, directory: Path, phase_key: str) -> tuple[Path, str]:
    """Content-addressed artifact first, summary commit second; orphan files are safe."""
    if not phase_key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in phase_key):
        raise ValueError("Invalid phase key")
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=phase_key + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as writer:
                    rows.to_csv(writer, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        sha = file_digest(Path(name))
        path = directory / f"{phase_key}-{sha}.csv.gz"
        os.replace(name, path)
        return path, sha
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_oof(summary_path: Path, row: dict):
    """Verify checkpoint artifact, experiment, schema, fold settings, and coverage."""
    import numpy as np
    import pandas as pd
    from .oof import OOF_SCHEMA_VERSION, validate_oof_rows

    meta = json.loads(row["run_metadata"])
    if digest(meta) != row["run_id"] or meta.get("oof_schema_version") != OOF_SCHEMA_VERSION:
        raise ValueError("OOF experiment/schema mismatch")
    directory = summary_path.parent / f"{summary_path.stem}_oof"
    expected_name = f"{row['phase_key']}-{row['oof_sha256']}.csv.gz"
    path = summary_path.parent / row["oof_path"]
    if path.resolve() != (directory / expected_name).resolve():
        raise ValueError("OOF path mismatch")
    if not path.is_file() or file_digest(path) != row["oof_sha256"]:
        raise ValueError("OOF artifact missing or hash mismatch; use a new output prefix")
    try:
        rows = pd.read_csv(path, dtype={"ID": "str"}, float_precision="round_trip")
        if list(rows.columns) != json.loads(row["oof_columns"]):
            raise ValueError("OOF column schema mismatch")
        validate_oof_rows(rows, run_id=row["run_id"], phase_key=row["phase_key"],
                          seed=meta["cv"]["seed"], n_rows=int(row["n_rows"]),
                          n_splits=meta["cv"]["n_splits"])
        per_fold = rows.groupby("fold")[["threshold", "n_estimators"]].first()
        if not np.array_equal(per_fold.threshold, json.loads(row["per_fold_thresholds"])) or not np.array_equal(
            per_fold.n_estimators, json.loads(row["selected_n_estimators"])
        ):
            raise ValueError("OOF fold settings differ from summary")
    except (KeyError, TypeError, pd.errors.ParserError) as exc:
        raise ValueError("Malformed OOF artifact") from exc
    return rows
