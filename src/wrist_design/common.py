"""Isolated design inputs and aggregate outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from wrist_prepare.common import CLASSES
from wrist_prepare.records import load_records

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/class_aware_design"
PUBLIC = ROOT / "docs/evidence/class_aware_design"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def training_records():
    # Verify the existing frozen metadata, then discard holdout rows before any analysis.
    rows = load_records(ROOT / "outputs/dataset_preparation")
    rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["filestem"])
    if len(rows) != 14061 or len({r["patient_id"] for r in rows}) != 4264:
        raise ValueError("Unexpected frozen training population")
    return rows


def arrays(rows):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Policy analysis accepts training records only")
    counts = np.zeros((len(rows), 8), dtype=np.int32)
    patients = sorted({r["patient_id"] for r in rows})
    index = {patient: i for i, patient in enumerate(patients)}
    for i, row in enumerate(rows):
        for box in row["boxes"]:
            counts[i, box["class_id"]] += 1
    return counts, np.array([index[r["patient_id"]] for r in rows]), patients


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max()),
            "mean": float(values.mean())}
