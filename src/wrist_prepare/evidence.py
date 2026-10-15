"""Publish only aggregate preparation evidence after both validation gates pass."""

import argparse
import csv
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import zlib

from wrist_audit.audit import validate_output_path
from .common import CLASSES, SPLITS, json_bytes, protected_write


def export_evidence(state, destination):
    def read(name):
        return json.loads((state / name).read_text(encoding="utf-8"))
    freeze, stats = read("freeze.json"), read("construction_statistics.json")
    validation, loader = read("validation.json"), read("loader_validation.json")
    if (validation["images_checked"] != freeze["images"] or validation["objects_checked"] != freeze["objects"]
            or not validation["raw_stat_snapshot_unchanged"] or validation["patient_overlap"] or validation["pixel_mapping_mismatches"]):
        raise ValueError("Full preparation validation has not passed")
    for split in SPLITS:
        if loader["splits"][split]["images_scanned"] != stats[split]["images"] or loader["splits"][split]["objects_loaded"] != stats[split]["objects"]:
            raise ValueError("Actual training loader coverage is incomplete")
    total_patients = sum(s["patients"] for s in stats.values())
    for split in SPLITS:
        stats[split]["patient_percent"] = 100 * stats[split]["patients"] / total_patients
        stats[split]["image_percent"] = 100 * stats[split]["images"] / freeze["images"]
    public = {"seed": 42, "target_ratios": [.7, .15, .15], "splits": stats,
              "images": freeze["images"], "patients": freeze["patients"], "objects": freeze["objects"],
              "class_mapping": {i: c for i, c in enumerate(CLASSES)},
              "ready_for_training_feasibility_data_checks": True,
              "gpu_software_feasibility_verified": False,
              "test_use": "Construction distribution and technical preparation checks only; no method tuning or model evaluation"}
    destination.mkdir(parents=True, exist_ok=True)
    protected_write(destination / "aggregate_statistics.json", json_bytes(public))
    protected_write(destination / "frozen_manifest_hashes.json", json_bytes(freeze))
    for name in ("validation.json", "loader_validation.json", "training_statistics.json", "pilot_validation.json", "pilot_review.json"):
        protected_write(destination / name, json_bytes(read(name)))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["split", "class_id", "class", "objects", "images", "patients"])
    for split in SPLITS:
        for c, name in enumerate(CLASSES):
            values = stats[split]["classes"][name]
            writer.writerow([split, c, name, values["objects"], values["images"], values["patients"]])
    protected_write(destination / "class_distribution.csv", buffer.getvalue().encode())
    environment = {"python": platform.python_version(), "zlib": zlib.ZLIB_RUNTIME_VERSION,
                   "packages": {p: importlib.metadata.version(p) for p in ("numpy", "Pillow", "opencv-python", "ultralytics", "torch", "torchvision")},
                   "implementation_commit": subprocess.check_output([
                       "git", "log", "-1", "--format=%H", "--", "src/wrist_prepare", "tests/test_preparation.py",
                       "configs/dataset_preparation.json", "pyproject.toml", "requirements.txt"], text=True).strip()}
    protected_write(destination / "environment.json", json_bytes(environment))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("outputs/dataset_preparation"))
    parser.add_argument("--destination", type=Path, default=Path("docs/evidence/dataset_preparation"))
    args = parser.parse_args()
    validate_output_path(Path("data/raw"), args.destination)
    export_evidence(args.state, args.destination)


if __name__ == "__main__":
    main()
