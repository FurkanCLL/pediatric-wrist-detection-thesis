"""Read-only inputs and private/public boundaries for Phase 3B."""

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

from wrist_train.common import configure_runtime

configure_runtime()
from wrist_prepare.common import CLASSES
from wrist_prepare.records import load_records
from wrist_train.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/error_analysis_sahi"
PUBLIC = ROOT / "docs/evidence/error_analysis_sahi"
BASELINE = ROOT / "outputs/training/standard_baseline"
BEST = BASELINE / "weights/best.pt"
BEST_SHA = "31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b"
PREPARED = ROOT / "data/prepared/grazpedwri_detection"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def config():
    return json.loads((ROOT / "configs/inference_feasibility.json").read_text(encoding="utf-8"))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validation_records():
    # Metadata containers are hash-verified, then filtered before any image/label access.
    rows = sorted((r for r in load_records(ROOT / "outputs/dataset_preparation") if r["split"] == "val"),
                  key=lambda r: r["filestem"])
    if len(rows) != 3088 or len({r["patient_id"] for r in rows}) != 914:
        raise ValueError("Frozen validation membership changed")
    return rows


def image_path(row):
    if row["split"] != "val" or Path(row["filestem"]).name != row["filestem"]:
        raise ValueError("Only frozen validation images may be opened")
    path = PREPARED / "images/val" / (row["filestem"] + ".png")
    if path.resolve().parent != (PREPARED / "images/val").resolve():
        raise ValueError("Image escaped the validation directory")
    return path


def baseline_inventory():
    return {p.relative_to(BASELINE).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(BASELINE.rglob("*")) if p.is_file()}


def verify_baseline():
    # Reconcile all artifacts against the previous preservation receipt before loading any model.
    import torch
    from wrist_train.checkpoints import checked_bytes, check_payload
    protocol, lock = load_protocol()
    inventory = baseline_inventory()
    previous = json.loads((ROOT / "outputs/class_aware_design/baseline_inventory.json").read_text())
    if inventory != previous or sha(BEST) != BEST_SHA:
        raise ValueError("Original baseline artifacts changed")
    checkpoints = []
    for path in sorted((BASELINE / "weights").glob("*.pt")):
        payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
        check_payload(payload, lock, require_resume=False)
        checkpoints.append({"file": path.name, "completed_epoch": payload["epoch"] + 1, "sha256": sha(path)})
        if path == BEST:
            if payload["epoch"] != 21:
                raise ValueError("Wrong selected epoch")
            selected = {"metrics": payload["train_metrics"], "classes": payload["per_class"]}
        del payload
    result = {"checkpoint_sha256": BEST_SHA, "checkpoints": checkpoints, "artifact_count": len(inventory),
              "inventory_sha256": identity(inventory), "protocol_sha256": lock["protocol_sha256"],
              "selected_epoch": 22, "training_epochs": 80, "training_completion": json.loads((BASELINE / "completed.json").read_text()),
              "unchanged": True, **selected}
    save(LOCAL / "baseline_inventory_before.json", inventory)
    save(PUBLIC / "baseline_verification.json", result)
    return result


def load_model():
    import torch
    from wrist_train.checkpoints import checked_bytes, check_payload
    _, lock = load_protocol()
    if sha(BEST) != BEST_SHA:
        raise ValueError("Wrong checkpoint identity")
    payload = torch.load(io.BytesIO(checked_bytes(BEST)), map_location="cpu", weights_only=False)
    check_payload(payload, lock, require_resume=False)
    # The native training validator evaluated EMA, without Conv/BN fusion or FP16 weight conversion.
    model = payload["ema"].float().eval().to("cuda:0")
    model.requires_grad_(False)
    if model.names != dict(enumerate(CLASSES)):
        raise ValueError("Unexpected class IDs")
    return model


def verify_validation_files(rows):
    # Read the conversion journal without allowing SQLite to create side files.
    database = ROOT / "outputs/dataset_preparation/completed.sqlite"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        hashes = {s: (im, lab) for s, im, lab in connection.execute("select stem, image_sha, label_sha from completed")}
    for row in rows:
        image = image_path(row)
        label = PREPARED / "labels/val" / (row["filestem"] + ".txt")
        if (sha(image), sha(label)) != hashes[row["filestem"]]:
            raise ValueError("Prepared validation content differs from frozen conversion")
    return {"images": len(rows), "labels": len(rows), "all_sha256_match": True}


def source_snapshot():
    # Stat-only inventory detects accidental rewrites without opening held-out image contents.
    paths = (ROOT / "data/raw", PREPARED)
    result={}
    for base in paths:
        for directory,_,_ in os.walk(base):
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        stat=entry.stat()
                        result[Path(entry.path).relative_to(ROOT).as_posix()]=(stat.st_size,stat.st_mtime_ns)
    return result
