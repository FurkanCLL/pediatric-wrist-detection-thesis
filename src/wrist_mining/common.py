"""Frozen training inputs and explicit private/public boundaries."""

import json
from pathlib import Path
import sqlite3

from wrist_design.common import training_records, arrays
from wrist_eval.common import ROOT, PREPARED, BEST_SHA, identity, sha, save
from wrist_prepare.common import CLASSES

LOCAL = ROOT / "outputs/error_guided_training_feasibility"
PUBLIC = ROOT / "docs/evidence/error_guided_training_feasibility"
CONFIG = ROOT / "configs/error_guided_feasibility.json"


def config():
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value["split"] != "train" or value["official_training_authorized"] or value["heldout_evaluation_authorized"]:
        raise ValueError("This diagnostic accepts training inference only")
    return value


# Reject other partitions before constructing any image or label path.
def training_path(row, kind="images"):
    stem = row["filestem"]
    if row["split"] != "train" or Path(stem).name != stem or kind not in ("images", "labels"):
        raise ValueError("Only frozen training files may be opened")
    base = PREPARED / kind / "train"
    path = base / (stem + (".png" if kind == "images" else ".txt"))
    if path.resolve().parent != base.resolve():
        raise ValueError("Training path escaped its partition")
    return path


def training_hashes():
    database = ROOT / "outputs/dataset_preparation/completed.sqlite"
    with sqlite3.connect(database.as_uri()+"?mode=ro", uri=True) as connection:
        return {stem: (image, label) for stem, image, label in connection.execute("SELECT * FROM completed")}


def stability_subset(rows):
    cfg = config()["stability"]
    ranked = sorted(rows, key=lambda r: identity({"purpose": "phase4a_stability", "seed": 42, "stem": r["filestem"]}))
    selected = {r["filestem"] for r in ranked[:cfg["uniform_images"]]}
    for class_id in cfg["class_ids"]:
        eligible = [r for r in ranked if any(b["class_id"] == class_id for b in r["boxes"])]
        selected.update(r["filestem"] for r in eligible[:cfg["per_class_images"]])
    return [r for r in rows if r["filestem"] in selected]


def prediction_file(checkpoint="best.pt"):
    pointer = json.loads((LOCAL / "prediction_active.json").read_text())
    directory = LOCAL / "inference" / pointer[checkpoint]
    summary = json.loads((directory / "summary.json").read_text())
    path = directory / "predictions.jsonl"
    if sha(path) != summary["predictions_sha256"] or summary["config_sha256"] != identity(config()):
        raise ValueError("Completed training prediction identity changed")
    return path, summary
