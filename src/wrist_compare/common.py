"""Provenance, private attempt boundaries and reuse of preserved baseline evidence."""

import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from wrist_eval.common import (ROOT, LOCAL as OLD, PUBLIC as OLD_PUBLIC, CLASSES, config,
                               identity, sha, source_snapshot, validation_records)
from wrist_eval.metrics import validate_predictions
from wrist_train.checkpoints import checked_bytes, check_payload, sidecar
from wrist_train.protocol import load_protocol
from wrist_rarity.protocol import load_condition, RUN_ROOT
from wrist_rarity.launch import check_condition_checkpoint
from wrist_completion.original import write_new

LOCAL = ROOT/"outputs/rarity_comparison"
PUBLIC = ROOT/"docs/evidence/rarity_comparison"
START_REVISION = "c5e64e9da4596fe8c752c816b140b3984f195110"
PROTECTED_OUTPUTS = ("training/standard_baseline", "training/rarity_only", "error_analysis_sahi",
    "adaptive_refinement", "rarity_training_completion", "rarity_training_readiness",
    "error_guided_training_feasibility", "class_aware_design")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    cfg = read(ROOT/"configs/rarity_comparison.json")
    require(identity(config()) == cfg["common_inference_config_sha256"], "Common inference configuration changed")
    require(cfg["validation_images"] == 3088 and cfg["validation_patients"] == 914 and
            cfg["bootstrap_replicates"] == 500 and cfg["bootstrap_seed"] == 314159 and
            not any(cfg[k] for k in ("training", "test_evaluation", "sliced_or_adaptive_inference")), "Comparison scope differs")
    original = read(OLD_PUBLIC/"implementation_provenance.json")
    for name, expected in original["normalized_source_sha256"].items():
        require(hashlib.sha256((ROOT/name).read_text(encoding="utf-8").encode()).hexdigest() == expected,
                "Preserved evaluator source changed: "+name)
    # The standard path still uses the existing library adapter, without invoking tiled inference.
    import sahi
    compatibility = read(OLD_PUBLIC/"compatibility_verified.json")
    require(sahi.__version__ == compatibility["sahi"], "Inference adapter dependency version differs")
    for name, expected in compatibility["sahi_source_sha256"].items():
        require(sha(Path(sahi.__file__).parent/name) == expected, "Inference adapter dependency source changed")
    return cfg


def source_identity():
    files = list((ROOT/"src/wrist_compare").glob("*.py"))+[ROOT/"configs/rarity_comparison.json"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in sorted(files)}


def checked_checkpoint(condition):
    import torch
    cfg = settings()
    require(condition in ("A", "D"), "Only A and D checkpoints are authorized")
    if condition == "A":
        path = ROOT/"outputs/training/standard_baseline/weights/best.pt"
        protocol, lock = load_protocol()
        expected, epoch = cfg["baseline_checkpoint_sha256"], 22
    else:
        path = RUN_ROOT/"weights/best.pt"
        protocol, lock = load_condition()
        expected, epoch = cfg["rarity_checkpoint_sha256"], 26
        completion = read(ROOT/"docs/evidence/rarity_training_completion/verification_summary.json")
        require(completion["passed"] and completion["rarity_condition_sha256"] == lock["rarity_condition_sha256"] and
                next(c for c in completion["checkpoints"] if c["name"] == "best.pt")["sha256"] == expected,
                "Phase 4C verified checkpoint binding differs")
    require(sha(path) == expected, "Checkpoint hash differs")
    payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
    check_payload(payload, lock, require_resume=False)
    require(payload["epoch"]+1 == epoch and read(sidecar(path))["completed_epoch"] == epoch and
            read(sidecar(path))["protocol_sha256"] == lock["protocol_sha256"], "Checkpoint epoch/sidecar differs")
    if condition == "D":
        check_condition_checkpoint(payload, lock)
    for key, value in protocol["training"].items():
        require(payload["train_args"].get(key) == (True if key == "exist_ok" else value), "Frozen training argument differs: "+key)
    for model in (payload["model"], payload["ema"], payload["best_ema"]):
        require(model.names == dict(enumerate(CLASSES)) and all(not t.is_floating_point() or t.dtype == torch.float32
                for t in model.state_dict().values()), "Model mapping or FP32 state differs")
    return payload, {"condition": condition, "sha256": expected, "epoch": epoch,
        "protocol_sha256": lock["protocol_sha256"], "pretrained_sha256": lock["pretrained_sha256"]}


def load_predictions(path, rows, expected_sha):
    require(sha(path) == expected_sha, "Prediction artifact hash differs")
    items = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]
    require([r["stem"] for r in items] == [r["filestem"] for r in rows], "Prediction count/order/membership differs")
    result = {}
    for row, item in zip(rows, items):
        pred = np.asarray(item["pred"], dtype=np.float32).reshape(-1, 6)
        checked = validate_predictions(pred, row["width"], row["height"])
        require(np.array_equal(pred, checked) and len(pred) <= 300, "Prediction order or cap differs")
        if "patient_id" in item:
            require(item["patient_id"] == row["patient_id"], "Prediction patient linkage differs")
        result[row["filestem"]] = pred
    return result, items


def references(rows):
    cfg = settings()
    summary = read(OLD_PUBLIC/"full_standard_serial.json")
    require(summary["checkpoint_sha256"] == cfg["baseline_checkpoint_sha256"] and
            summary["config_sha256"] == cfg["common_inference_config_sha256"] and
            summary["prediction_file_sha256"] == cfg["baseline_predictions_sha256"], "Preserved A provenance differs")
    predictions, _ = load_predictions(OLD/"full/standard_serial/predictions.jsonl", rows, cfg["baseline_predictions_sha256"])
    active = (OLD/read(OLD/"reference_active.json")["directory"]).resolve()
    require(active.is_relative_to(OLD.resolve()), "Native reference escaped its output root")
    native = read(active/"predictions.json")
    require(set(native) == set(predictions), "Native shape-reference membership differs")
    # Reconstruct native rectangular batch shapes from dimensions, without opening images or using outcomes.
    ratios = np.array([r["height"]/r["width"] for r in rows])
    order, shapes = ratios.argsort(), {}
    for start in range(0, len(rows), 12):
        indices = order[start:start+12]
        minimum, maximum = ratios[indices].min(), ratios[indices].max()
        shape = [maximum, 1] if maximum < 1 else [1, 1/minimum] if minimum > 1 else [1, 1]
        net = (np.ceil(np.array(shape)*768/32+.5).astype(int)*32).tolist()
        for i in indices:
            stem = rows[i]["filestem"]
            require(native[stem]["net_shape"] == net, "Native rectangular geometry reference differs")
            shapes[stem] = net
    return predictions, shapes, summary


def protected_inventory():
    return {p.relative_to(ROOT).as_posix(): sha(p) for name in PROTECTED_OUTPUTS
            for p in sorted((ROOT/"outputs"/name).rglob("*")) if p.is_file()}


def preservation(directory, stage):
    frozen = read(ROOT/"outputs/error_analysis_sahi/source_snapshot_before.json")
    snapshot = source_snapshot()
    require(identity(snapshot) == identity(frozen), "Immutable raw/prepared stat inventory changed")
    protected = ["src/wrist_eval", "src/wrist_rarity", "src/wrist_train", "src/wrist_completion", "src/wrist_prepare",
        "src/wrist_design", "src/wrist_mining", "src/wrist_audit", "requirements.txt", "requirements-cuda.txt",
        "requirements-inference.txt", "pyproject.toml"]
    # New comparison files are additive; every pre-existing config/evidence file stays frozen.
    previous = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", START_REVISION,
                                       "configs", "docs/evidence"], cwd=ROOT, text=True).splitlines()
    require(not subprocess.check_output(["git", "diff", START_REVISION, "--", *protected, *previous], cwd=ROOT).strip(),
            "Earlier implementation/configuration/public evidence changed")
    current = protected_inventory()
    if stage == "before":
        write_new(directory/"protected_inventory_before.json", current)
    else:
        require(current == read(directory/"protected_inventory_before.json"), "Earlier private experiment output changed")
    result = {"passed": True, "stage": stage, "protected_private_files": len(current),
        "protected_private_inventory_sha256": identity(current), "data_stat_entries": len(snapshot),
        "data_stat_inventory_sha256": identity(snapshot), "earlier_source_configs_public_evidence_unchanged": True}
    write_new(directory/f"preservation_{stage}.json", result)
    return result


def study_path(name):
    require(name and all(c.islower() or c.isdigit() or c in "_-" for c in name), "Use a simple new study name")
    return LOCAL/name


def frozen_study(directory):
    manifest = read(directory/"manifest.json")
    require(manifest["source_identity"] == source_identity() and manifest["configuration"] == settings(),
            "Comparison source/configuration changed after initialization")
    return manifest


def install_access_guard():
    import sys
    blocked = [(ROOT/p).resolve() for p in ("data/raw", "data/prepared/grazpedwri_detection/images/test",
                                           "data/prepared/grazpedwri_detection/labels/test")]
    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, Path)):
            path = Path(args[0]).resolve()
            if path.name in (".env", ".env.local") or any(path.is_relative_to(base) for base in blocked):
                raise PermissionError("Comparison attempted a forbidden file access")
    sys.addaudithook(guard)
