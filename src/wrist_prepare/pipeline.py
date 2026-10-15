"""Freeze splits, inspect a bounded pilot, then prepare and validate the dataset."""

import argparse
import json
from pathlib import Path

from wrist_audit.audit import validate_output_path
from .common import CLASSES, SPLITS, digest, json_bytes, protected_write, raw_snapshot
from .images import convert_records
from .records import initialize, load_records


def write_yaml(prepared):
    # This local YAML uses an absolute root to avoid Ultralytics' global datasets-dir resolution.
    text = "path: " + json.dumps(prepared.resolve().as_posix()) + "\n"
    text += "train: images/train\nval: images/val\ntest: images/test\nnc: 8\nchannels: 3\nnames:\n"
    text += "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    protected_write(prepared / "dataset.yaml", text.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "pilot", "record-pilot-review", "prepare", "validate", "loader"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/grazpedwri-dx"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--state", type=Path, default=Path("outputs/dataset_preparation"))
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared/grazpedwri_detection"))
    parser.add_argument("--policy", type=Path, default=Path("configs/dataset_preparation.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--note", help="Technical observations after viewing the pilot sheets")
    args = parser.parse_args()
    for path in (args.state, args.prepared):
        validate_output_path(args.raw, path)
    if not 1 <= args.workers <= 8:
        parser.error("Use between one and eight workers")
    policy = json.loads(args.policy.read_text())
    if tuple(policy["classes"]) != CLASSES or policy["seed"] != 42 or policy["ratios"] != [.7, .15, .15]:
        raise ValueError("This command implements the approved seed-42 eight-class policy")
    if args.stage == "freeze":
        initialize(args.raw, args.audit, args.state, policy)
        return
    records = load_records(args.state)
    freeze = json.loads((args.state / "freeze.json").read_text())
    if digest(json_bytes(policy)) != freeze["policy_sha256"]:
        raise ValueError("Configured policy differs from the frozen policy")
    if args.stage == "pilot":
        from .pilot import review_sheets, select_pilot
        from .validation import validate_dataset
        selection = select_pilot(records, args.audit)
        pilot = args.state / "pilot"
        result = convert_records([e["record"] for e in selection], args.raw, pilot, args.state / "pilot.sqlite", freeze, args.workers)
        validation = validate_dataset([e["record"] for e in selection], args.raw, pilot, args.workers)
        protected_write(args.state / "pilot_selection.json", json_bytes(selection))
        protected_write(args.state / "pilot_validation.json", json_bytes(validation))
        paths = review_sheets(selection, args.raw, pilot, args.state / "visual_review")
        print(json.dumps({"conversion": result, "validation": validation, "review_sheets": [p.as_posix() for p in paths]}, indent=2))
    elif args.stage == "record-pilot-review":
        if not args.note or not (args.state / "pilot_validation.json").is_file():
            raise ValueError("Complete the pilot and record the actual visual observations")
        receipt = {"passed": True, "note": args.note, "policy_sha256": freeze["policy_sha256"],
                   "split_manifest_sha256": freeze["split_manifest_sha256"],
                   "pilot_validation_sha256": digest((args.state / "pilot_validation.json").read_bytes())}
        protected_write(args.state / "pilot_review.json", json_bytes(receipt))
    elif args.stage == "prepare":
        review = json.loads((args.state / "pilot_review.json").read_text())
        if not review["passed"] or review["split_manifest_sha256"] != freeze["split_manifest_sha256"] or review["policy_sha256"] != freeze["policy_sha256"]:
            raise ValueError("A passed pilot review for this frozen policy is required")
        if review["pilot_validation_sha256"] != digest((args.state / "pilot_validation.json").read_bytes()):
            raise ValueError("Pilot validation receipt changed")
        result = convert_records(records, args.raw, args.prepared, args.state / "completed.sqlite", freeze, args.workers)
        write_yaml(args.prepared)
        protected_write(args.state / "preparation_receipt.json", json_bytes({"images": result["images"], "objects": freeze["objects"], "complete": True}))
        print(json.dumps(result))
    elif args.stage == "validate":
        from .validation import validate_dataset
        result = validate_dataset(records, args.raw, args.prepared, args.workers)
        before = json.loads((args.state / "raw_snapshot_before.json").read_text())
        result["raw_stat_snapshot_unchanged"] = before == raw_snapshot(args.raw)
        if not result["raw_stat_snapshot_unchanged"] or result["objects_checked"] != policy["expected_target_objects"]:
            raise ValueError("Raw source changed or prepared object totals differ")
        protected_write(args.state / "validation.json", json_bytes(result))
        print(json.dumps(result, indent=2))
    elif args.stage == "loader":
        from .loader import verify_loader
        result = verify_loader(records, args.prepared, args.state)
        protected_write(args.state / "loader_validation.json", json_bytes(result))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
