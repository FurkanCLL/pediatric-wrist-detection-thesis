"""Read-only baseline preservation, including actual checkpoint/sidecar verification."""

import csv
import hashlib
import io
import json

from wrist_train.common import configure_runtime

configure_runtime()
import torch

from wrist_train.checkpoints import check_payload, checked_bytes, nested_finite, sidecar
from wrist_train.protocol import load_protocol
from .common import LOCAL, PUBLIC, ROOT, save_json, sha256

BASELINE = ROOT / "outputs/training/standard_baseline"


def inventory():
    # Hash every existing run artifact without rewriting its timestamps, contents or names.
    return {p.relative_to(BASELINE).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(BASELINE.rglob("*")) if p.is_file()}


def verify():
    protocol, identity = load_protocol()
    if json.loads((BASELINE / "run_identity.json").read_text()) != identity:
        raise ValueError("Baseline identity differs from the still-locked implementation")
    if json.loads((BASELINE / "frozen_protocol.json").read_text()) != protocol:
        raise ValueError("Baseline configuration differs from the frozen protocol")
    before = inventory()
    prior = LOCAL / "baseline_inventory.json"
    if prior.exists() and json.loads(prior.read_text()) != before:
        raise ValueError("Baseline artifacts changed since preservation began")
    save_json(prior, before)
    with (BASELINE / "results.csv").open() as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if [r["epoch"] for r in rows] != list(range(1, 81)) or not nested_finite(rows):
        raise ValueError("Missing or non-finite baseline epoch history")
    best_row = max(rows, key=lambda r: (r["metrics/mAP50-95(B)"], r["epoch"]))
    summaries = [json.loads(line) for line in (BASELINE / "epochs.jsonl").read_text().splitlines()]
    classes = {r["epoch"]: r["classes"] for r in map(json.loads, (BASELINE / "per_class.jsonl").read_text().splitlines())}
    if len(summaries) != 80 or set(classes) != set(range(1, 81)):
        raise ValueError("Incomplete numerical or per-class logs")
    checkpoints = []
    selected = None
    for path in sorted((BASELINE / "weights").glob("*.pt")):
        content = checked_bytes(path)
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
        check_payload(payload, identity, require_resume=False)
        completed = payload["epoch"] + 1
        manifest = json.loads(sidecar(path).read_text())
        if manifest["completed_epoch"] != completed or manifest["protocol_sha256"] != identity["protocol_sha256"]:
            raise ValueError("Checkpoint sidecar metadata disagrees with saved identity/epoch")
        if not nested_finite({k: payload[k] for k in ("optimizer", "scaler", "scheduler", "accounting")}):
            raise ValueError("Non-finite saved recovery state")
        if payload["updates"] != payload["accounting"]["successful"] or payload["scheduler"]["last_epoch"] != completed - 1:
            raise ValueError("Inconsistent recovery state")
        if any(payload["train_args"].get(k) != v for k, v in protocol["training"].items() if k != "exist_ok"):
            raise ValueError("A checkpoint training parameter differs from the frozen protocol")
        if not all(p.dtype == torch.float32 for key in ("model", "ema", "best_ema") for p in payload[key].parameters()):
            raise ValueError("Baseline checkpoint model precision changed")
        receipt = {"file": path.name, **json.loads(sidecar(path).read_text()),
                   "metadata_completed_epoch": completed, "finite_model_ema_best_and_optimizer": True,
                   "model_state_tensors": len(payload["model"].state_dict()), "class_mapping_verified": True}
        if path.name == "best.pt":
            if completed != int(best_row["epoch"]) or payload["official"]["best_epoch"] + 1 != completed:
                raise ValueError("Best checkpoint metadata does not match selection history")
            measured = payload["train_metrics"]
            for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if abs(measured[key] - best_row[key]) > 1e-8:
                    raise ValueError("Best checkpoint metrics disagree with native CSV")
            if payload["per_class"] != classes[completed]:
                raise ValueError("Best checkpoint per-class metrics disagree with measured epoch")
            selected = {"completed_epoch": completed, "sha256": receipt["sha256"],
                        "map50": measured["metrics/mAP50(B)"], "map50_95": measured["metrics/mAP50-95(B)"],
                        "classes": payload["per_class"]}
        if path.name == "last.pt":
            if completed != 80 or payload["accounting"] != summaries[-1]["optimizer"]:
                raise ValueError("Final checkpoint and numerical log disagree")
        checkpoints.append(receipt)
        del payload, content
    if selected is None:
        raise ValueError("Best checkpoint is missing")
    completed = json.loads((BASELINE / "completed.json").read_text())
    expected = summaries[-1]["optimizer"]
    if completed != {"epochs": 80, "successful_updates": expected["successful"], "skipped_updates": expected["skipped"]}:
        raise ValueError("Completion receipt and accounting disagree")
    events = [json.loads(line) for line in (BASELINE / "events.jsonl").read_text().splitlines()]
    if any(e["event"] == "resume" for e in events):
        raise ValueError("Unexpected baseline continuation; inspect its segments before simulation")
    amp_events = [e for e in events if e["event"] == "amp"]
    if sum(not e["updated"] for e in amp_events) != expected["skipped"]:
        raise ValueError("Recorded AMP skips disagree with final accounting")
    if inventory() != before:
        raise ValueError("Baseline changed while it was inspected")
    result = {"status": "verified and unchanged", "protocol_sha256": identity["protocol_sha256"],
              "split_manifest_sha256": identity["split_manifest_sha256"], "pretrained_sha256": identity["pretrained_sha256"],
              "implementation_source_sha256": identity["implementation_source_sha256"], "environment": identity["environment"],
              "artifact_inventory_sha256": hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
              "metrics_file_hashes": {name: before[name]["sha256"] for name in ("results.csv", "epochs.jsonl", "per_class.jsonl", "events.jsonl", "frozen_protocol.json", "run_identity.json")},
              "implementation_revision": identity["implementation_revision"], "epochs": 80,
              "artifact_files": len(before), "checkpoint_files": checkpoints, "best": selected,
              "successful_updates": expected["successful"], "skipped_updates": expected["skipped"],
              "nominal_attempts": expected["attempts"], "resumes": 0,
              "amp_recorded_skips": sum(not e["updated"] for e in amp_events),
              "amp_scale_min": min(e["scale_after"] for e in amp_events), "amp_scale_max": max(e["scale_after"] for e in amp_events),
              "total_epoch_seconds": sum(r["epoch_seconds"] for r in summaries),
              "test_inference_performed": False, "new_model_inference_performed": False,
              "limitations": ["validation-selected checkpoint; no held-out test result", "rare-class validation support is small",
                              "baseline metrics describe performance, never enter policy construction"]}
    save_json(PUBLIC / "baseline_preservation.json", result)
    print({"best": selected, "successful_updates": expected["successful"], "skipped_updates": expected["skipped"]})


if __name__ == "__main__":
    verify()
