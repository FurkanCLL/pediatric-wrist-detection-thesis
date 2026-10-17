"""Supplement the frozen verifier using existing evidence only; no model forward calls."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
import yaml

from wrist_rarity.protocol import RUN_ROOT, ROOT, load_condition
from wrist_rarity.launch import check_condition_checkpoint
from wrist_train.common import verify_frozen_data
from wrist_train.protocol import canonical_hash
from wrist_train.checkpoints import checked_bytes, sidecar, check_payload, nested_finite
from wrist_train.budget import optimizer_budget
from wrist_eval.common import baseline_inventory, source_snapshot
from wrist_design.common import training_records, arrays
from .checks import require, ordered, csv_history, class_history, sampling_counts, amp_history, optimizer_coverage
from .original import inventory, write_new


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def preservation():
    # Dataset preservation is stat-only: held-out radiograph/label bytes are never opened.
    baseline = baseline_inventory()
    require(baseline == read(ROOT/"outputs/error_analysis_sahi/baseline_inventory_before.json"), "Baseline changed")
    snapshot = source_snapshot()
    require(canonical_hash(snapshot) == canonical_hash(read(ROOT/"outputs/error_analysis_sahi/source_snapshot_before.json")),
            "Raw/prepared stat inventory changed")
    protected = ["src/wrist_train", "src/wrist_rarity", "src/wrist_design", "src/wrist_mining", "src/wrist_eval",
        "src/wrist_prepare", "src/wrist_audit", "configs", "requirements.txt", "requirements-cuda.txt",
        "requirements-inference.txt", "pyproject.toml"]
    protected += [p.relative_to(ROOT).as_posix() for p in (ROOT/"docs/evidence").iterdir()
                  if p.is_dir() and p.name != "rarity_training_completion"]
    reference = "bb5a60de9c31e241d29a403348d1654d216162b2"
    require(not subprocess.check_output(["git", "diff", reference, "--", *protected], cwd=ROOT).strip(),
            "Frozen source/configuration or historical public evidence changed")
    return {"baseline_files_unchanged": len(baseline), "baseline_inventory_sha256": canonical_hash(baseline),
        "raw_prepared_stat_entries_unchanged": len(snapshot), "data_stat_inventory_sha256": canonical_hash(snapshot),
        "protected_sources_configs_earlier_public_evidence_unchanged_since": reference}


def checkpoint(path, protocol, identity, rows, classes, epochs, receipts, snapshots):
    # Load only local hash-checked checkpoints on CPU; there is no inference or optimizer step.
    payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
    check_payload(payload, identity, require_resume=False)
    check_condition_checkpoint(payload, identity)
    n = payload["epoch"]+1
    require(1 <= n <= len(rows), "Checkpoint epoch outside completed history")
    manifest = read(sidecar(path))
    require(manifest["completed_epoch"] == n and manifest["protocol_sha256"] == identity["protocol_sha256"],
            "Sidecar semantic identity differs")
    require(payload["official"].get("resumable") and not payload["official"].get("best_snapshot_only"),
            "Expected a full-state uninterrupted checkpoint")
    for key, value in identity.items():
        require(payload["official"].get(key) == value, "Checkpoint provenance differs: "+key)
    require(payload["official"]["rarity_completed_epoch"] == receipts[n-1], "Embedded consumption evidence differs")
    expected_best = max(range(n), key=lambda i: (rows[i]["metrics/mAP50-95(B)"], i))
    require(payload["official"]["best_epoch"] == expected_best and
            payload["official"]["rarity_best_epoch_evidence"] == receipts[expected_best], "Embedded best selection differs")
    require(payload["best_fitness"] == rows[expected_best]["metrics/mAP50-95(B)"] and
            payload["stopper"] == {"best_fitness": payload["best_fitness"], "best_epoch": expected_best+1},
            "Best fitness/stopper differs")
    require(payload["per_class"] == classes[n-1]["classes"], "Checkpoint class metrics differ")
    for key, values in payload["train_results"].items():
        require(key in rows[0] and values == [r[key] for r in rows[:n]], "Checkpoint CSV history differs")
    require(set(payload["train_results"]) == set(rows[0]), "Checkpoint CSV columns differ")
    for key in (*[k for k in rows[0] if k.startswith(("metrics/", "val/"))], "fitness"):
        expected = rows[n-1]["metrics/mAP50-95(B)" if key == "fitness" else key]
        require(payload["train_metrics"][key] == expected, "Checkpoint native metric differs")

    # Compare actual arguments, including every frozen augmentation and validation option.
    args = payload["train_args"]
    for key, value in protocol["training"].items():
        require(args.get(key) == (True if key == "exist_ok" else value), "Checkpoint argument differs: "+key)
    for key, expected in {"model": ROOT/"outputs/gpu_feasibility/weights/yolov8s.pt", "data": RUN_ROOT/"data.yaml",
                          "project": RUN_ROOT.parent, "save_dir": RUN_ROOT}.items():
        require(Path(args[key]).resolve() == expected.resolve(), "Checkpoint path argument differs: "+key)
    require(args["name"] == RUN_ROOT.name, "Checkpoint condition name differs")

    state = snapshots[n-1]
    require(payload["accounting"] == epochs[n-1]["optimizer"] == state["accounting"], "Checkpoint accounting differs")
    require(payload["updates"] == state["accounting"]["successful"], "EMA update count differs")
    require(payload["scheduler"]["last_epoch"] == n-1, "Scheduler position differs")
    require(nested_finite({k: payload[k] for k in ("optimizer", "scaler", "scheduler", "best_fitness")}),
            "Non-finite optimizer/scaler/scheduler")
    expected_scaler = {"scale": state["scale"], "growth_factor": protocol["runtime"]["amp_growth_factor"],
        "backoff_factor": protocol["runtime"]["amp_backoff_factor"],
        "growth_interval": protocol["runtime"]["amp_growth_interval"], "_growth_tracker": state["tracker"]}
    require(payload["scaler"] == expected_scaler, "Saved scaler does not match complete AMP replay")
    for name in ("model", "ema", "best_ema"):
        require(all(not t.is_floating_point() or t.dtype == torch.float32
                    for t in payload[name].state_dict().values()), "Checkpoint model precision differs")
    optimizer = payload["optimizer"]
    # The fixed DFL projection belongs to a group but correctly never acquires Adam moments.
    fixed = optimizer_coverage(payload["model"], optimizer)
    require(fixed == ["model.22.dfl.conv.weight"], "Unexpected frozen parameter")
    for values in optimizer["state"].values():
        require(float(values["step"]) == payload["updates"], "AdamW parameter step differs from EMA")
        require(all(not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.dtype == torch.float32
                    for t in values.values()), "Optimizer precision differs")
    require(len(optimizer["param_groups"]) == 3, "AdamW parameter groups differ")
    t = protocol["training"]
    lr = t["lr0"]*((1-math.cos(math.pi*(n-1)/t["epochs"]))/2*(t["lrf"]-1)+1)
    for i, (group, decay) in enumerate(zip(optimizer["param_groups"], (t["weight_decay"], 0., 0.))):
        require(group["weight_decay"] == decay and group["betas"] == (t["momentum"], .999) and
                group["initial_lr"] == t["lr0"] and group["fused"] and
                math.isclose(group["lr"], lr, abs_tol=1e-14) and
                float(f"{group['lr']:.6g}") == rows[n-1][f"lr/pg{i}"], "AdamW settings/schedule differ")
    require(0 <= payload["official"]["pending_microbatches"] < 4 and
            set(payload["rng"]) == {"python", "numpy", "torch", "cuda"}, "Recovery bookkeeping differs")
    return payload, {"name": path.name, "completed_epoch": n, "sha256": manifest["sha256"],
        "bytes": manifest["bytes"], "all_model_optimizer_states_finite_fp32": True,
        "optimizer_parameter_states": len(optimizer["state"]), "fixed_parameters_without_state": fixed,
        "successful_updates": payload["updates"],
        "pending_microbatches": payload["official"]["pending_microbatches"], "amp_scale": state["scale"]}


def verify(destination):
    destination = Path(destination).resolve()
    require(read(destination/"original_rerun.json")["passed"], "Frozen verifier rerun has not passed")
    before = read(destination/"run_inventory_before.json")
    require(inventory(RUN_ROOT) == before, "Run changed since the initial audit inventory")
    protocol, identity = load_condition()
    complete, launch = read(RUN_ROOT/"completed.json"), read(RUN_ROOT/"launch.json")
    # This completed run is uninterrupted. Reject an unfamiliar lineage instead of silently merging it.
    require(read(RUN_ROOT/"active_run.json") == {"directory": ".", "resume_checkpoint": None} and
            launch["resume"] is False and not (RUN_ROOT/"resumes").exists() and
            not list(RUN_ROOT.rglob("failure.json")) and not list(RUN_ROOT.rglob("*.partial")) and
            not (RUN_ROOT/"run.lock").exists(), "Incomplete, failed or continued run requires separate investigation")
    require(complete["epochs"] == 80 and complete["diagnostic_only"] is False and
            read(RUN_ROOT/"run_identity.json") == launch["identity"] == identity and
            read(RUN_ROOT/"condition_protocol.json") == protocol, "Run/protocol identity differs")
    require(launch["preflight"]["passed"] and launch["diagnostic_only"] is False and
            launch["git_revision"] == "bb5a60de9c31e241d29a403348d1654d216162b2", "Official launch provenance differs")
    records, frozen = verify_frozen_data()
    require(all(frozen[k] == protocol["dataset"][k] for k in ("split_manifest_sha256", "records_sha256", "policy_sha256")),
            "Frozen dataset identity differs")
    require(hashlib.sha256((ROOT/"outputs/dataset_preparation/completed.sqlite").read_bytes()).hexdigest() ==
            identity["prepared_journal_sha256"], "Preparation journal changed")
    require(hashlib.sha256(Path(launch["source"]).read_bytes()).hexdigest() == identity["pretrained_sha256"],
            "COCO initialization bytes differ")
    data = yaml.safe_load((RUN_ROOT/"data.yaml").read_text())
    require("test" not in data and data["names"] == dict(enumerate(protocol["dataset"]["classes"])) and
            data["channels"] == 3 and data["nc"] == 8, "Train/validation YAML class or split differs")
    for split in ("train", "val"):
        require(Path(data[split]).resolve() == (ROOT/f"data/prepared/grazpedwri_detection/images/{split}").resolve(),
                "Run YAML split path differs")
    with (RUN_ROOT/"results.csv").open() as stream:
        rows = csv_history(list(csv.DictReader(stream)))
    classes, epochs, events = [lines(RUN_ROOT/n) for n in ("per_class.jsonl", "epochs.jsonl", "events.jsonl")]
    best_epoch = class_history(classes, rows)
    ordered(epochs, 80, "Epoch progress")
    receipts = [read(RUN_ROOT/f"sampling/epoch{n:03}_verified.json") for n in range(1, 81)]
    ordered(receipts, 80, "Consumption receipts")
    snapshots, amp = amp_history(events, receipts, protocol["runtime"])
    expected_budget = optimizer_budget(14061, 6, 24, 80, 1)["updates_per_epoch"]
    counts, patient, _ = arrays(training_records())
    positive, lesion_patients = counts[:, 0] > 0, np.unique(patient[counts[:, 5] > 0])
    sampling = []
    previous = {k: 0 for k in ("attempts", "successful", "skipped")}
    for i, (receipt, row, epoch, state) in enumerate(zip(receipts, rows, epochs, snapshots)):
        actual = np.array([r[2] for b in lines(RUN_ROOT/f"sampling/epoch{i+1:03}_consumed.jsonl")
                           for r in b["receipts"]], dtype=np.int32)
        sample = sampling_counts(actual, patient, positive, lesion_patients)
        require(all(receipt[k] == v for k, v in sample.items() if k in receipt), "Saved sampling summary differs")
        sampling.append(sample)
        require(epoch["optimizer"] == state["accounting"], "Epoch optimizer cumulative counters differ")
        delta = {k: state["accounting"][k]-previous[k] for k in previous}
        require(delta == receipt["optimizer_epoch"] and delta["attempts"] == expected_budget[i], "Epoch update budget differs")
        previous = {k: state["accounting"][k] for k in previous}
        require(receipt["accumulation_values"] == ([1, 2, 3, 4] if i == 0 else [4]), "Accumulation schedule differs")
        generator = torch.Generator().manual_seed(6148914691236517205+42-1+i)
        seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator).item())
        require(receipt["worker_seeds"] == [seed, seed+1] and receipt["workers"] == [0, 1], "Worker schedule differs")
        for key, metric in zip(("precision", "recall", "map50", "map50_95"),
                               ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")):
            require(epoch[key] == row[metric], "Progress native metrics differ")
        for key in ("box_loss", "cls_loss", "dfl_loss"):
            require(float(f"{epoch['train_losses'][key]:.5f}") == row["train/"+key] and
                    epoch["validation_losses"][key] == row["val/"+key], "Progress losses differ")
        require(epoch["training_seconds"] == receipt["training_seconds"] and
                epoch["training_memory"] == receipt["training_memory"] and receipt["finite_live_and_ema"],
                "Epoch training health differs")
        val = complete["validation_runs"][i]
        require(all(val[k] == v for k, v in {"trainer_integrated": True, "images": 3088, "batches": 258,
                "batch": 12, "workers": 4, "rect": True, "augmentation": False}.items()) and
                val["seconds"] == epoch["validation_seconds"], "Existing validation record differs")
    require(len(complete["validation_runs"]) == 80 and complete["accounting"] == snapshots[-1]["accounting"] and
            complete["best_epoch"] == best_epoch and complete["successful_updates"] == previous["successful"] and
            complete["skipped_updates"] == previous["skipped"], "Completion counters differ")
    require([e["event"] for e in events if e["event"] != "amp"] ==
            ["training_started", "rarity_loader_ready"]+["rarity_epoch_plan"]*80+["training_loop_completed"],
            "Unexpected interruption/retry/failure event")

    checkpoints, best_state, last_best = [], None, None
    expected_names = {"best.pt", "best.previous.pt", "last.pt", "last.previous.pt"} | {f"epoch{i:03}.pt" for i in range(10, 81, 10)}
    require({p.name for p in (RUN_ROOT/"weights").glob("*.pt")} == expected_names, "Checkpoint set differs")
    for path in sorted((RUN_ROOT/"weights").glob("*.pt")):
        payload, summary = checkpoint(path, protocol, identity, rows, classes, epochs, receipts, snapshots)
        checkpoints.append(summary)
        if path.name == "best.pt":
            require(summary["completed_epoch"] == best_epoch, "Selected best checkpoint epoch differs")
            best_state = {k: v.clone() for k, v in payload["ema"].state_dict().items()}
        if path.name == "last.pt":
            require(summary["completed_epoch"] == 80, "Last checkpoint is not final epoch")
            last_best = {k: v.clone() for k, v in payload["best_ema"].state_dict().items()}
        if path.name.startswith("epoch"):
            require(summary["completed_epoch"] == int(path.stem[5:]), "Periodic filename epoch differs")
        del payload
    require(best_state.keys() == last_best.keys() and all(torch.equal(best_state[k], last_best[k]) for k in best_state),
            "Final embedded best EMA differs from selected checkpoint EMA")
    preserved = preservation()
    require(inventory(RUN_ROOT) == before, "Training artifacts changed during supplementary audit")
    result = {"passed": True, "decision": "GO for Phase 4D review; no Phase 4D work started",
        "training_revision": launch["git_revision"], "audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "common_protocol_sha256": identity["protocol_sha256"], "rarity_condition_sha256": identity["rarity_condition_sha256"],
        "completed_epochs": 80, "uninterrupted": True, "wall_seconds": complete["wall_seconds"],
        "csv_elapsed_seconds": rows[-1]["time"], "training_seconds": sum(e["training_seconds"] for e in epochs),
        "validation_seconds": sum(e["validation_seconds"] for e in epochs), "best_epoch": best_epoch,
        "best_native_metrics": {k: rows[best_epoch-1][k] for k in rows[0] if k.startswith("metrics/")},
        "final_native_metrics": {k: rows[-1][k] for k in rows[0] if k.startswith("metrics/")},
        "best_per_class": classes[best_epoch-1]["classes"], "csv_rows": len(rows), "class_records": len(classes)*8,
        "selection_rule": "macro eight-class native mAP50-95 rounded to five decimals; latest exact tie",
        "sampling_per_epoch_ranges": {k: [min(s[k] for s in sampling), max(s[k] for s in sampling)] for k in sampling[0]},
        "total_consumed_draws": sum(s["draws"] for s in sampling), "total_consumed_batches": 80*2344,
        "different_repeated_augmented_pairs_range": [min(r["different_augmented_pairs"] for r in receipts), max(r["different_augmented_pairs"] for r in receipts)],
        "accounting": complete["accounting"], "amp": amp, "checkpoints": checkpoints,
        "peak_training_memory_gib": {k: max(e["training_memory"][k] for e in epochs) for k in epochs[0]["training_memory"]},
        "telemetry": complete["telemetry"], "preservation": preserved, "original_training_files_unchanged": len(before),
        "training_started_by_audit": False, "new_validation_inference": False, "test_image_label_prediction_contents_opened": 0}
    write_new(destination/"supplementary_verification.json", result)
    print(f"Supplementary verification passes: 80 epochs, {len(checkpoints)} checkpoints, best epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    verify(parser.parse_args().destination)
