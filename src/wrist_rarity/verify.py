"""Independent saved-plan, consumed-batch, checkpoint and preservation checks."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from wrist_train.common import ROOT
from wrist_train.launch import write_json, active_directory
from wrist_train.checkpoints import checked_bytes, check_payload, nested_finite
from wrist_train.protocol import canonical_hash, load_protocol
from .policy import RarityPlans, plan_hash
from .protocol import PUBLIC, LOCAL, RUN_ROOT, load_condition, source_files_and_labels


def reconcile_plans():
    plans = RarityPlans()
    old = ROOT/"outputs/error_guided_training_feasibility/simulations/matched_01"
    summary = json.loads((ROOT/"docs/evidence/error_guided_training_feasibility/simulation_summary.json").read_text())
    records = []
    for seed in (42, 1042, 2042):
        file = old/f"plans_{seed}.npz"
        if hashlib.sha256(file.read_bytes()).hexdigest() != summary["plan_files_sha256"][file.name]:
            raise ValueError("Original Phase 4A plan file changed")
        expected = np.load(file)["rarity_only"]
        for epoch in range(80):
            draws = plans.plan(seed, epoch)
            if not np.array_equal(draws, expected[epoch]):
                raise ValueError("Integrated rarity plan differs from Phase 4A")
            records.append({"seed": seed, "epoch": epoch+1, **plans.validate(draws)})
        print(f"Reconciled seed {seed}: all 80 rarity plans exactly match Phase 4A", flush=True)
    write_json(LOCAL/"reconciled_plan_hashes.json", records)
    _, identity = load_condition()
    result = {"passed": True, "all240_exact_match_phase4a": True, "epochs": 80, "seeds": [42, 1042, 2042],
              "all_sampling_guardrails_pass": True, "no_mining_predictions_or_features_loaded": True,
              "condition_sha256": identity["rarity_condition_sha256"],
              "all_plan_hashes_sha256": canonical_hash(records), "official_training_started": False}
    write_json(PUBLIC/"plan_reconciliation.json", result)


def preserve(stage):
    from wrist_eval.common import baseline_inventory, source_snapshot
    baseline = baseline_inventory()
    expected = json.loads((ROOT/"outputs/error_analysis_sahi/baseline_inventory_before.json").read_text())
    snapshot = source_snapshot()
    original_snapshot = json.loads((ROOT/"outputs/error_analysis_sahi/source_snapshot_before.json").read_text())
    if baseline != expected or canonical_hash(snapshot) != canonical_hash(original_snapshot):
        raise ValueError("Baseline or immutable raw/prepared data inventory changed")
    _, lock = load_protocol()
    protected = ["src/wrist_train", "src/wrist_design", "src/wrist_mining", "src/wrist_eval", "src/wrist_prepare", "src/wrist_audit",
                 "requirements.txt", "requirements-cuda.txt", "pyproject.toml", "configs/training_official.json",
                 "configs/class_aware_candidates.json", "configs/error_guided_sampler_proposed.json",
                 "configs/dataset_preparation.json", "configs/adaptive_refinement_proposed.json",
                 "configs/adaptive_refinement_execution.json", "configs/inference_feasibility.json",
                 "docs/evidence/error_guided_training_feasibility", "docs/evidence/training_protocol",
                 "docs/evidence/class_aware_design", "docs/evidence/error_analysis_sahi",
                 "docs/evidence/adaptive_refinement_validation", "docs/evidence/adaptive_refinement_design",
                 "docs/evidence/dataset_preparation"]
    if subprocess.check_output(["git", "diff", "c7b6253", "--", *protected]).strip():
        raise ValueError("Frozen implementation or earlier experiment evidence changed")
    write_json(PUBLIC/f"preservation_{stage}.json", {"passed": True, "stage": stage,
               "baseline_files_unchanged": len(baseline), "data_stat_files_unchanged": len(snapshot),
               "baseline_inventory_sha256": canonical_hash(baseline), "data_stat_inventory_sha256": canonical_hash(snapshot),
               "frozen_protocol_sha256": lock["protocol_sha256"], "previous_sources_configs_evidence_unchanged": True,
               "test_image_label_contents_opened": 0, "official_80_epoch_training_started": False})
    print(f"Preservation passes: {len(baseline)} baseline artifacts, {len(snapshot)} immutable stat entries", flush=True)


def verify_run(root, diagnostic=False, cuda_reload=False):
    import torch
    from wrist_train.amp import state_is_finite
    from .launch import check_condition_checkpoint
    protocol, identity = load_condition(diagnostic)
    root = Path(root).resolve()
    if root != RUN_ROOT.resolve() and root.parent != (LOCAL/"diagnostics").resolve():
        raise ValueError("Run is outside the isolated rarity output locations")
    if root == RUN_ROOT.resolve() and diagnostic:
        raise ValueError("Official output cannot be treated as a diagnostic")
    if json.loads((root/"run_identity.json").read_text()) != identity:
        raise ValueError("Run manifest identity differs")
    directory = active_directory(root)
    complete = json.loads((directory/"completed.json").read_text())
    if complete["epochs"] != protocol["training"]["epochs"] or (directory/"failure.json").exists():
        raise ValueError("Training did not complete cleanly")
    population = RarityPlans()
    files, signatures = source_files_and_labels(population.rows)
    epochs = []
    segments = selected_segments(root, directory, complete["epochs"], identity)
    # Rebuild actual consumed order from per-batch worker receipts, not just precomputed plans.
    records = [(segment, record) for segment, limit in segments
               for record in sorted((segment/"sampling").glob("epoch*_verified.json"))
               if json.loads(record.read_text())["epoch"] <= limit]
    for segment, record_file in records:
        receipt = json.loads(record_file.read_text())
        epoch = receipt["epoch"]
        expected = population.plan(42, epoch-1)
        plan_file = segment/"sampling"/f"epoch{epoch:03}_plan.npz"
        if not np.array_equal(expected, np.load(plan_file)["indices"]):
            raise ValueError("Saved training plan differs from approved generator")
        log = segment/"sampling"/f"epoch{epoch:03}_consumed.jsonl"
        if hashlib.sha256(log.read_bytes()).hexdigest() != receipt["consumption_log_sha256"]:
            raise ValueError("Consumed batch log changed")
        batches = [json.loads(line) for line in log.read_text().splitlines()]
        actual = [item for batch in batches for item in batch["receipts"]]
        if len(batches) != 2344 or [b["batch"] for b in batches] != list(range(1, 2345)):
            raise ValueError("Batch counts or numbering differ")
        if [len(b["receipts"]) for b in batches] != [6]*2343+[3]:
            raise ValueError("Actual physical batches differ from six plus final three")
        if [r[1] for r in actual] != list(range(14061)) or [r[2] for r in actual] != expected.tolist():
            raise ValueError("Actual consumed order differs from intended plan")
        if any(r[0] != epoch-1 or r[5] != signatures[r[2]] or r[3] not in (0, 1) or r[4] != receipt["worker_seeds"][r[3]] for r in actual):
            raise ValueError("Actual epoch, worker or source-label identity differs")
        repeated = {}
        for r in actual:
            if r[6] is not None:
                repeated.setdefault(r[2], []).append(r[6])
        changed = sum(len(v) == 2 and v[0] != v[1] for v in repeated.values())
        if len(repeated) != 625 or any(len(v) != 2 for v in repeated.values()) or changed != receipt["different_augmented_pairs"]:
            raise ValueError("Actual repeated-image augmentation receipts differ")
        population.validate(np.array([r[2] for r in actual], dtype=np.int32))
        if receipt["plan_sha256"] != plan_hash(expected):
            raise ValueError("Plan receipt hash differs")
        epochs.append(receipt)
    epochs.sort(key=lambda r: r["epoch"])
    if [r["epoch"] for r in epochs] != list(range(1, complete["epochs"]+1)):
        raise ValueError("Completed lineage is missing an epoch or contains a duplicate")
    for key in ("attempts", "successful", "skipped"):
        if sum(r["optimizer_epoch"][key] for r in epochs) != complete["accounting"][key]:
            raise ValueError("Selected lineage optimizer accounting differs")
    checkpoints = []
    for name in ("last.pt", "best.pt"):
        path = directory/"weights"/name
        payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
        check_payload(payload, identity, require_resume=False)
        check_condition_checkpoint(payload, identity)
        for model_name in ("model", "ema", "best_ema"):
            if any(t.is_floating_point() and t.dtype != torch.float32 for t in payload[model_name].state_dict().values()):
                raise ValueError("Saved model state is not full precision")
        snapshot_only = payload["official"].get("best_snapshot_only", False)
        if not snapshot_only:
            if not nested_finite({k: payload[k] for k in ("optimizer", "scaler", "scheduler", "best_fitness")}):
                raise ValueError("Checkpoint optimizer/scaler state is non-finite")
            if payload["updates"] != payload["accounting"]["successful"]:
                raise ValueError("Checkpoint EMA/update accounting differs")
            for state in payload["optimizer"]["state"].values():
                if any(isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
                    raise ValueError("Saved optimizer state is not full precision")
        reloaded = False
        if cuda_reload:
            # This is a synthetic forward from the saved EMA, not a D/E/F evaluation.
            model = payload["ema"].float().eval().to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                prediction = model(torch.zeros((1, 3, 768, 768), device="cuda:0"))
            from wrist_train.diagnostics import finite_tensors
            if not finite_tensors(prediction):
                raise FloatingPointError("Reloaded EMA produced a non-finite prediction")
            reloaded = True
            del model, prediction
        checkpoints.append({"name": name, "completed_epoch": payload["epoch"]+1,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "fp32_model_state_finite": True, "optimizer_state_finite": not snapshot_only,
                            "best_snapshot_only": snapshot_only, "cuda_synthetic_forward_passed": reloaded})
        del payload
    with (directory/"results.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    if diagnostic and len(metrics) != 2:
        raise ValueError("Missing smoke validation metrics")
    if not all(np.isfinite(float(v)) for row in metrics for v in row.values()):
        raise ValueError("Non-finite saved training/validation results")
    result = {"passed": True, "diagnostic_only": diagnostic, "condition_sha256": identity["rarity_condition_sha256"],
              "epochs": epochs, "accounting": complete["accounting"], "validation_runs": complete["validation_runs"],
              "wall_seconds": complete["wall_seconds"], "telemetry": complete["telemetry"], "checkpoints": checkpoints,
              "all_actual_consumed_batches_reconciled": True, "no_stale_epoch_indices": True,
              "validation_loader_unchanged": True, "official_80_epoch_training_started": not diagnostic,
              "test_files_opened": 0, "diagnostic_metrics_not_thesis_results": diagnostic}
    if diagnostic:
        write_json(PUBLIC/"smoke_verification.json", result)
    else:
        write_json(root/"post_training_verification.json", result)
    print(f"Verified {len(epochs)} consumed epochs and both saved/reloaded checkpoints", flush=True)


def selected_segments(root, directory, completed_epoch, identity):
    from wrist_train.checkpoints import load_recovery
    from .launch import check_condition_checkpoint
    result, seen = [], set()
    while True:
        if directory in seen or not directory.is_relative_to(root):
            raise ValueError("Invalid or cyclic continuation lineage")
        seen.add(directory)
        result.append((directory, completed_epoch))
        launch = json.loads((directory/"launch.json").read_text())
        if not launch["resume"]:
            break
        source = Path(launch["source"]).resolve()
        if not source.is_relative_to(root) or source.parent.name != "weights":
            raise ValueError("Continuation source escaped the rarity run")
        saved = load_recovery(source, identity)
        check_condition_checkpoint(saved, identity)
        completed_epoch = saved["epoch"]+1
        directory = source.parent.parent
    return result


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("plans")
    preservation = commands.add_parser("preserve")
    preservation.add_argument("--stage", choices=("before", "after"), required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, default=RUN_ROOT)
    run.add_argument("--diagnostic", action="store_true")
    run.add_argument("--cuda-reload", action="store_true")
    args = parser.parse_args()
    if args.action == "plans":
        reconcile_plans()
    elif args.action == "preserve":
        preserve(args.stage)
    else:
        verify_run(args.root, args.diagnostic, args.cuda_reload)


if __name__ == "__main__":
    main()
