"""Bind new execution authorization to unchanged algorithms and preserved predictions."""

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from wrist_compare.common import (ROOT, OLD, OLD_PUBLIC, read, write_new, require, settings,
    checked_checkpoint, load_predictions, references, validation_records, install_access_guard,
    protected_inventory as previous_inventory)
from wrist_eval.common import identity, sha, source_snapshot, config
from wrist_eval.adaptive_config import approved_settings
from wrist_eval.adaptive_study import source_identity as adaptive_sources
from wrist_eval.metrics import validate_predictions

LOCAL = ROOT/"outputs/rarity_inference_comparison"
PUBLIC = ROOT/"docs/evidence/rarity_inference_comparison"
PREVIOUS = ROOT/"outputs/rarity_comparison/study_01"
ADAPTIVE = ROOT/"outputs/adaptive_refinement"
ADAPTIVE_PUBLIC = ROOT/"docs/evidence/adaptive_refinement_validation"
START = "323f221ae6978b9bea0e7398c976642e62e48036"
CONFIGS = ("configs/rarity_sahi_execution.json", "configs/rarity_adaptive_execution.json")


def manifests():
    settings()  # Checks the pinned common inference source and library dependencies.
    result = {c: read(ROOT/p) for c, p in zip(("E", "F"), CONFIGS)}
    proposal, original = approved_settings()
    for condition, value in result.items():
        require(value["experiment"] == condition and value["student_approved"] and value["split"] == "val" and
            value["images"] == 3088 and value["patients"] == 914 and value["pilot_images"] == 24 and
            not any(value[k] for k in ("training_authorized", "test_authorized", "optimization_authorized")), "Execution scope differs")
        require(value["checkpoint"] == {"path": "outputs/training/rarity_only/weights/best.pt", "epoch": 26,
            "sha256": "cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e"}, "Unverified rarity checkpoint")
        require(value["D_predictions_sha256"] == "b653b3d5e1df5ca227b29588e6085bb115d8530c99e934495d6e1d630a2b661d" and
                value["bootstrap"] == {"replicates": 500, "seed": 314159}, "D or uncertainty identity differs")
    require(result["E"]["historical_config_path"] == "configs/inference_feasibility.json" and
        result["E"]["historical_config_sha256"] == identity(config()) and result["E"]["variant"] == "sahi_context" and
        result["E"]["fixed_calls_per_image"] == 5 and result["E"]["fixed_slices_per_image"] == 4, "Historical SAHI binding differs")
    require(result["F"]["historical_proposal_sha256"] == identity(proposal) and
        result["F"]["historical_proposal_path"] == "configs/adaptive_refinement_proposed.json" and
        result["F"]["historical_execution_path"] == "configs/adaptive_refinement_execution.json" and
        result["F"]["historical_execution_sha256"] == identity(original) and
        result["F"]["historical_source_sha256"] == adaptive_sources() and result["F"]["variant"] == "V2" and
        result["F"]["maximum_local_rois"] == 2 and result["F"]["maximum_calls_per_image"] == 3, "Historical adaptive binding differs")
    prior = read(ADAPTIVE_PUBLIC/"full_summary.json")
    require(prior["implementation_source_sha256"] == adaptive_sources(), "Frozen adaptive source changed")
    return result, proposal


def source_identity():
    files = list((ROOT/"src/wrist_inference_comparison").glob("*.py"))+[ROOT/p for p in CONFIGS]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in sorted(files)}


def study_path(name):
    require(name and all(c.islower() or c.isdigit() or c in "_-" for c in name), "Use a new simple study name")
    return LOCAL/name


def frozen_study(directory):
    result = read(directory/"manifest.json")
    current, _ = manifests()
    require(result["source_identity"] == source_identity() and result["executions"] == current, "Execution source/manifest changed")
    return result


def historical_inputs(rows):
    executions, _ = manifests()
    a, shapes, a_summary = references(rows)
    b_summary = read(OLD_PUBLIC/"full_sahi_context.json")
    require(b_summary["checkpoint_sha256"] == a_summary["checkpoint_sha256"] and
            b_summary["config_sha256"] == executions["E"]["historical_config_sha256"], "B checkpoint/configuration differs")
    b, _ = load_predictions(OLD/"full/sahi_context/predictions.jsonl", rows, b_summary["prediction_file_sha256"])
    c_summary = read(ADAPTIVE_PUBLIC/"full_summary.json")
    active = (ADAPTIVE/"full"/read(ADAPTIVE/"full_active.json")["attempt"]).resolve()
    require(active.is_relative_to((ADAPTIVE/"full").resolve()) and sha(active/"bundles.jsonl") == c_summary["bundles_sha256"], "C artifact differs")
    require(c_summary["proposal_sha256"] == executions["F"]["historical_proposal_sha256"] and
        c_summary["execution_manifest_sha256"] == executions["F"]["historical_execution_sha256"], "C provenance differs")
    import json
    bundles = [json.loads(s) for s in (active/"bundles.jsonl").read_text().splitlines()]
    require([v["stem"] for v in bundles] == [r["filestem"] for r in rows], "C membership/order differs")
    c = {}
    for row, item in zip(rows, bundles):
        pred = np.asarray(item["variants"]["V2"]["pred"], dtype=np.float32).reshape(-1, 6)
        require(np.array_equal(pred, validate_predictions(pred, row["width"], row["height"])), "C geometry/order differs")
        require(np.array_equal(np.asarray(item["global"], np.float32).reshape(-1, 6), a[row["filestem"]]), "Historical C globals differ from A")
        c[row["filestem"]] = pred
    require(read(PREVIOUS/"verification.json")["passed"], "D has not passed verification")
    d_summary = read(PREVIOUS/"full/summary.json")
    require(d_summary["prediction_file_sha256"] == executions["F"]["D_predictions_sha256"], "D prediction provenance differs")
    d, _ = load_predictions(PREVIOUS/"full/predictions.jsonl", rows, executions["F"]["D_predictions_sha256"])
    linkage = read(PREVIOUS/"validation_linkage.json")
    require([v["stem"] for v in linkage] == [r["filestem"] for r in rows] and
            all(v["net_shape"] == shapes[v["stem"]] for v in linkage), "D geometry profile differs")
    metrics = {"A": a_summary["metrics"], "B": b_summary["metrics"],
        "C": read(ADAPTIVE_PUBLIC/"metrics_comparisons.json")["methods"]["V2"]["ap"],
        "D": read(ROOT/"docs/evidence/rarity_comparison/comparison_summary.json")["metrics"]["D"]}
    c_timing = read(ADAPTIVE_PUBLIC/"standalone_timing.json")["variants"]["V2"]
    sources = {
        "A": {"prediction_sha256": a_summary["prediction_file_sha256"], "report": "docs/evidence/error_analysis_sahi/full_standard_serial.json", "timing": a_summary["timing"], "mean_passes": 1},
        "B": {"prediction_sha256": b_summary["prediction_file_sha256"], "report": "docs/evidence/error_analysis_sahi/full_sahi_context.json", "timing": b_summary["timing"], "mean_passes": 5},
        "C": {"prediction_sha256": c_summary["bundles_sha256"], "report": "docs/evidence/adaptive_refinement_validation/metrics_comparisons.json", "timing": c_timing["timing"], "mean_passes": c_summary["mean_passes"]},
        "D": {"prediction_sha256": d_summary["prediction_file_sha256"], "report": "docs/evidence/rarity_comparison/comparison_summary.json", "timing": d_summary["timing"], "mean_passes": 1}}
    return {"A": a, "B": b, "C": c, "D": d}, shapes, metrics, sources


def protected_inventory():
    current = previous_inventory()
    expected = read(ROOT/"docs/evidence/rarity_comparison/preservation.json")["after"]
    require(identity(current) == expected["protected_private_inventory_sha256"], "Historical private evidence changed since D")
    for name in ("rarity_comparison", "rarity_comparison_checks"):
        current.update({p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/"outputs"/name).rglob("*")) if p.is_file()})
    return current


def preservation(directory, stage):
    snapshot = source_snapshot()
    require(identity(snapshot) == identity(read(OLD/"source_snapshot_before.json")), "Immutable data stat inventory changed")
    changed = subprocess.check_output(["git", "diff", START, "--name-only"], text=True).splitlines()
    allowed = ("src/wrist_inference_comparison/", "docs/evidence/rarity_inference_comparison/",
               "tests/test_rarity_inference.py")
    require(all(p == "README.md" or p in CONFIGS or p.startswith(allowed) for p in changed), "Protected tracked file changed")
    current = protected_inventory()
    if stage == "before":
        write_new(directory/"protected_inventory_before.json", current)
    else:
        require(current == read(directory/"protected_inventory_before.json"), "Prior experiment artifact changed")
    result = {"passed": True, "stage": stage, "protected_private_files": len(current),
        "protected_private_inventory_sha256": identity(current), "data_stat_entries": len(snapshot),
        "data_stat_inventory_sha256": identity(snapshot), "earlier_source_configs_reports_unchanged": True}
    write_new(directory/f"preservation_{stage}.json", result)
    return result
