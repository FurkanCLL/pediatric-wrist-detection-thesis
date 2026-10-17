"""Publish aggregate six-condition tables only after independent verification."""

import argparse
import csv
from pathlib import Path

from wrist_eval.common import CLASSES, identity, sha
from wrist_eval.adaptive_validation import assess_variant
from wrist_compare.analysis import compact_ap
from .analysis import CONTRASTS, contrast
from .common import (PUBLIC, read, write_new, require, study_path, frozen_study, manifests,
                     install_access_guard)

LABELS = {"A": ("standard", "standard"), "B": ("standard", "SAHI Context"),
    "C": ("standard", "CG-ALR V2"), "D": ("rarity-only", "standard"),
    "E": ("rarity-only", "SAHI Context"), "F": ("rarity-only", "CG-ALR V2")}


def scientific_gate(result, bootstrap, f_summary, cfg):
    # Reuse the original gate evaluator against D, including its contemporaneous required global call.
    delta = result["contrasts"]["F-D"]
    comparison = {"overall_ap_delta": delta["overall"], "per_class_ap_delta": {c: delta[c] for c in CLASSES}}
    metrics = {"methods": {name: {"ap": result["metrics"][c], "operating_points": result["operating_points"][c]}
        for name, c in (("V2", "F"), ("standard_serial", "D"))},
        "comparisons": {"V2_minus_standard_serial": comparison}}
    uncertainty = {"comparisons": {"V2_minus_standard_serial": bootstrap["contrasts"]["F-D"]}}
    timing = f_summary["timing"]
    return assess_variant("V2", metrics, uncertainty, timing["e2e_seconds"]["mean"],
        timing["paired_global_e2e_seconds"]["mean"], cfg["phase_3c2_plan"]["proposed_success"])


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact_points(points):
    return {name: {"confidence": value["confidence"], "iou": value["iou"],
        **{key: value[key] for key in ("total_tp", "total_fp", "total_fn")},
        "classes": {c: {key: item[key] for key in ("objects", "patients", "images", "tp", "fp", "fn", "precision", "recall")}
                    for c, item in value["classes"].items()}} for name, value in points.items()}


def export(directory):
    manifest = frozen_study(directory)
    verification = read(directory/"verification.json")
    require(verification["passed"], "Independent verification has not passed")
    result, bootstrap = read(directory/"analysis/comparison.json"), read(directory/"bootstrap/summary.json")
    summaries = {c: read(directory/f"full_{c}/summary.json") for c in ("E", "F")}
    _, cfg = manifests()
    gate = scientific_gate(result, bootstrap, summaries["F"], cfg)
    compact = {c: compact_ap(value) for c, value in result["metrics"].items()}
    require(result["contrasts"] == {name: contrast(compact, weights) for name, weights in CONTRASTS.items()},
            "Published AP differences differ from points")
    payload = {"scope": "Phase 4E; frozen validation only; no final method selection", "images": 3088, "patients": 914,
        "objects": 7204, "conditions": {c: {"training": LABELS[c][0], "inference": LABELS[c][1],
            "AP": compact[c], "operating_points": compact_points(result["operating_points"][c]), "source": result["sources"][c]}
            for c in LABELS}, "AP_contrasts": result["contrasts"], "bootstrap": bootstrap,
        "detection_changes": {name: {"recovery_by_iou": {iou: value["classes"] for iou, value in pair["recovery"].items()},
            "unmatched_prediction_correspondence": pair["false_positive_correspondence"]["classes"]}
            for name, pair in result["paired_detection_changes"].items()},
        "F_original_scientific_gates": gate, "F_usage": summaries["F"]["diagnostics"],
        "F_roi_count_images": summaries["F"]["roi_count_images"], "frozen_executions": manifest["executions"],
        "implementation_revision": manifest["implementation_revision"], "no_training_optimization_or_test": True,
        "cautions": ["Single training seed; validation-selected checkpoints and validation method development",
            "Nominal paired percentile intervals, no multiplicity correction or external clinical inference",
            "Overall AP requires all eight classes; absent-class AP stays undefined",
            "Historical timings come from separate sessions; no controlled speedup claim"]}
    write_new(PUBLIC/"results.json", payload)
    for condition in ("E", "F"):
        write_new(PUBLIC/f"full_{condition}.json", summaries[condition])
        write_new(PUBLIC/f"pilot_{condition}.json", read(directory/f"pilot_{condition}/summary.json"))
    write_new(PUBLIC/"verification.json", verification)
    write_new(PUBLIC/"preservation.json", {stage: read(directory/f"preservation_{stage}.json") for stage in ("before", "after")})
    matrix, classes, points, changes = [], [], [], []
    for c, (training, method) in LABELS.items():
        source = result["sources"][c]
        point = result["operating_points"][c]["conf0.25_iou0.5"]
        matrix.append({"experiment": c, "training": training, "inference": method, **compact[c]["overall"],
            **{k: point["total_"+k] for k in ("tp", "fp", "fn")},
            "precision": point["total_tp"]/(point["total_tp"]+point["total_fp"]), "recall": point["total_tp"]/7204,
            "mean_e2e_ms": source["timing"]["e2e_seconds"]["mean"]*1000,
            "mean_model_calls": source["mean_passes"], "prediction_artifact_sha256": source["prediction_sha256"],
            "source_report": source["report"]})
        for name in CLASSES:
            classes.append({"experiment": c, "class": name, **compact[c][name]})
        for name, value in result["operating_points"][c].items():
            for label, counts in value["classes"].items():
                points.append({"experiment": c, "operating_point": name, "class": label,
                    **{k: counts[k] for k in ("tp", "fp", "fn", "precision", "recall")}})
    for name, scopes in result["contrasts"].items():
        for scope, metrics in scopes.items():
            for metric, point in metrics.items():
                changes.append({"contrast": name, "scope": scope, "metric": metric, "point_delta": point,
                    **bootstrap["contrasts"][name][scope][metric]})
    for filename, values in (("six_condition_matrix.csv", matrix), ("per_class_AP.csv", classes),
                             ("operating_points.csv", points), ("paired_AP_contrasts.csv", changes)):
        write_csv(PUBLIC/filename, values)
    artifacts = {p.name: sha(p) for p in sorted(PUBLIC.iterdir()) if p.is_file() and p.suffix in (".json", ".csv")}
    write_new(PUBLIC/"publication_receipt.json", {"passed": True, "artifacts_sha256": artifacts,
        "matrix_rows": 6, "class_AP_rows": 48, "class_operating_rows": len(points), "AP_contrast_rows": len(changes),
        "private_patient_ids_images_boxes_and_predictions_excluded": True, "results_identity": identity(payload)})
    print("Aggregate E/F evidence and six-condition tables exported", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="study_01")
    args = parser.parse_args()
    install_access_guard()
    export(study_path(args.study))
