"""Independent output invariants and the approved accuracy/cost decision gates."""

import argparse
import hashlib
import json
import subprocess
import numpy as np

from .common import ROOT, save, sha, identity, validation_records
from .adaptive_config import approved_settings
from .adaptive_study import LOCAL, PUBLIC, source_identity, references
from .adaptive_evidence import active_records
from .adaptive import overlap
from .metrics import validate_predictions, distribution


def accounting():
    cfg, manifest = approved_settings()
    directory, records = active_records("full")
    summary = json.loads((PUBLIC/"full_summary.json").read_text())
    if sha(directory/"bundles.jsonl") != summary["bundles_sha256"] or summary["implementation_source_sha256"] != source_identity():
        raise ValueError("Full prediction bundle/source identity differs")
    if summary["execution_manifest_sha256"] != identity(manifest) or summary["proposal_sha256"] != identity(cfg):
        raise ValueError("Approved execution identity differs")
    rows = validation_records()
    if [r["stem"] for r in records] != [r["filestem"] for r in rows]:
        raise ValueError("Full prediction order/membership differs")
    serial, _ = references()
    physical_calls = 0
    for item in records:
        original = serial[item["stem"]]
        if not np.array_equal(np.asarray(item["global"], dtype=np.float32).reshape(-1, 6), original):
            raise ValueError("Global rows differ from preserved serial inference")
        h, w = item["plan"]["image_hw"]
        k = len(item["plan"]["rois"])
        calls = sum(v["counters"]["forward_passes"] for v in [item["global_view"], *item["local_views"]])
        if calls != 1+k or calls > 3 or calls != item["timing"]["passes"]:
            raise ValueError("Physical call budget differs")
        physical_calls += calls
        by_variant = {}
        for variant in ("V1", "V2"):
            result = item["variants"][variant]
            pred = np.asarray(result["pred"], dtype=np.float32).reshape(-1, 6)
            validate_predictions(pred, w, h)
            ids = result["diagnostics"]["output_global_ids"]
            if sorted(ids) != list(range(len(original))) or len(pred) != len(original):
                raise ValueError("Global membership changed")
            restored = pred[np.argsort(ids)]
            by_variant[variant] = restored
            mutable = {c["global_id"] for c in item["plan"]["candidates"]}
            protected = [i for i in range(len(original)) if i not in mutable]
            if not np.array_equal(restored[protected], original[protected]) or not np.array_equal(restored[:, 5], original[:, 5]):
                raise ValueError("Protected fields/class IDs changed")
            if not np.array_equal(pred[pred[:, 4] >= .5], original[original[:, 4] >= .5]):
                raise ValueError("Confidence-.50 invariant failed")
            matches = result["diagnostics"]["matches"]
            if len({p["global_id"] for p in matches}) != len(matches) or len({p["evidence_id"] for p in matches}) != len(matches):
                raise ValueError("Evidence assignment is not one-to-one")
            for i in mutable:
                if restored[i, 4] < original[i, 4] or float(restored[i, 4]) > min(float(original[i, 4])+.1, .35):
                    raise ValueError("Confidence bounds failed")
            if variant == "V1" and not np.array_equal(restored[:, :4], original[:, :4]):
                raise ValueError("Confirmation-only coordinates changed")
            if variant == "V2":
                for i in mutable:
                    if np.array_equal(restored[i, :4], original[i, :4]):
                        continue
                    for j in range(len(original)):
                        if j != i and original[j, 5] == original[i, 5] and overlap(restored[i, :4], restored[j, :4]) > max(
                                .7, overlap(original[i, :4], original[j, :4])):
                            raise ValueError("Final output creates/increases an above-.70 collision")
        if not np.array_equal(by_variant["V1"][:, 4:], by_variant["V2"][:, 4:]):
            raise ValueError("V1/V2 differ outside the intended geometry factor")
    result = {"images": len(records), "physical_full_run_model_calls": physical_calls,
        "same_view_bundle_for_both_variants": True, "all_global_predictions_exact": True,
        "all_outputs_finite_valid": True, "all_global_membership_and_protected_fields_preserved": True,
        "one_to_one_evidence_assignment": True, "score_caps_and_confidence_0_50_invariant": True,
        "v1_coordinates_unchanged": True, "v2_collision_bound": True,
        "v1_v2_scores_and_class_ids_identical": True, "no_training_or_test_evaluation": True,
        "full_bundle_sha256": summary["bundles_sha256"]}
    save(PUBLIC/"independent_accounting.json", result)
    print(result, flush=True)


def assess_variant(variant, metrics, bootstrap, timing, baseline_timing, limits):
    method = metrics["methods"][variant]
    baseline = metrics["methods"]["standard_serial"]
    diff = metrics["comparisons"][variant+"_minus_standard_serial"]
    ci = bootstrap["comparisons"][variant+"_minus_standard_serial"]
    point = method["operating_points"]["conf0.25_iou0.5"]
    base_point = baseline["operating_points"]["conf0.25_iou0.5"]
    fracture_point = all(diff["per_class_ap_delta"]["fracture"][k] >= 0 for k in ("ap50", "ap75", "ap50_95"))
    fracture_interval = all(ci["fracture"][k]["lower"] >= limits["fracture_ap_lower_interval_tolerance"] for k in ("ap50", "ap75", "ap50_95"))
    fp_ratio = point["total_fp"]/base_point["total_fp"]
    fracture_fp_ratio = point["classes"]["fracture"]["fp"]/base_point["classes"]["fracture"]["fp"]
    checks = {"overall_map50_95_positive": diff["overall_ap_delta"]["map50_95"] > 0,
        "overall_map50_95_interval_positive": ci["overall"]["map50_95"]["lower"] > 0,
        "overall_map50_nondecreasing": diff["overall_ap_delta"]["map50"] >= 0,
        "fracture_ap_points_preserved": fracture_point, "fracture_interval_tolerance": fracture_interval,
        "all_class_fp_ratio": fp_ratio <= limits["false_positive_count_ratio_max"],
        "fracture_fp_ratio": fracture_fp_ratio <= limits["false_positive_count_ratio_max"],
        "mean_latency_ratio": timing/baseline_timing <= limits["mean_latency_reference_ratio_max"]}
    primary = all(checks.values())
    preservation = fracture_point and fracture_interval and checks["all_class_fp_ratio"] and checks["fracture_fp_ratio"] and checks["mean_latency_ratio"]
    secondary = (diff["overall_ap_delta"]["map50_95"] >= 0 and ci["overall"]["map50_95"]["lower"] >= -.005
                 and preservation and any(ci[c]["ap50_95"]["lower"] > 0 for c in ("fracture", "periostealreaction")))
    return {"checks": checks, "primary_success": primary, "secondary_tradeoff_success": secondary,
            "passes": primary or secondary, "fp_count_ratio": fp_ratio, "fracture_fp_count_ratio": fracture_fp_ratio,
            "mean_latency_ratio": timing/baseline_timing}


def decisions():
    cfg, _ = approved_settings()
    metrics = json.loads((PUBLIC/"metrics_comparisons.json").read_text())
    bootstrap = json.loads((PUBLIC/"patient_bootstrap.json").read_text())
    summary = json.loads((PUBLIC/"full_summary.json").read_text())
    baseline = json.loads((PUBLIC.parent/"error_analysis_sahi/full_standard_serial.json").read_text())
    standalone = json.loads((PUBLIC/"standalone_timing.json").read_text())
    results = {v: assess_variant(v, metrics, bootstrap, standalone["variants"][v]["timing"]["e2e_seconds"]["mean"],
                                standalone["variants"][v]["timing"]["paired_global_e2e_seconds"]["mean"],
                                cfg["phase_3c2_plan"]["proposed_success"]) for v in ("V1", "V2")}
    for v in ("V1", "V2"):
        results[v]["historical_standard_latency_ratio"] = standalone["variants"][v]["timing"]["e2e_seconds"]["mean"]/baseline["timing"]["e2e_seconds"]["mean"]
    selection = None
    if results["V1"]["passes"]:
        selection = "V1"
        if results["V2"]["passes"] and bootstrap["comparisons"]["V2_minus_V1"]["overall"]["map50_95"]["lower"] > 0:
            selection = "V2"
    elif results["V2"]["passes"]:
        selection = "V2"
    save(PUBLIC/"decision_gates.json", {"variants": results, "recommended_variant": selection,
        "recommendation": "Retain standard accuracy/cost default; preserve negative/inconclusive adaptive findings" if selection is None else
                          f"Recommend {selection} under the predefined development gates; student review and final protocol selection remain separate",
        "timing_caveat": "Final timing uses complete standalone paths. The <=2x gate uses contemporaneous decode+verified-global-output wall time from the already-required global call, with identical baseline predictions and no extra model pass. Historical standard ratios remain descriptive.",
        "no_new_optimization_cycle": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["accounting", "decisions"])
    if parser.parse_args().action == "accounting":
        accounting()
    else:
        decisions()
