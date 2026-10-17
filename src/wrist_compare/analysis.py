"""Common AP, declared operating points and paired patient uncertainty from saved predictions."""

import argparse
from collections import Counter
import json
import time

import numpy as np

from wrist_eval.common import CLASSES, config, identity, sha
from wrist_eval.metrics import ap_summary, native_stats, match
from wrist_eval.analyze import truth, describe, compare, compare_false_positives
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from .common import (OLD, OLD_PUBLIC, read, require, write_new, frozen_study, references,
    load_predictions, validation_records, study_path, install_access_guard)
from .infer import assert_metrics


def inputs(directory):
    manifest = frozen_study(directory)
    rows = validation_records()
    a, _, a_summary = references(rows)
    d_summary = read(directory/"full/summary.json")
    require(d_summary["passed"] and d_summary["images"] == 3088 and d_summary["patients"] == 914 and
            d_summary["checkpoint_sha256"] == manifest["configuration"]["rarity_checkpoint_sha256"], "Full D evidence is incomplete")
    d, items = load_predictions(directory/"full/predictions.jsonl", rows, d_summary["prediction_file_sha256"])
    from .infer import verify_item
    for row, item in zip(rows, items):
        verify_item(row, d[row["filestem"]], item["timing"])
    return rows, {"A": a, "D": d}, {"A": a_summary, "D": d_summary}


def metric_delta(a, d):
    # Only jointly defined class metrics can produce a paired AP difference.
    result = {key: d[key]-a[key] for key in ("map50", "map75", "map50_95")}
    result["classes"] = {name: {key: d["classes"][name][key]-a["classes"][name][key]
        for key in ("ap50", "ap75", "ap50_95")} for name in CLASSES if name in a["classes"] and name in d["classes"]}
    return result


def operating_counts(rows, predictions):
    values = np.zeros((len(rows), 8, 3), dtype=np.int64)
    for index, row in enumerate(rows):
        gt, pred = truth(row), predictions[row["filestem"]]
        targets, assigned, _ = match(gt, pred, .5, .25)
        for c in range(8):
            values[index, c] = [(gt[:, 4] == c).sum(), ((gt[:, 4] == c) & (targets >= 0)).sum(),
                ((pred[:, 5] == c) & (assigned < 0) & (pred[:, 4] >= .25)).sum()]
    return values


def count_metrics(values):
    result = {}
    for name, (objects, tp, fp) in zip([*CLASSES, "overall"], np.vstack([values, values.sum(axis=0)])):
        result[name] = {"tp": int(tp), "fp": int(fp), "fn": int(objects-tp)}
        if objects:
            result[name]["recall"] = float(tp/objects)
        if tp+fp:
            result[name]["precision"] = float(tp/(tp+fp))
    return result


def compact_ap(value):
    result = {name: {key: value["classes"][name][key] for key in ("ap50", "ap75", "ap50_95")}
              for name in CLASSES if name in value["classes"]}
    # The primary estimand stays the same eight classes; do not silently average fewer classes.
    if len(value["classes"]) == 8:
        result["overall"] = {key: value[key] for key in ("map50", "map75", "map50_95")}
    return result


def paired_values(a, d):
    return {name: {key: d[name][key]-a[name][key] for key in a[name] if key in d[name]}
            for name in a if name in d}


def analyze(directory):
    destination = directory/"analysis"
    destination.mkdir(exist_ok=False)
    rows, predictions, summaries = inputs(directory)
    result = {"scope": "A versus D, frozen validation only", "images": len(rows), "patients": 914,
        "metrics": {}, "operating_points": {}, "prediction_artifacts": {c: s["prediction_file_sha256"] for c, s in summaries.items()}}
    for condition, pred in predictions.items():
        result["metrics"][condition] = ap_summary([native_stats(truth(r), pred[r["filestem"]]) for r in rows])
        points = {}
        for confidence in config()["operating_confidences"]:
            for threshold in config()["descriptive_iou"]:
                counts, details = describe(rows, pred, confidence, threshold)
                points[f"conf{confidence}_iou{threshold}"] = counts
                if confidence == .25 and threshold == .5:
                    write_new(destination/f"{condition}_details.json", details)
        result["operating_points"][condition] = points
    assert_metrics(result["metrics"]["A"], summaries["A"]["metrics"])
    result["metric_difference_D_minus_A"] = metric_delta(result["metrics"]["A"], result["metrics"]["D"])
    result["recovery"] = compare(rows, predictions["A"], predictions["D"])
    fp = compare_false_positives(rows, predictions["A"], predictions["D"])
    for values in fp["classes"].values():
        values["A_fp"] = values.pop("standard_fp")
        values["D_fp"] = values.pop("sahi_fp")
    result["false_positive_correspondence"] = fp
    result["timing"] = {c: s["timing"] for c, s in summaries.items()}
    write_new(destination/"comparison.json", result)
    print({"A": result["metrics"]["A"]["map50_95"], "D": result["metrics"]["D"]["map50_95"],
           "difference": result["metric_difference_D_minus_A"]["map50_95"]}, flush=True)


def bootstrap(directory):
    destination = directory/"bootstrap"
    destination.mkdir(exist_ok=False)
    rows, predictions, _ = inputs(directory)
    groups = patient_groups(rows)
    stats = {c: [native_stats(truth(r), pred[r["filestem"]]) for r in rows] for c, pred in predictions.items()}
    counts = {c: operating_counts(rows, pred) for c, pred in predictions.items()}
    rng = np.random.default_rng(314159)
    collected = {"A": {}, "D": {}, "D_minus_A": {}, "operating_D_minus_A": {}}
    present = Counter()
    started = time.perf_counter()
    with (destination/"replicates.jsonl").open("x", encoding="utf-8") as stream:
        for replicate in range(500):
            indices = resample_indices(groups, rng)
            sampled = {c: compact_ap(ap_summary([stats[c][i] for i in indices])) for c in ("A", "D")}
            operating = {c: count_metrics(counts[c][indices].sum(axis=0)) for c in ("A", "D")}
            require(sampled["A"].keys() == sampled["D"].keys(), "Paired class support differs")
            delta, op_delta = paired_values(sampled["A"], sampled["D"]), paired_values(operating["A"], operating["D"])
            item = {"replicate": replicate, "resampled_indices_sha256": identity(indices), "A": sampled["A"],
                "D": sampled["D"], "D_minus_A": delta, "operating_D_minus_A": op_delta}
            stream.write(json.dumps(item, allow_nan=False)+"\n")
            stream.flush()
            for category in collected:
                for name, values in item[category].items():
                    for metric, value in values.items():
                        collected[category].setdefault(name, {}).setdefault(metric, []).append(value)
            present.update(sampled["A"].keys())
            if replicate % 25 == 0:
                print(f"A/D paired patient bootstrap {replicate+1}/500 ({time.perf_counter()-started:.1f}s)", flush=True)
    result = {"replicates": 500, "seed": 314159, "patients": len(groups), "unit": "patient with all views/examinations",
        "method": "paired percentile 95% interval; conditional class presence; overall requires all eight classes",
        "class_presence_replicates": dict(present), "seconds": time.perf_counter()-started,
        "replicates_sha256": sha(destination/"replicates.jsonl"),
        "intervals": {category: {name: {metric: interval(values) for metric, values in metrics.items()}
            for name, metrics in classes.items()} for category, classes in collected.items()}}
    write_new(destination/"summary.json", result)


def verify(directory):
    rows, predictions, summaries = inputs(directory)
    result = read(directory/"analysis/comparison.json")
    counts = {c: operating_counts(rows, pred) for c, pred in predictions.items()}
    for condition, pred in predictions.items():
        aps = ap_summary([native_stats(truth(r), pred[r["filestem"]]) for r in rows])
        assert_metrics(aps, result["metrics"][condition])
        point = result["operating_points"][condition]["conf0.25_iou0.5"]
        measured = count_metrics(counts[condition].sum(axis=0))
        require(all(measured["overall"][key] == point["total_"+key] for key in ("tp", "fp", "fn")), "Independent operating totals differ")
        for name in CLASSES:
            require(all(measured[name][key] == point["classes"][name][key] for key in ("tp", "fp", "fn")), "Per-class operating totals differ")
            for threshold in ("0.5", "0.75"):
                partition = result["recovery"][threshold]["classes"][name]
                require(sum(partition.values()) == point["classes"][name]["objects"], "Recovery does not partition annotations")
                other_point = result["operating_points"][condition][f"conf0.25_iou{threshold}"]["classes"][name]
                require(partition["both"]+partition["lost" if condition == "A" else "recovered"] == other_point["tp"],
                        "Recovery does not reconcile to each detector's TP")
    bootstrap_summary = read(directory/"bootstrap/summary.json")
    require(sha(directory/"bootstrap/replicates.jsonl") == bootstrap_summary["replicates_sha256"], "Bootstrap file changed")
    replicates = [json.loads(s) for s in (directory/"bootstrap/replicates.jsonl").read_text().splitlines()]
    require([r["replicate"] for r in replicates] == list(range(500)), "Bootstrap count/order differs")
    groups, rng = patient_groups(rows), np.random.default_rng(314159)
    stats = {c: [native_stats(truth(r), pred[r["filestem"]]) for r in rows] for c, pred in predictions.items()}
    support = np.array([np.bincount(truth(r)[:, 4].astype(int), minlength=8) for r in rows])
    collected = {k: {} for k in ("A", "D", "D_minus_A", "operating_D_minus_A")}
    for i, item in enumerate(replicates):
        indices = resample_indices(groups, rng)
        require(item["resampled_indices_sha256"] == identity(indices), "Patient resampling sequence differs")
        names = {CLASSES[c] for c in np.flatnonzero(support[indices].sum(axis=0))}
        if len(names) == 8:
            names.add("overall")
        require(set(item["A"]) == set(item["D"]) == names, "Absent class treated as defined AP")
        require(item["D_minus_A"] == paired_values(item["A"], item["D"]), "Paired AP difference differs")
        op = {c: count_metrics(counts[c][indices].sum(axis=0)) for c in ("A", "D")}
        require(item["operating_D_minus_A"] == paired_values(op["A"], op["D"]), "Bootstrap operating difference differs")
        # Recompute bounded representative AP replicates; all 500 memberships/deltas/intervals are checked.
        if i in (0, 1, 249, 499):
            for c in ("A", "D"):
                require(item[c] == compact_ap(ap_summary([stats[c][j] for j in indices])), "Bootstrap AP replay differs")
        for category in collected:
            for name, metrics in item[category].items():
                for metric, value in metrics.items():
                    require(np.isfinite(value), "Non-finite bootstrap metric")
                    collected[category].setdefault(name, {}).setdefault(metric, []).append(value)
    intervals = {category: {name: {metric: interval(values) for metric, values in metrics.items()}
        for name, metrics in classes.items()} for category, classes in collected.items()}
    require(intervals == bootstrap_summary["intervals"], "Published bootstrap intervals differ")
    from .common import preservation
    preserved = preservation(directory, "after")
    write_new(directory/"verification.json", {"passed": True, "images_per_condition": 3088, "patients": 914,
        "prediction_membership_geometry_and_timing_passed": True, "AP_recomputed_both_conditions": True,
        "operating_counts_independently_reconciled": True, "recovery_partitions_reconciled": True,
        "bootstrap_memberships_differences_and_intervals_checked": 500, "AP_bootstrap_replicates_recomputed": [0, 1, 249, 499],
        "preservation_after": preserved, "new_A_GPU_inference": False, "test_evaluation": False})
    print("A/D final verification passes, including paired uncertainty and preservation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "bootstrap", "verify"))
    parser.add_argument("--study", default="study_01")
    args = parser.parse_args()
    install_access_guard()
    {"analyze": analyze, "bootstrap": bootstrap, "verify": verify}[args.stage](study_path(args.study))


if __name__ == "__main__":
    main()
