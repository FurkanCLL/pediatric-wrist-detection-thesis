"""Six-condition saved-output accounting and paired inference-effect contrasts."""

import argparse
from collections import Counter
import json
import time

import numpy as np

from wrist_eval.common import CLASSES, config, identity, sha
from wrist_eval.metrics import ap_summary, native_stats
from wrist_eval.analyze import truth, describe, compare, compare_false_positives
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_compare.analysis import compact_ap, paired_values, operating_counts, count_metrics
from wrist_compare.infer import assert_metrics
from .common import (ROOT, OLD, OLD_PUBLIC, ADAPTIVE, ADAPTIVE_PUBLIC, PREVIOUS, read, write_new,
    require, historical_inputs, load_predictions, validation_records, frozen_study, study_path, install_access_guard, preservation)
from .checks import verify_f, valid_timing

CONTRASTS = {"E-D": {"E": 1, "D": -1}, "F-D": {"F": 1, "D": -1},
    "E-B": {"E": 1, "B": -1}, "F-C": {"F": 1, "C": -1},
    "(E-D)-(B-A)": {"E": 1, "D": -1, "B": -1, "A": 1},
    "(F-D)-(C-A)": {"F": 1, "D": -1, "C": -1, "A": 1}}


def contrast(values, weights):
    # Calculate the interaction within each paired replicate, never from interval endpoints.
    names = set.intersection(*(set(values[c]) for c in weights))
    return {name: {key: sum(weight*values[c][name][key] for c, weight in weights.items())
        for key in set.intersection(*(set(values[c][name]) for c in weights))} for name in sorted(names)}


def inputs(directory):
    frozen_study(directory)
    rows = validation_records()
    predictions, _, previous_metrics, sources = historical_inputs(rows)
    for condition in ("E", "F"):
        summary = read(directory/f"full_{condition}/summary.json")
        require(summary["passed"] and summary["images"] == 3088 and summary["patients"] == 914 and
            not (directory/f"full_{condition}/failure.json").exists(), "Required full experiment incomplete")
        predictions[condition], items = load_predictions(directory/f"full_{condition}/predictions.jsonl", rows,
                                                        summary["prediction_file_sha256"])
        for item in items:
            calls = 5 if condition == "E" else item["timing"]["forward_passes"]
            valid_timing(item["timing"], calls)
            require(calls == 5 if condition == "E" else 1 <= calls <= 3, "Physical call budget differs")
        sources[condition] = {"prediction_sha256": summary["prediction_file_sha256"],
            "report": f"full_{condition}.json", "timing": summary["timing"],
            "mean_passes": summary["physical_model_calls"]/3088}
    return rows, predictions, previous_metrics, sources


def analyze(directory):
    destination = directory/"analysis"
    destination.mkdir(exist_ok=False)
    rows, predictions, expected, sources = inputs(directory)
    previous_ab = read(OLD_PUBLIC/"error_analysis.json")["variants"]
    previous_c = read(ADAPTIVE_PUBLIC/"metrics_comparisons.json")["methods"]["V2"]
    previous_d = read(PREVIOUS/"analysis/comparison.json")["operating_points"]["D"]
    points = {"A": previous_ab["standard_serial"]["operating_points"],
        "B": previous_ab["sahi_context"]["operating_points"], "C": previous_c["operating_points"], "D": previous_d}
    metrics = {}
    for condition, pred in predictions.items():
        metrics[condition] = ap_summary([native_stats(truth(r), pred[r["filestem"]]) for r in rows])
        if condition in expected:
            assert_metrics(metrics[condition], expected[condition])
        else:
            points[condition] = {}
            for confidence in config()["operating_confidences"]:
                for threshold in config()["descriptive_iou"]:
                    value, details = describe(rows, pred, confidence, threshold)
                    points[condition][f"conf{confidence}_iou{threshold}"] = value
                    if confidence == .25 and threshold == .5:
                        write_new(destination/f"{condition}_details.json", details)
        print(f"Saved-output metrics and operating points: {condition}", flush=True)
    compact = {c: compact_ap(v) for c, v in metrics.items()}
    pairs = {}
    for other, base in (("E", "D"), ("F", "D"), ("E", "B"), ("F", "C")):
        fp = compare_false_positives(rows, predictions[base], predictions[other])
        for counts in fp["classes"].values():
            counts["base_fp"], counts["other_fp"] = counts.pop("standard_fp"), counts.pop("sahi_fp")
        pairs[other+"-"+base] = {"recovery": compare(rows, predictions[base], predictions[other]), "false_positive_correspondence": fp}
    write_new(destination/"comparison.json", {"metrics": metrics, "operating_points": points,
        "contrasts": {name: contrast(compact, weights) for name, weights in CONTRASTS.items()},
        "paired_detection_changes": pairs, "sources": sources, "images": 3088, "patients": 914})
    print({c: v["map50_95"] for c, v in metrics.items()}, flush=True)


def summaries(collected):
    return {name: {scope: {metric: interval(values) for metric, values in metrics.items()}
        for scope, metrics in scopes.items()} for name, scopes in collected.items()}


def collect(destination, values):
    for name, scopes in values.items():
        for scope, metrics in scopes.items():
            for metric, value in metrics.items():
                require(np.isfinite(value), "Non-finite bootstrap value")
                destination.setdefault(name, {}).setdefault(scope, {}).setdefault(metric, []).append(value)


def historical_bootstraps():
    ab = read(OLD/"bootstrap_replicates.json")
    c = read(ADAPTIVE/"bootstrap_replicates.json")
    d = [json.loads(s) for s in (PREVIOUS/"bootstrap/replicates.jsonl").read_text().splitlines()]
    require(len(ab) == len(c) == len(d) == 500, "Historical bootstrap length differs")
    return [{"A": v["standard_serial"], "B": v["sahi_context"],
        "C": {k: value for k, value in c[i]["V2"].items() if k != "operating"}, "D": d[i]["D"],
        "indices_sha256": d[i]["resampled_indices_sha256"]} for i, v in enumerate(ab)]


def bootstrap(directory):
    destination = directory/"bootstrap"
    destination.mkdir(exist_ok=False)
    rows, predictions, _, _ = inputs(directory)
    groups, rng = patient_groups(rows), np.random.default_rng(314159)
    stats = {c: [native_stats(truth(r), pred[r["filestem"]]) for r in rows] for c, pred in predictions.items()}
    counts = {c: operating_counts(rows, pred) for c, pred in predictions.items()}
    prior = historical_bootstraps()
    collected, collected_operating, present = {}, {}, Counter()
    started = time.perf_counter()
    with (destination/"replicates.jsonl").open("x", encoding="utf-8") as stream:
        for rep in range(500):
            indices = resample_indices(groups, rng)
            require(identity(indices) == prior[rep]["indices_sha256"], "Paired historical patient draws differ")
            sampled = {c: compact_ap(ap_summary([stats[c][i] for i in indices])) for c in predictions}
            require(all(sampled[c] == prior[rep][c] for c in "ABCD"), "Historical A/B/C/D bootstrap does not reproduce")
            operating = {c: count_metrics(counts[c][indices].sum(axis=0)) for c in predictions}
            changes = {name: contrast(sampled, weights) for name, weights in CONTRASTS.items()}
            op_changes = {name: contrast(operating, weights) for name, weights in CONTRASTS.items()}
            stream.write(json.dumps({"replicate": rep, "indices_sha256": identity(indices), "metrics": sampled,
                "contrasts": changes, "operating_contrasts": op_changes}, allow_nan=False)+"\n")
            stream.flush()
            collect(collected, changes)
            collect(collected_operating, op_changes)
            present.update(sampled["A"].keys())
            if rep % 25 == 0:
                print(f"Six-condition paired patient bootstrap {rep+1}/500 ({time.perf_counter()-started:.1f}s)", flush=True)
    write_new(destination/"summary.json", {"replicates": 500, "seed": 314159, "patients": len(groups),
        "unit": "patient, all views/examinations together", "method": "paired percentile 95%; conditional class presence",
        "class_presence_replicates": dict(present), "contrasts": summaries(collected),
        "operating_contrasts": summaries(collected_operating), "all_2000_historical_ABCD_AP_replicates_exact": True,
        "replicates_sha256": sha(destination/"replicates.jsonl"), "seconds": time.perf_counter()-started})


def verify(directory):
    rows, predictions, expected, _ = inputs(directory)
    result = read(directory/"analysis/comparison.json")
    counts = {c: operating_counts(rows, pred) for c, pred in predictions.items()}
    for condition, pred in predictions.items():
        actual = ap_summary([native_stats(truth(r), pred[r["filestem"]]) for r in rows])
        assert_metrics(actual, result["metrics"][condition])
        measured = count_metrics(counts[condition].sum(axis=0))
        point = result["operating_points"][condition]["conf0.25_iou0.5"]
        require(all(measured["overall"][k] == point["total_"+k] for k in ("tp", "fp", "fn")), "Primary count totals differ")
        require(all(measured[c][k] == point["classes"][c][k] for c in CLASSES for k in ("tp", "fp", "fn")), "Class counts differ")
    for name, pair in result["paired_detection_changes"].items():
        other, base = name.split("-")
        for threshold in ("0.5", "0.75"):
            for c in CLASSES:
                p = pair["recovery"][threshold]["classes"][c]
                base_point = result["operating_points"][base][f"conf0.25_iou{threshold}"]["classes"][c]
                other_point = result["operating_points"][other][f"conf0.25_iou{threshold}"]["classes"][c]
                require(sum(p.values()) == base_point["objects"] and p["both"]+p["lost"] == base_point["tp"] and
                        p["both"]+p["recovered"] == other_point["tp"], "Recovery accounting differs")
        for p in pair["false_positive_correspondence"]["classes"].values():
            require(p["shared_fp"]+p["removed_fp"] == p["base_fp"] and
                    p["shared_fp"]+p["introduced_fp"] == p["other_fp"], "Unmatched-prediction correspondence differs")
    from .common import manifests
    _, cfg = manifests()
    full_f = read(directory/"full_F/summary.json")
    require(sha(directory/"full_F/bundles.jsonl") == full_f["bundles_sha256"], "F bundle hash differs")
    bundles = [json.loads(s) for s in (directory/"full_F/bundles.jsonl").read_text().splitlines()]
    require([v["stem"] for v in bundles] == [r["filestem"] for r in rows], "F bundle membership differs")
    totals = Counter()
    for row, item in zip(rows, bundles):
        totals.update(verify_f(item, predictions["D"][row["filestem"]], cfg))
    require(dict(totals) == full_f["diagnostics"], "F usage diagnostics differ")
    summary = read(directory/"bootstrap/summary.json")
    require(sha(directory/"bootstrap/replicates.jsonl") == summary["replicates_sha256"], "Bootstrap artifact changed")
    values = [json.loads(s) for s in (directory/"bootstrap/replicates.jsonl").read_text().splitlines()]
    require([v["replicate"] for v in values] == list(range(500)), "Bootstrap ordering/count differs")
    groups, rng, stored = patient_groups(rows), np.random.default_rng(314159), historical_bootstraps()
    support = np.array([np.bincount(truth(r)[:, 4].astype(int), minlength=8) for r in rows])
    stats = {c: [native_stats(truth(r), pred[r["filestem"]]) for r in rows] for c, pred in predictions.items()}
    collected, collected_op = {}, {}
    for rep, item in enumerate(values):
        indices = resample_indices(groups, rng)
        require(item["indices_sha256"] == identity(indices) == stored[rep]["indices_sha256"], "Patient sample differs")
        names = {CLASSES[c] for c in np.flatnonzero(support[indices].sum(axis=0))}
        if len(names) == 8:
            names.add("overall")
        require(all(set(v) == names for v in item["metrics"].values()), "Absent-class handling differs")
        require(all(item["metrics"][c] == stored[rep][c] for c in "ABCD"), "Historical AP replicate changed")
        changes = {name: contrast(item["metrics"], weights) for name, weights in CONTRASTS.items()}
        operating = {c: count_metrics(counts[c][indices].sum(axis=0)) for c in predictions}
        op = {name: contrast(operating, weights) for name, weights in CONTRASTS.items()}
        require(changes == item["contrasts"] and op == item["operating_contrasts"], "Paired/interaction differences differ")
        if rep in (0, 1, 249, 499):
            require(all(compact_ap(ap_summary([stats[c][i] for i in indices])) == item["metrics"][c]
                        for c in predictions), "Bootstrap AP replay differs")
        collect(collected, changes)
        collect(collected_op, op)
    require(summaries(collected) == summary["contrasts"] and summaries(collected_op) == summary["operating_contrasts"],
            "Bootstrap intervals differ")
    preserved = preservation(directory, "after")
    write_new(directory/"verification.json", {"passed": True, "images_per_condition": 3088, "patients": 914,
        "AP_recomputed_conditions": list(predictions), "all_3088_F_global_rows_bitwise_equal_D": True,
        "all_3088_F_fusion_invariants_and_offline_replays_pass": True, "all_2000_historical_ABCD_AP_replicates_exact": True,
        "bootstrap_memberships_contrasts_interactions_intervals_checked": 500,
        "AP_bootstrap_recomputed_replicates": [0, 1, 249, 499], "preservation_after": preserved,
        "no_training_optimization_or_test": True})
    print("E/F final verification passes, including six-condition paired contrasts and preservation", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "bootstrap", "verify"))
    parser.add_argument("--study", default="study_01")
    args = parser.parse_args()
    install_access_guard()
    {"analyze": analyze, "bootstrap": bootstrap, "verify": verify}[args.stage](study_path(args.study))
