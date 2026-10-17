"""Saved-prediction comparisons and paired patient uncertainty; no GPU inference."""

import argparse
from collections import Counter
import csv
import json
import numpy as np

from .common import CLASSES, save, validation_records
from .adaptive_study import LOCAL, PUBLIC
from .adaptive_evidence import active_records
from .analyze import load_predictions, truth, describe, compare, compare_false_positives
from .metrics import ap_summary, native_stats, match, distribution
from .bootstrap import patient_groups, resample_indices, interval

METHODS = ("standard_serial", "sahi_context", "V1", "V2")
PAIRS = (("V1", "standard_serial"), ("V2", "standard_serial"), ("V1", "sahi_context"),
         ("V2", "sahi_context"), ("V2", "V1"), ("sahi_context", "standard_serial"))


def predictions():
    directory, records = active_records("full")
    result = {method: load_predictions(method) for method in METHODS[:2]}
    for method in METHODS[2:]:
        result[method] = {r["stem"]: np.asarray(r["variants"][method]["pred"], dtype=np.float32).reshape(-1, 6) for r in records}
    return result, records


def point_counts(gt, pred):
    g, p, _ = match(gt, pred)
    fracture_gt, fracture_pred = gt[:, 4] == 0, pred[:, 5] == 0
    return [int(sum(g >= 0)), int(sum((p < 0) & (pred[:, 4] >= .25))), int(sum(g < 0)),
            int(sum((g >= 0) & fracture_gt)), int(sum((p < 0) & (pred[:, 4] >= .25) & fracture_pred)),
            int(sum((g < 0) & fracture_gt))]


def operating_values(counts, images):
    tp, fp, fn, ft, ff, fm = map(float, counts)
    return {"recall": tp/(tp+fn) if tp+fn else 0., "precision": tp/(tp+fp) if tp+fp else 0.,
            "fp_per_image": fp/images, "fracture_recall": ft/(ft+fm) if ft+fm else 0.,
            "fracture_precision": ft/(ft+ff) if ft+ff else 0., "fracture_fp_per_image": ff/images}


def analyze():
    rows = validation_records()
    methods, records = predictions()
    previous = json.loads((PUBLIC.parent/"error_analysis_sahi/error_analysis.json").read_text())
    result = {"scope": "Validation/development; approved two initial variants, no tuning or test evaluation",
              "images": len(rows), "patients": 914, "methods": {}, "comparisons": {}}
    stats = {}
    counts = {}
    for method in METHODS:
        pred = methods[method]
        if set(pred) != {r["filestem"] for r in rows}:
            raise ValueError("Prediction membership differs")
        stats[method] = [native_stats(truth(row), pred[row["filestem"]]) for row in rows]
        ap = ap_summary(stats[method])
        if method in METHODS[:2]:
            if ap != previous["variants"][method]["ap"]:
                raise ValueError("Preserved comparator AP does not reproduce")
            points = previous["variants"][method]["operating_points"]
        else:
            points = {}
            for confidence in (.05, .25, .5):
                for threshold in (.5, .75):
                    key = f"conf{confidence}_iou{threshold}"
                    point, details = describe(rows, pred, confidence, threshold)
                    save(LOCAL/"analysis"/f"{method}_{key}_details.json", details)
                    save(LOCAL/"analysis"/f"{method}_{key}_strata.json", point["strata"])
                    points[key] = point
        result["methods"][method] = {"ap": ap, "operating_points": {
            key: {k: v for k, v in value.items() if k != "strata"} for key, value in points.items()}}
        counts[method] = [point_counts(truth(row), pred[row["filestem"]]) for row in rows]
        print({"method": method, "map50_95": ap["map50_95"]}, flush=True)
    for other, base in PAIRS:
        key = other+"_minus_"+base
        result["comparisons"][key] = {"recovery": compare(rows, methods[base], methods[other]),
            "false_positive_correspondence": compare_false_positives(rows, methods[base], methods[other]),
            "overall_ap_delta": {k: result["methods"][other]["ap"][k]-result["methods"][base]["ap"][k]
                                 for k in ("map50", "map75", "map50_95")},
            "per_class_ap_delta": {c: {k: result["methods"][other]["ap"]["classes"][c][k]-result["methods"][base]["ap"]["classes"][c][k]
                                       for k in ("ap50", "ap75", "ap50_95")} for c in CLASSES}}
    save(PUBLIC/"metrics_comparisons.json", result)
    save(LOCAL/"native_stats.json", {m: [{k: v.tolist() for k, v in s.items()} for s in values] for m, values in stats.items()})
    save(LOCAL/"primary_image_counts.json", counts)
    with (PUBLIC/"class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["method", "class", "ap50", "ap75", "ap50_95", "tp", "fp", "fn", "precision", "recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, value in result["methods"].items():
            for c in CLASSES:
                point = value["operating_points"]["conf0.25_iou0.5"]["classes"][c]
                writer.writerow({"method": method, "class": c,
                    **{k: value["ap"]["classes"][c][k] for k in ("ap50", "ap75", "ap50_95")},
                    **{k: point[k] for k in ("tp", "fp", "fn", "precision", "recall")}})
    diagnostics(records, methods, rows)


def diagnostics(records, methods, rows):
    by_stem = {r["filestem"]: r for r in rows}
    totals = {v: Counter() for v in ("V1", "V2")}
    roi = Counter()
    candidates, decisions, exclusions, local_filters = Counter(), Counter(), Counter(), Counter()
    stage_values = {key: [] for key in ("planning_seconds", "decode_seconds", "common_seconds", "gpu_forward_ms",
                                      "preprocess_seconds", "nms_restore_seconds")}
    changed_images = {v: set() for v in ("V1", "V2")}
    for item in records:
        plan = item["plan"]
        roi[len(plan["rois"]) ] += 1
        for c in plan["candidates"]:
            candidates[CLASSES[c["class_id"]]] += 1
            if c["geometry_rejection"]:
                exclusions[c["geometry_rejection"]] += 1
        decisions.update(d["action"] for d in plan["selection_decisions"])
        exclusions.update(d["reason"] for d in plan["excluded"])
        counters = [item["global_view"]["counters"], *[v["counters"] for v in item["local_views"]]]
        for key in stage_values:
            stage_values[key].append(item["timing"][key] if key in item["timing"] else sum(c[key] for c in counters))
        local_filters.update(d["reason"] for d in item["variants"]["V1"]["diagnostics"]["local_decisions"])
        gt = truth(by_stem[item["stem"]])
        for variant in ("V1", "V2"):
            d = item["variants"][variant]["diagnostics"]
            values = methods[variant][item["stem"]]
            _, matches, _ = match(gt, values)
            inverse = {original: i for i, original in enumerate(d["output_global_ids"])}
            counts = totals[variant]
            counts["images"] += 1
            counts["matched_mutable"] += sum(p["mutable"] for p in d["matches"])
            counts["protected_matches"] += d["protected_matches"]
            counts["score_updates"] += len(d["score_changes"])
            if d["score_changes"]:
                counts["images_with_score_update"] += 1
                changed_images[variant].add(item["stem"])
            for change in d["score_changes"]:
                counts["score_before_below_0.15"] += int(change["before"] < .15)
                if change["before"] < .25 <= change["after"]:
                    counts["crossed_0.25"] += 1
                    counts["crossed_0.25_tp" if matches[inverse[change["global_id"]]] >= 0 else "crossed_0.25_fp"] += 1
            counts.update("geometry_"+g["reason"] for g in d["geometry"])
            if any(g["reason"] == "accepted" for g in d["geometry"]):
                counts["images_with_geometry_update"] += 1
                changed_images[variant].add(item["stem"])
    nontarget = {}
    for variant in ("V1", "V2"):
        totals[variant]["images_with_any_update"] = len(changed_images[variant])
        for class_id in (1, 2, 4, 5, 6, 7):
            nontarget[f"{variant}_{CLASSES[class_id]}"] = all(np.array_equal(
                methods["standard_serial"][row["filestem"]][methods["standard_serial"][row["filestem"]][:, 5] == class_id],
                methods[variant][row["filestem"]][methods[variant][row["filestem"]][:, 5] == class_id]) for row in rows)
    save(PUBLIC/"utilization_and_ranking.json", {"roi_count_images": {str(i): roi[i] for i in range(3)},
        "eligible_anchor_counts": candidates, "selection_decisions": decisions, "candidate_exclusions": exclusions,
        "local_evidence_decisions": local_filters, "variants": totals,
        "nontarget_prediction_fields_and_within_class_order_unchanged": nontarget,
        "timing_stages": {k: distribution(v) for k, v in stage_values.items()}})


def bootstrap():
    rows = validation_records()
    groups = patient_groups(rows)
    stored = json.loads((LOCAL/"native_stats.json").read_text())
    stats = {m: [{k: np.asarray(v, dtype=bool if k == "tp" else np.float32).reshape((-1, 10) if k == "tp" else (-1,))
                 for k, v in s.items()} for s in items]
             for m, items in stored.items()}
    counts = {m: np.asarray(v, dtype=np.int64) for m, v in json.loads((LOCAL/"primary_image_counts.json").read_text()).items()}
    rng = np.random.default_rng(314159)
    samples = []
    for rep in range(500):
        indices = resample_indices(groups, rng)
        sample = {}
        for method in METHODS:
            ap = ap_summary([stats[method][i] for i in indices])
            sample[method] = {c: {k: ap["classes"][c][k] for k in ("ap50", "ap75", "ap50_95")}
                              for c in CLASSES if c in ap["classes"]}
            if len(ap["classes"]) == 8:
                sample[method]["overall"] = {k: ap[k] for k in ("map50", "map75", "map50_95")}
            sample[method]["operating"] = operating_values(counts[method][indices].sum(0), len(indices))
        samples.append(sample)
        if rep % 50 == 0:
            print(f"Adaptive paired patient bootstrap {rep+1}/500", flush=True)
    save(LOCAL/"bootstrap_replicates.json", samples)
    result = {"patients": 914, "replicates": 500, "seed": 314159, "interval": .95,
              "method": "paired percentile patient clusters, all views/examinations, fixed checkpoint, conditional on class presence",
              "comparisons": {}}
    for other, base in PAIRS:
        pair = {}
        for c in [*CLASSES, "overall", "operating"]:
            keys = ("map50", "map75", "map50_95") if c == "overall" else (
                tuple(operating_values([1, 1, 1, 1, 1, 1], 1)) if c == "operating" else ("ap50", "ap75", "ap50_95"))
            pair[c] = {k: interval([s[other][c][k]-s[base][c][k] for s in samples if c in s[other] and c in s[base]]) for k in keys}
        result["comparisons"][other+"_minus_"+base] = pair
    save(PUBLIC/"patient_bootstrap.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["analyze", "bootstrap"])
    if parser.parse_args().action == "analyze":
        analyze()
    else:
        bootstrap()
