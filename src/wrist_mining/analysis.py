"""Training-only distributions and checkpoint sensitivity from saved predictions."""

import argparse
from collections import Counter
import csv
import json
import math

import numpy as np

from .common import (ROOT, LOCAL, PUBLIC, CLASSES, training_records, arrays, prediction_file,
                     stability_subset, identity, save)
from .difficulty import characterize, rank_correlation
from wrist_design.sampling import DesignPopulation
from wrist_eval.metrics import distribution

CLINICAL = (0, 3, 4, 6, 7)
SIZE_LABELS = ("<16", "16-<32", "32-<64", "64-<128", ">=128")


def predictions(checkpoint="best.pt"):
    path, summary = prediction_file(checkpoint)
    items = [json.loads(line) for line in path.read_text().splitlines()]
    result = {r["stem"]: np.asarray(r["pred"], dtype=np.float32).reshape(-1, 6) for r in items}
    if len(items) != len(result) or len(items) != summary["images"]:
        raise ValueError("Duplicate or missing training predictions")
    return result


def concentration(values):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if not total:
        return {"total": 0, "patients_nonzero": 0, "top_1_percent_share": 0., "top_10_percent_share": 0.}
    ranked = np.sort(values)
    n = len(values)
    return {"total": float(total), "patients_nonzero": int((values > 0).sum()),
        "top_1_percent_share": float(ranked[-max(1, math.ceil(n*.01)):].sum()/total),
        "top_10_percent_share": float(ranked[-max(1, math.ceil(n*.1)):].sum()/total),
        "max_patient_share": float(ranked[-1]/total), "effective_patients": float(total**2/(values**2).sum()),
        "gini": float((2*np.dot(np.arange(1, n+1), ranked)/(n*total))-(n+1)/n)}


def source_uncertainty(rows):
    selected = {r["filestem"] for r in rows}
    flags = {}
    with (ROOT/"data/raw/grazpedwri-dx/dataset.csv").open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            if record["filestem"] not in selected:
                continue
            value = record["diagnosis_uncertain"].strip().lower()
            if value not in ("", "0", "1", "false", "true", "0.0", "1.0"):
                raise ValueError("Unknown released training uncertainty flag")
            # A blank is only absence of a positive flag, not an assertion of diagnostic certainty.
            flags[record["filestem"]] = value in ("1", "true", "1.0")
    if set(flags) != selected:
        raise ValueError("Missing training image-level uncertainty metadata")
    return np.array([flags[r["filestem"]] for r in rows], dtype=bool)


def analyze():
    rows = training_records()
    pred = predictions()
    if set(pred) != {r["filestem"] for r in rows}:
        raise ValueError("Mining prediction membership differs from frozen training")
    counts, patients, patient_ids = arrays(rows)
    n = len(rows)
    per_class = {c: {"annotations": int(counts[:, i].sum()), "images": int((counts[:, i] > 0).sum()),
                    "patients": int(len(np.unique(patients[counts[:, i] > 0]))), "hard_positive_objects": 0,
                    "operating": {t: {"tp": 0, "fn": 0, "potential_fp": 0, "gt_reasons": Counter()} for t in ("0.5", "0.75")},
                    "potential_fp_categories": Counter(), "potential_fp_ge_0.50": 0} for i, c in enumerate(CLASSES)}
    scores = np.zeros((n, 8))
    hard = np.zeros((n, 8), dtype=np.int32)
    fp = np.zeros((n, 8), dtype=np.int32)
    size = {(c, s): Counter() for c in CLASSES for s in SIZE_LABELS}
    private = LOCAL/"analysis"
    private.mkdir(parents=True, exist_ok=True)
    with (private/"best_details.jsonl").open("w", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            values = pred[row["filestem"]]
            detail = characterize(row, values)
            scores[i] = detail["class_difficulty"]
            for j, obj in enumerate(detail["objects"]):
                c = CLASSES[obj["class_id"]]
                is_hard = obj["difficulty"] > 0
                hard[i, obj["class_id"]] += int(is_hard)
                per_class[c]["hard_positive_objects"] += int(is_hard)
                s = size[c, SIZE_LABELS[obj["size_bin"]]]
                s["annotations"] += 1
                s["hard_positive"] += int(is_hard)
                for threshold, point in detail["operating"].items():
                    matched = point["matched_gt"][j] >= 0
                    d = per_class[c]["operating"][threshold]
                    d["tp" if matched else "fn"] += 1
                    d["gt_reasons"][point["gt_reasons"][j]] += 1
                    s["tp_"+threshold] += int(matched)
            for threshold, point in detail["operating"].items():
                for j, box in enumerate(values):
                    if box[4] >= .25 and point["matched_pred"][j] < 0:
                        c = CLASSES[int(box[5])]
                        per_class[c]["operating"][threshold]["potential_fp"] += 1
                        if threshold == "0.5":
                            fp[i, int(box[5])] += 1
                            per_class[c]["potential_fp_categories"][point["potential_fp_reasons"][j]] += 1
                            per_class[c]["potential_fp_ge_0.50"] += int(box[4] >= .5)
            stream.write(json.dumps({"stem": row["filestem"], **detail}, allow_nan=False)+"\n")
            if i % 2000 == 0:
                print(f"Training difficulty: {i+1}/{n}", flush=True)
    specification = json.loads((ROOT/"configs/class_aware_candidates.json").read_text())
    population = DesignPopulation(rows, specification)
    _, weights = population.weights(specification["candidates"]["B"])
    rarity = weights/specification["candidates"]["B"]["weight_cap"]
    difficulty = scores[:, CLINICAL].max(1)
    difficulty[np.isin(patients, population.lesion_patients)] = 0
    uncertain = source_uncertainty(rows)
    np.savez_compressed(private/"features.npz", class_difficulty=scores, hard_counts=hard, potential_fp_counts=fp,
                        difficulty=difficulty, rarity=rarity, uncertain=uncertain)
    patient_count = len(patient_ids)
    p_hard = np.stack([np.bincount(patients, weights=hard[:, c], minlength=patient_count) for c in range(8)], axis=1)
    p_objects = np.stack([np.bincount(patients, weights=counts[:, c], minlength=patient_count) for c in range(8)], axis=1)
    p_fp = np.bincount(patients, weights=fp.sum(1), minlength=patient_count)
    patient_rows = []
    for p, patient_id in enumerate(patient_ids):
        members = patients == p
        patient_rows.append({"patient_id": patient_id, "images": int(members.sum()),
            "objects_by_class": p_objects[p].astype(int).tolist(), "hard_by_class": p_hard[p].astype(int).tolist(),
            "difficulty_mean": float(difficulty[members].mean()), "potential_fp": int(p_fp[p])})
    save(private/"patient_distributions.json", patient_rows)
    result = {"scope": "Training-only in-sample difficulty, not generalization or verified clinical errors",
        "images": n, "patients": patient_count, "annotations": int(counts.sum()),
        "classes": per_class, "patient_concentration": {c: {"hard_positive": concentration(p_hard[:, i]),
                "all_annotations": concentration(p_objects[:, i]),
                "hard_fraction_among_class_positive_patients": distribution(p_hard[p_objects[:, i] > 0, i]/p_objects[p_objects[:, i] > 0, i])}
                for i, c in enumerate(CLASSES)},
        "all_hard_positive_patient_concentration": concentration(p_hard.sum(1)),
        "clinical_hard_positive_patient_concentration": concentration(p_hard[:, CLINICAL].sum(1)),
        "all_annotation_patient_concentration": concentration(p_objects.sum(1)),
        "potential_fp_patient_concentration": concentration(p_fp),
        "patient_difficulty_mean": distribution([r["difficulty_mean"] for r in patient_rows]),
        "difficulty": {"formula": "max class-mean object score over 0,3,4,6,7; 16 bone-lesion patients excluded from bonus priority",
            "image_distribution": distribution(difficulty), "nonzero_images": int((difficulty > 0).sum()),
            "nonzero_patients": int(len(np.unique(patients[difficulty > 0]))),
            "spearman_with_rarity": rank_correlation(difficulty, rarity),
            "hard_images_without_rarity_priority": int(((difficulty > 0) & (rarity == 0)).sum()),
            "hard_objects_without_rarity_priority": hard[rarity == 0].sum(0).tolist(),
            "hard_objects_without_rarity_priority_in_error_eligible_images": hard[(rarity == 0) & (difficulty > 0)].sum(0).tolist(),
            "hard_images_with_rarity_priority": int(((difficulty > 0) & (rarity > 0)).sum())},
        "released_uncertainty_flag": {"all_training_images": int(uncertain.sum()),
            "unmarked_or_missing_images": int((~uncertain).sum()),
            "hard_images": int((uncertain & (difficulty > 0)).sum()),
            "potential_fp_images": int((uncertain & (fp.sum(1) > 0)).sum()),
            "interpretation": "Only 1 marks uncertainty; blank is unmarked/missing, not confirmed certainty. Image-level flags are not box-quality adjudication and never change sampling weights"},
        "potential_fp_images": int((fp.sum(1) > 0).sum()),
        "in_sample_limitation": {"training_overlap_fraction": 1., "patients_seen_during_fitting": patient_count,
            "selected_checkpoint_completed_epochs": 22, "nominal_prior_image_presentations": n*22,
            "checkpoint_selection_used_validation": True, "independent_out_of_fold_predictions": False,
            "training_diagnostic_ap_is_not_reported_as_generalization": True}}
    for c, value in per_class.items():
        value["hard_positive_fraction"] = value["hard_positive_objects"]/value["annotations"]
        for point in value["operating"].values():
            point["precision"] = point["tp"]/(point["tp"]+point["potential_fp"]) if point["tp"]+point["potential_fp"] else 0.
            point["recall"] = point["tp"]/value["annotations"]
            if point["tp"]+point["fn"] != value["annotations"]:
                raise ValueError("Training object accounting differs")
    save(PUBLIC/"training_mining.json", result)
    with (PUBLIC/"size_strata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["class", "short_768", "annotations", "hard_positive", "tp_0.5", "tp_0.75"])
        writer.writeheader()
        writer.writerows({"class": c, "short_768": s, **{k: v[k] for k in writer.fieldnames[2:]}} for (c, s), v in size.items())


def stability():
    training = training_records()
    rows = stability_subset(training)
    lesion_patients = {r["patient_id"] for r in training if any(b["class_id"] == 5 for b in r["boxes"])}
    versions = {name: predictions(name) for name in ("best.pt", "epoch010.pt", "epoch040.pt")}
    details = {name: [characterize(r, values[r["filestem"]]) for r in rows] for name, values in versions.items()}
    scores = {name: np.array([d["class_difficulty"] for d in ds])[:, CLINICAL].max(1) for name, ds in details.items()}
    for values in scores.values():
        values[[r["patient_id"] in lesion_patients for r in rows]] = 0
    ordering = sorted(range(len(rows)), key=lambda i: identity({"purpose": "stability_ties", "stem": rows[i]["filestem"]}))
    tops = {name: set(sorted(ordering, key=lambda i: -values[i])[:math.ceil(len(rows)*.1)]) for name, values in scores.items()}
    result = {"images": len(rows), "patients": len({r["patient_id"] for r in rows}),
        "sampling": "1024 hash-uniform plus up to32/class; not population-weighted; includes bone-lesion diagnostic cases",
        "same_training_trajectory_not_independent_models": True, "comparisons": {}}
    for name in ("epoch010.pt", "epoch040.pt"):
        base, other = scores["best.pt"], scores[name]
        a, b = base > 0, other > 0
        classes = {}
        for c in range(8):
            da, db = [], []
            for row, x, y in zip(rows, details["best.pt"], details[name]):
                for j, box in enumerate(row["boxes"]):
                    if box["class_id"] == c:
                        da.append(x["objects"][j]["difficulty"] > 0)
                        db.append(y["objects"][j]["difficulty"] > 0)
            da, db = np.asarray(da, dtype=bool), np.asarray(db, dtype=bool)
            classes[CLASSES[c]] = {"annotations": len(da), "best_hard": int(da.sum()), "other_hard": int(db.sum()),
                "hard_jaccard": float((da & db).sum()/(da | db).sum()) if (da | db).any() else None,
                "agreement": float((da == db).mean()) if len(da) else None}
        result["comparisons"][name+"_versus_best"] = {"spearman": rank_correlation(base, other),
            "positive_difficulty_jaccard": float((a & b).sum()/(a | b).sum()) if (a | b).any() else None,
            "top_decile_jaccard": len(tops[name] & tops["best.pt"])/len(tops[name] | tops["best.pt"]),
            "top_decile_ties": {v: int((s == np.sort(s)[-math.ceil(len(rows)*.1)]).sum()) for v, s in scores.items()},
            "classes": classes}
    save(PUBLIC/"checkpoint_stability.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["analyze", "stability"])
    if parser.parse_args().action == "analyze":
        analyze()
    else:
        stability()
