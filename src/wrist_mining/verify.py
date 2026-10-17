"""Independent checks of saved training predictions and every simulated epoch."""

import csv
import hashlib
import json
import math
from collections import Counter

import numpy as np

from .common import ROOT, LOCAL, PUBLIC, CLASSES, training_records, arrays, identity, sha, save, stability_subset, prediction_file
from .difficulty import characterize
from .analysis import predictions
from .inference import source_identity
from wrist_eval.metrics import distribution, validate_predictions


def run():
    # Bind saved predictions and features to the sealed training population and sources.
    rows = training_records()
    counts, patient, patient_ids = arrays(rows)
    features = dict(np.load(LOCAL/"analysis/features.npz"))
    proposal = json.loads((ROOT/"configs/error_guided_sampler_proposed.json").read_text())
    if identity({k: features[k].tolist() for k in sorted(features)}) != proposal["feature_content_sha256"]:
        raise ValueError("Difficulty feature content changed")
    for checkpoint in ("best.pt", "epoch010.pt", "epoch040.pt"):
        values = predictions(checkpoint)
        expected = rows if checkpoint == "best.pt" else stability_subset(rows)
        if set(values) != {r["filestem"] for r in expected}:
            raise ValueError("Training/checkpoint subset membership changed")
        _, inference = prediction_file(checkpoint)
        if inference["normalized_source_sha256"] != source_identity():
            raise ValueError("Inference implementation changed")
        for row in expected:
            validate_predictions(values[row["filestem"]], row["width"], row["height"])
    best = predictions()
    selected = sorted(range(len(rows)), key=lambda i: identity({"purpose": "phase4a_verify", "stem": rows[i]["filestem"]}))[:32]
    for i in selected:
        detail = characterize(rows[i], best[rows[i]["filestem"]])
        if not np.array_equal(features["class_difficulty"][i], detail["class_difficulty"]):
            raise ValueError("Independent descriptive difficulty replay differs")
    summary = json.loads((PUBLIC/"simulation_summary.json").read_text())
    if summary["proposal_sha256"] != identity(proposal):
        raise ValueError("Sampler proposal changed")
    for path, wanted in summary["normalized_source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_text(encoding="utf-8").encode()).hexdigest() != wanted:
            raise ValueError("Simulated source implementation changed")
    active = json.loads((LOCAL/"simulation_active.json").read_text())
    directory = LOCAL/"simulations"/active["attempt"]
    with (PUBLIC/"epoch_exposures.csv").open() as f:
        reported = {(r["policy"], int(r["seed"]), int(r["epoch"])): r for r in csv.DictReader(f)}
    if len(reported) != 720:
        raise ValueError("Missing or duplicate simulated epoch rows")
    with (PUBLIC/"selection_overlap.csv").open() as f:
        overlap = {(int(r["seed"]), int(r["epoch"])): r for r in csv.DictReader(f)}
    if len(overlap) != 240:
        raise ValueError("Missing or duplicate paired overlap rows")
    # Reconstruct missed-at-.50 counts from the saved per-object characterization.
    details = [json.loads(line) for line in (LOCAL/"analysis/best_details.jsonl").read_text().splitlines()]
    if [d["stem"] for d in details] != [r["filestem"] for r in rows]:
        raise ValueError("Object diagnostic row order changed")
    miss50 = np.zeros_like(counts)
    for i, detail in enumerate(details):
        for obj in detail["objects"]:
            miss50[i, obj["class_id"]] += int(obj["difficulty"] == 1)
    lesion_patients = np.unique(patient[counts[:, 5] > 0])
    lesion = np.isin(patient, lesion_patients)
    patient_size = np.bincount(patient)
    patient_cap = np.minimum(3, np.ceil(patient_size*.5).astype(int))
    patient_cap[lesion_patients] = 0
    examined, total_draws = 0, 0
    ordered_differences = []
    for seed in proposal["seeds"]:
        file = directory/f"plans_{seed}.npz"
        if sha(file) != summary["plan_files_sha256"][file.name]:
            raise ValueError("Private epoch plans changed")
        plans = np.load(file)
        for epoch in range(80):
            # Frequency overlap and ordered disagreement answer different questions.
            rare, guided = plans["rarity_only"][epoch], plans["paeg_cat"][epoch]
            different = rare != guided
            ordered_differences.append(int(different.sum()))
            delta = np.bincount(rare, minlength=len(rows))-np.bincount(guided, minlength=len(rows))
            if not math.isclose(float(overlap[seed, epoch+1]["full_draw_total_variation"]),
                                float(np.abs(delta).sum()/(2*len(rows))), abs_tol=1e-12):
                raise ValueError("Published source-frequency total variation differs")
            # Paired random streams place the shared unique coverage at identical positions.
            ordering = np.random.default_rng(np.random.SeedSequence([seed, epoch, 617]))
            ordering.random(len(rows))
            bonus_position = ordering.permutation(len(rows)) >= len(rows)-625
            if np.any(different & ~bonus_position):
                raise ValueError("Shared coverage positions differ between paired policies")
            paired_unique, replacement_count = [], []
            for policy in proposal["policies"]:
                draws = plans[policy][epoch]
                if len(draws) != 14061 or np.any(draws < 0) or np.any(draws >= len(rows)):
                    raise ValueError("Plan budget/index range failed")
                repeats = np.bincount(draws, minlength=len(rows))
                unique = np.flatnonzero(repeats)
                if repeats.max() > (1 if policy == "standard" else 2) or np.any(repeats[lesion] != 1):
                    raise ValueError("Image/lesion cohort repetition failed")
                if len(np.unique(patient[unique])) != 4264:
                    raise ValueError("Patient coverage failed")
                extra = np.bincount(patient, weights=np.maximum(repeats-1, 0), minlength=len(patient_ids))
                if np.any(extra > patient_cap):
                    raise ValueError("Patient extra cap failed")
                report = reported[policy, seed, epoch+1]
                for status in (False, True):
                    mask = (counts[:, 0] > 0) == status
                    if mask[unique].sum() < math.ceil(.95*mask.sum()) or mask[draws].sum() != mask.sum():
                        raise ValueError("An individual epoch violates fracture-stratum coverage/quota")
                for c, name in enumerate(CLASSES):
                    if int(report[name+"_objects"]) != int(repeats@counts[:, c]) or int(report[name+"_hard_positive_exposure"]) != int(repeats@features["hard_counts"][:, c]):
                        raise ValueError("Published class exposure does not match saved source-index plan")
                    if int(report[name+"_missed50_exposure"]) != int(repeats@miss50[:, c]):
                        raise ValueError("Published missed-at-.50 exposure differs")
                for field, values in (("clinical_hard_positive_exposure", features["hard_counts"]),
                                      ("clinical_missed50_exposure", miss50)):
                    if int(report[field]) != int(repeats@values[:, [0, 3, 4, 6, 7]].sum(1)):
                        raise ValueError("Published priority-class exposure differs")
                if int(report["potential_fp_exposure"]) != int(repeats@features["potential_fp_counts"].sum(1)):
                    raise ValueError("Potential FP exposure mismatch")
                if int(report["source_uncertainty_mark_exposure"]) != int(repeats@features["uncertain"]):
                    raise ValueError("Uncertainty-marker exposure mismatch")
                if int(report["unique_images"]) != len(unique):
                    raise ValueError("Published unique-image count mismatch")
                if policy != "standard":
                    paired_unique.append(unique)
                    replacement_count.append(int(np.maximum(repeats-1, 0).sum()))
                examined += 1; total_draws += len(draws)
            if not np.array_equal(*paired_unique) or replacement_count != [625, 625]:
                raise ValueError("Matched coverage or intervention budget differs")
    if examined != 720 or total_draws != 10123920:
        raise ValueError("Independent simulation accounting differs")
    save(PUBLIC/"independent_verification.json", {"passed": True, "training_predictions": 14061,
        "stability_images_per_checkpoint": len(stability_subset(rows)), "all_outputs_original_coordinate_valid": True,
        "training_feature_replay_images": 32, "all_simulated_epochs_checked": examined,
        "all_simulated_draws_checked": total_draws, "no_per_epoch_guardrail_failure": True,
        "all_classes_exposure_reconciles_against_saved_draw_indices": True,
        "missed50_and_priority_exposure_reconciles": True,
        "source_frequency_total_variation_reconciles": True,
        "shared_coverage_positions_match_every_paired_epoch": True,
        "all_16_lesion_patients_every_view_exactly_once_per_epoch": True,
        "paired_budget_and_coverage_match_every_epoch": True, "no_new_gpu_work": True})
    save(PUBLIC/"ordered_plan_comparison.json", {
        "paired_epochs": len(ordered_differences),
        "different_ordered_positions": distribution(ordered_differences),
        "different_position_fraction": distribution(np.array(ordered_differences)/14061),
        "interpretation": "Ordered positions differ beyond source-frequency total variation because changed weighted ranking also reorders common bonus images. Shared coverage positions match. This is not an AP or gradient-effect estimate."})
    patient_diagnostics(rows, patient, patient_ids)
    print(f"Independent verification passes: {examined} epochs / {total_draws} draws", flush=True)


def patient_diagnostics(rows, patient, patient_ids):
    # Retain detailed error-type records privately and publish only patient distributions.
    details = [json.loads(line) for line in (LOCAL/"analysis/best_details.jsonl").read_text().splitlines()]
    if [d["stem"] for d in details] != [r["filestem"] for r in rows]:
        raise ValueError("Patient diagnostic row order changed")
    categories = ("TP", "classification", "low_confidence", "localization", "no_residual_candidate")
    by_patient = [{"patient_id": p, "errors": {t: {c: Counter() for c in CLASSES} for t in ("0.5", "0.75")}} for p in patient_ids]
    for row, d, p in zip(rows, details, patient):
        for threshold in ("0.5", "0.75"):
            for box, reason in zip(row["boxes"], d["operating"][threshold]["gt_reasons"]):
                by_patient[p]["errors"][threshold][CLASSES[box["class_id"]]][reason] += 1
    save(LOCAL/"analysis/patient_error_types.json", by_patient)
    aggregates = {t: {c: {reason: distribution([r["errors"][t][c][reason] for r in by_patient])
                         for reason in categories} for c in CLASSES} for t in ("0.5", "0.75")}
    save(PUBLIC/"patient_error_distributions.json", {"patients": len(patient_ids), "zero_counts_include_all_training_patients": True,
        "unit": "Per-patient annotated object counts, fixed training split", "distributions": aggregates,
        "detailed_patient_records_private": True})


if __name__ == "__main__":
    run()
