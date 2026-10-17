"""Matched 80-epoch source-index simulations; no loader integration or model fitting."""

import argparse
import csv
import hashlib
import json
import math
import subprocess

import numpy as np

from .common import ROOT, LOCAL, PUBLIC, CLASSES, training_records, identity, sha, save
from .analysis import concentration, CLINICAL
from .sampling import MatchedPopulation
from wrist_eval.metrics import distribution
from wrist_train.budget import optimizer_budget

PROPOSAL = ROOT/"configs/error_guided_sampler_proposed.json"


def metric_row(pop, draws, flags, features, miss50, seed, epoch, policy):
    repeats = np.bincount(draws, minlength=pop.n)
    selected = np.flatnonzero(repeats)
    patients = np.bincount(pop.patient, weights=repeats, minlength=len(pop.patient_size))
    extras = np.maximum(repeats-1, 0)
    patient_extra = np.bincount(pop.patient, weights=extras, minlength=len(pop.patient_size))
    result = {"policy": policy, "seed": seed, "epoch": epoch+1, "draws": len(draws),
        "unique_images": len(selected), "unique_patients": int((patients > 0).sum()),
        "image_max_draws": int(repeats.max()), "patient_max_draws": int(patients.max()),
        "patient_draws_p99": float(np.quantile(patients, .99)), "patient_max_extras": int(patient_extra.max()),
        "patient_max_relative": float((patients/pop.patient_size).max()),
        "replacement_draws": int(extras.sum()), "replacement_fraction": float(extras.sum()/pop.n),
        "hard_positive_exposure": int(repeats@features["hard_counts"].sum(1)),
        "clinical_hard_positive_exposure": int(repeats@features["hard_counts"][:, CLINICAL].sum(1)),
        "clinical_missed50_exposure": int(repeats@miss50[:, CLINICAL].sum(1)),
        "potential_fp_exposure": int(repeats@features["potential_fp_counts"].sum(1)),
        "clinical_potential_fp_exposure": int(repeats@features["potential_fp_counts"][:, CLINICAL].sum(1)),
        "source_uncertainty_mark_exposure": int(repeats@features["uncertain"]),
        "difficulty_draw_mean": float(pop.difficulty[draws].mean()),
        "extra_difficulty_mean": float(pop.difficulty[draws[flags]].mean()) if flags.any() else 0.,
        "extra_zero_difficulty_images": int((pop.difficulty[draws[flags]] == 0).sum()),
        "extra_images_without_rarity_priority": int((pop.rarity[draws[flags]] == 0).sum()),
        "lesion_cohort_views_exactly_once": bool(np.all(repeats[np.isin(pop.patient, pop.lesion_patients)] == 1))}
    for positive, name in ((True, "fracture_positive"), (False, "no_fracture_box")):
        status = pop.present[:, 0] == positive
        result[name+"_unique"] = int(status[selected].sum())
        result[name+"_unique_fraction"] = float(status[selected].sum()/status.sum())
        result[name+"_draws"] = int(status[draws].sum())
    for c, name in enumerate(CLASSES):
        result[name+"_objects"] = int(repeats@pop.counts[:, c])
        result[name+"_image_exposure"] = int(repeats@pop.present[:, c])
        result[name+"_hard_positive_exposure"] = int(repeats@features["hard_counts"][:, c])
        result[name+"_missed50_exposure"] = int(repeats@miss50[:, c])
        patient_objects = np.bincount(pop.patient, weights=repeats*pop.counts[:, c], minlength=len(pop.patient_size))
        result[name+"_independent_patients"] = int((patient_objects > 0).sum())
        result[name+"_patient_max_object_share"] = float(patient_objects.max()/patient_objects.sum())
        result[name+"_effective_patients"] = float(patient_objects.sum()**2/(patient_objects**2).sum())
    return result, repeats, patients


def run(attempt):
    proposal = json.loads(PROPOSAL.read_text())
    if proposal["official_training_authorized"] or not proposal["simulations_authorized"]:
        raise ValueError("Simulation authorization differs")
    if proposal["epochs"] != 80 or proposal["seeds"] != [42, 1042, 2042] or proposal["policies"] != ["standard", "rarity_only", "paeg_cat"]:
        raise ValueError("The bounded three-policy simulation plan differs")
    rows = training_records()
    features = dict(np.load(LOCAL/"analysis/features.npz"))
    if identity({k: features[k].tolist() for k in sorted(features)}) != proposal["feature_content_sha256"]:
        raise ValueError("Sealed difficulty features changed")
    specification = json.loads((ROOT/"configs/class_aware_candidates.json").read_text())
    if identity(specification) != proposal["class_aware_candidates_canonical_sha256"]:
        raise ValueError("Historical A/B/C proposal changed")
    directory = LOCAL/"simulations"/attempt
    directory.mkdir(parents=True, exist_ok=False)
    population = MatchedPopulation(rows, specification, features["difficulty"], proposal)
    if population.requested != proposal["requested_integer_budget"]:
        raise ValueError("Integer replacement proposal differs")
    details = [json.loads(line) for line in (LOCAL/"analysis/best_details.jsonl").read_text().splitlines()]
    if [r["stem"] for r in details] != [r["filestem"] for r in rows]:
        raise ValueError("Object difficulty row order differs")
    miss50 = np.zeros((population.n, 8), dtype=int)
    for i, item in enumerate(details):
        for obj in item["objects"]:
            miss50[i, obj["class_id"]] += int(obj["difficulty"] == 1)
    records, overlaps, seed_summaries = [], [], []
    indices = {r["filestem"]: i for i, r in enumerate(rows)}
    audit = json.loads((LOCAL/"qualitative/selection.json").read_text())
    audit_indices = [indices[c["stem"]] for c in audit]
    for seed in proposal["seeds"]:
        plans = {name: np.empty((proposal["epochs"], population.n), dtype=np.int32) for name in proposal["policies"]}
        cumulative = {name: np.zeros(population.n, dtype=np.int32) for name in proposal["policies"]}
        for epoch in range(proposal["epochs"]):
            standard = population.sample(None, seed, epoch)[0]
            rare, rf, guided, gf, accounting = population.paired(seed, epoch)
            extras_a, extras_b = set(rare[rf]), set(guided[gf])
            overlaps.append({"seed": seed, "epoch": epoch+1, "realized_bonus": len(extras_a),
                "extra_intersection": len(extras_a & extras_b), "extra_union": len(extras_a | extras_b),
                "extra_jaccard": len(extras_a & extras_b)/len(extras_a | extras_b) if extras_a | extras_b else 1.,
                "different_bonus_images": len(extras_b-extras_a),
                "full_draw_total_variation": len(extras_b-extras_a)/population.n,
                "requested": accounting["requested_bonus"], "coverage_restoration": accounting["coverage_fallback"]})
            for policy, draws, flags in (("standard", standard, np.zeros(len(standard), dtype=bool)),
                                         ("rarity_only", rare, rf), ("paeg_cat", guided, gf)):
                population.check(draws)
                record, repeats, _ = metric_row(population, draws, flags, features, miss50, seed, epoch, policy)
                records.append(record)
                cumulative[policy] += repeats
                plans[policy][epoch] = draws
            if epoch % 20 == 0:
                print(f"Matched sampler seed {seed}: epoch {epoch+1}/{proposal['epochs']}", flush=True)
        np.savez_compressed(directory/f"plans_{seed}.npz", **plans)
        for policy, repeats in cumulative.items():
            patient_draws = np.bincount(population.patient, weights=repeats, minlength=len(population.patient_size))
            subset = [r for r in records if r["seed"] == seed and r["policy"] == policy]
            numeric = [k for k in subset[0] if k not in ("policy", "seed", "epoch") and not isinstance(subset[0][k], bool)]
            seed_summaries.append({"seed": seed, "policy": policy,
                "epoch_distributions": {k: distribution([r[k] for r in subset]) for k in numeric},
                "per_epoch_all_guardrails_pass": True,
                "cumulative_image_draws": distribution(repeats), "cumulative_patient_relative": distribution(patient_draws/(population.patient_size*proposal["epochs"])),
                "all_images_seen": bool(np.all(repeats > 0)), "all_patients_seen": bool(np.all(patient_draws > 0)),
                "cumulative_patient_concentration": concentration(patient_draws),
                "audit_case_cumulative_draws": [{"case": c["case"], "draws": int(repeats[i])} for c, i in zip(audit, audit_indices)]})
    for name, values in (("epoch_exposures.csv", records), ("selection_overlap.csv", overlaps)):
        with (PUBLIC/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    paired = {}
    for seed in proposal["seeds"]:
        rare = [r for r in records if r["seed"] == seed and r["policy"] == "rarity_only"]
        guided = [r for r in records if r["seed"] == seed and r["policy"] == "paeg_cat"]
        keys = [k for k in rare[0] if k not in ("policy", "seed", "epoch") and not isinstance(rare[0][k], bool)]
        paired[str(seed)] = {k: distribution([b[k]-a[k] for a, b in zip(rare, guided)]) for k in keys}
    summary = {"policies": proposal["policies"], "epochs": proposal["epochs"], "seeds": proposal["seeds"],
        "simulated_epochs": len(records), "simulated_draws": sum(r["draws"] for r in records),
        "proposal_sha256": identity(proposal), "feature_content_sha256": proposal["feature_content_sha256"],
        "all_720_epoch_guardrails_pass": True, "matched_unique_image_sets_every_epoch": True,
        "matched_realized_replacement_counts_every_epoch": True, "potential_fp_in_weight": False,
        "no_model_training_or_holdout_evaluation": True, "full_image_only": True,
        "seed_summaries": seed_summaries, "paeg_minus_rarity_paired_epoch_differences": paired,
        "overlap": {k: distribution([r[k] for r in overlaps]) for k in overlaps[0] if k not in ("seed", "epoch")},
        "nominal_optimizer_budget": optimizer_budget(14061, 6, 24, 80, 1),
        "plan_files_sha256": {f"plans_{seed}.npz": sha(directory/f"plans_{seed}.npz") for seed in proposal["seeds"]},
        "miss50_details_sha256": sha(LOCAL/"analysis/best_details.jsonl"),
        "normalized_source_sha256": {f"src/wrist_mining/{name}": hashlib.sha256((ROOT/"src/wrist_mining"/name).read_text(encoding="utf-8").encode()).hexdigest()
                                     for name in ("simulate.py", "sampling.py", "analysis.py")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    save(directory/"summary.json", summary)
    save(PUBLIC/"simulation_summary.json", summary)
    save(LOCAL/"simulation_active.json", {"attempt": attempt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    run(parser.parse_args().attempt)
