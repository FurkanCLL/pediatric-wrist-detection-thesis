"""Deterministic exposure simulation; no network fitting or validation-policy search."""

from collections import Counter
import csv
import hashlib
import json

import numpy as np

from .common import CLASSES, LOCAL, PUBLIC, ROOT, distribution, save_json, training_records
from .crops import propose_crop
from .sampling import DesignPopulation


def run_policy(population, policy, seed, epochs):
    n = population.n
    patient = population.patient
    counts = population.counts
    present = population.present
    class_weights = population.weights(policy)[0] if policy else np.zeros(8)
    crop_cache = {}
    total_image = np.zeros(n, dtype=np.int32)
    total_patient = np.zeros(len(population.patient_size), dtype=np.int32)
    selected_union = np.zeros(n, dtype=bool)
    all_draws, all_crop_flags, records = [], [], []
    for epoch in range(epochs):
        draws, bonus, sampling = population.sample(policy, seed, epoch)
        image_repeats = np.bincount(draws, minlength=n)
        patient_repeats = np.bincount(patient[draws], minlength=len(total_patient))
        before = counts[draws].sum(axis=0)
        after = before.copy()
        image_after = present[draws].sum(axis=0)
        pair_before = present[draws].astype(np.int32).T @ present[draws].astype(np.int32)
        pair_after = pair_before.copy()
        full_flags = np.ones(n, dtype=bool)
        crop_flags = np.zeros(n, dtype=bool)
        rejected, omitted, clipped, crop_patients = Counter(), np.zeros(8, dtype=int), 0, Counter()
        attempts = 0
        zooms = []
        if policy:
            crop = policy["crop"]
            quota = round(n*crop["maximum_attempt_fraction"])
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9127]))
            for position in rng.permutation(np.flatnonzero(bonus)):
                index = int(draws[position])
                p = int(patient[index])
                if attempts >= quota:
                    break
                if crop_patients[p] >= crop["maximum_crops_per_patient_per_epoch"]:
                    continue
                crop_patients[p] += 1
                attempts += 1
                if index not in crop_cache:
                    crop_cache[index] = propose_crop(population.rows[index], crop,
                        population.specification["target_classes"], class_weights)
                plan = crop_cache[index]
                if not plan["accepted"]:
                    rejected[plan["reason"]] += 1
                    continue
                retained = np.bincount([b["class_id"] for b in plan["boxes"]], minlength=8)
                after += retained-counts[index]
                old_presence = present[index].astype(np.int32)
                new_presence = (retained > 0).astype(np.int32)
                image_after += new_presence-old_presence
                pair_after += np.outer(new_presence, new_presence)-np.outer(old_presence, old_presence)
                omitted += plan["omitted_objects"]
                clipped += plan["clipped_annotations"]
                crop_flags[position], full_flags[position] = True, False
                zooms.append(plan["zoom"])
        positive = present[:, 0]
        unique = image_repeats > 0
        lesion_draws = draws[present[draws, 5]]
        lesion_patient = np.bincount(patient[lesion_draws], minlength=len(total_patient))
        lesion_support = population.lesion_patients
        full_parent = np.bincount(draws[full_flags], minlength=n) > 0
        record = {"seed": seed, "epoch": epoch+1, "draws": len(draws),
            "unique_images": int(unique.sum()), "unique_patients": int((patient_repeats > 0).sum()),
            "fracture_positive_draws": int(present[draws, 0].sum()), "fracture_negative_draws": int((~present[draws, 0]).sum()),
            "fracture_unique_images": int((unique & positive).sum()), "negative_unique_images": int((unique & ~positive).sum()),
            "fracture_unique_patients": len(np.unique(patient[unique & positive])),
            "negative_unique_patients": len(np.unique(patient[unique & ~positive])),
            "full_draws": int(full_flags.sum()), "crop_draws": int(crop_flags.sum()), "crop_attempts": attempts,
            "crop_fallbacks": sum(rejected.values()), "crop_reasons": dict(rejected), "clipped_annotations": clipped,
            "image_repetition": distribution(image_repeats), "patient_repetition": distribution(patient_repeats),
            "patient_relative_repetition": distribution(patient_repeats/population.patient_size),
            "lesion_image_repetition": distribution(image_repeats[present[:, 5]]),
            "lesion_patient_draws": distribution(lesion_patient[lesion_support]),
            "lesion_top_patient_draw_share": float(lesion_patient.max()/max(1, len(lesion_draws))),
            "lesion_patient_hhi": float(((lesion_patient/max(1, len(lesion_draws)))**2).sum()),
            "protected_rare_images_with_full_view": int(full_parent[population.protected].sum()),
            "class_image_before": present[draws].sum(axis=0).astype(int).tolist(),
            "class_image_after": image_after.astype(int).tolist(),
            "class_objects_before": before.astype(int).tolist(), "class_objects_after": after.astype(int).tolist(),
            "class_unique_images": (present & unique[:, None]).sum(axis=0).astype(int).tolist(),
            "class_unique_patients": [len(np.unique(patient[unique & present[:, c]])) for c in range(8)],
            "class_objects_omitted": omitted.tolist(), "cooccurrence_before": pair_before.astype(int).tolist(),
            "cooccurrence_after": pair_after.astype(int).tolist(),
            "zoom": distribution(zooms) if zooms else None, **sampling}
        if record["fracture_positive_draws"] != int(positive.sum()) or after[0] != before[0]:
            raise ValueError("Fracture status or object retention changed during simulation")
        if policy and (image_repeats.max() > policy["maximum_image_draws"] or
                       np.any(patient_repeats > population.patient_size + np.minimum(
                       policy["maximum_patient_extra_absolute"], np.ceil(population.patient_size*policy["maximum_patient_extra_fraction"])))):
            raise ValueError("Image/patient repetition cap exceeded")
        if not np.all(full_parent[population.protected]):
            raise ValueError("A protected rare/empty image lost its full-image view")
        total_image += image_repeats
        total_patient += patient_repeats
        selected_union |= unique
        records.append(record)
        all_draws.append(draws)
        all_crop_flags.append(crop_flags)
    return (records, {"distinct_images_over_budget": int(selected_union.sum()),
                     "distinct_patients_over_budget": int((total_patient > 0).sum()),
                     "total_image_repetition": distribution(total_image),
                     "total_patient_repetition": distribution(total_patient),
                     "total_patient_relative_repetition": distribution(total_patient/(epochs*population.patient_size))},
           np.stack(all_draws), np.stack(all_crop_flags))


def summarize(records, cumulative, base_counts, base_presence):
    scalar = ("unique_images", "unique_patients", "fracture_positive_draws", "fracture_negative_draws",
              "fracture_unique_images", "negative_unique_images", "fracture_unique_patients", "negative_unique_patients",
              "full_draws", "crop_draws", "crop_attempts", "crop_fallbacks", "clipped_annotations",
              "requested_bonus", "rare_bonus", "coverage_fallback", "lesion_top_patient_draw_share", "lesion_patient_hhi")
    result = {"epochs": len(records), "per_epoch": {key: distribution([r[key] for r in records]) for key in scalar},
              "cumulative": cumulative, "classes": {}}
    for c, name in enumerate(CLASSES):
        row = {}
        for key in ("class_image_before", "class_image_after", "class_objects_before", "class_objects_after",
                    "class_unique_images", "class_unique_patients", "class_objects_omitted"):
            row[key] = distribution([r[key][c] for r in records])
        row["post_crop_object_ratio"] = row["class_objects_after"]["mean"]/base_counts[c]
        row["post_crop_image_ratio"] = row["class_image_after"]["mean"]/base_presence[c]
        result["classes"][name] = row
    for key in ("image_repetition", "patient_repetition", "patient_relative_repetition", "lesion_image_repetition", "lesion_patient_draws"):
        result[key] = {stat: distribution([r[key][stat] for r in records]) for stat in records[0][key]}
    result["crop_fallback_reasons"] = dict(sum((Counter(r["crop_reasons"]) for r in records), Counter()))
    result["mean_cooccurrence_after"] = np.mean([r["cooccurrence_after"] for r in records], axis=0).tolist()
    result["mean_cooccurrence_before"] = np.mean([r["cooccurrence_before"] for r in records], axis=0).tolist()
    return result


def main():
    config = json.loads((ROOT / "configs/class_aware_candidates.json").read_text())
    population = DesignPopulation(training_records(), config)
    save_json(LOCAL / "training_index.json", [{"filestem": r["filestem"], "patient_id": r["patient_id"]} for r in population.rows])
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    aggregate = {"status": "proposed; not approved for training", "configuration_sha256": identity,
                 "scope": "training-only nominal exposures before common random augmentation", "seeds": config["simulation_seeds"],
                 "baseline_sampling": "one full-image draw per source image per epoch, shuffled without replacement",
                 "results": {}, "class_weights": {}}
    csv_rows = []
    for name, policy in [("standard", None), *config["candidates"].items()]:
        aggregate["results"][name] = {}
        if policy:
            aggregate["class_weights"][name] = dict(zip(CLASSES, population.weights(policy)[0].tolist()))
        for seed in config["simulation_seeds"]:
            rows, cumulative, draws, crops = run_policy(population, policy, seed, config["epochs"])
            aggregate["results"][name][str(seed)] = summarize(rows, cumulative, population.counts.sum(axis=0), population.present.sum(axis=0))
            np.savez_compressed(LOCAL / f"{name}_seed{seed}.npz", draws=draws, cropped=crops)
            with (LOCAL / f"{name}_seed{seed}_epochs.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                    flat = {"candidate": name, **{k: v for k, v in row.items() if isinstance(v, (int, float))}}
                    for key in ("image_repetition", "patient_repetition", "patient_relative_repetition", "lesion_image_repetition", "lesion_patient_draws"):
                        flat.update({key+"_"+stat: value for stat, value in row[key].items()})
                    for key in ("class_image_before", "class_image_after", "class_objects_before", "class_objects_after", "class_unique_images", "class_unique_patients"):
                        flat.update({key+"_"+class_name: row[key][c] for c, class_name in enumerate(CLASSES)})
                    csv_rows.append(flat)
            print(f"Completed {name}, seed {seed}, {config['epochs']} epochs", flush=True)
    save_json(PUBLIC / "simulation_summary.json", aggregate)
    with (PUBLIC / "epoch_exposures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
