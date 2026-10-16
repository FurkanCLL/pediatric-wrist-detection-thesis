"""Independent reconciliation of saved draw arrays and crop exposure totals."""

import json

import numpy as np

from wrist_train.common import configure_runtime

configure_runtime()
from wrist_train.protocol import canonical_hash, load_protocol
from .common import CLASSES, LOCAL, PUBLIC, ROOT, distribution, save_json, training_records
from .crops import propose_crop
from .sampling import DesignPopulation


def main():
    _, baseline_lock = load_protocol()
    config = json.loads((ROOT / "configs/class_aware_candidates.json").read_text())
    public = json.loads((PUBLIC / "simulation_summary.json").read_text())
    if canonical_hash(config) != public["configuration_sha256"]:
        raise ValueError("Simulation policy identity changed")
    population = DesignPopulation(training_records(), config)
    rows, patient, counts, present = population.rows, population.patient, population.counts, population.present
    total_epochs = 0
    result = {"status": "passed", "configuration_sha256": canonical_hash(config),
              "frozen_baseline_protocol_sha256": baseline_lock["protocol_sha256"], "policies": {}}
    for name, policy in [("standard", None), *config["candidates"].items()]:
        result["policies"][name] = {}
        cache = {}
        for seed in config["simulation_seeds"]:
            arrays = np.load(LOCAL / f"{name}_seed{seed}.npz", allow_pickle=False)
            draws, cropped = arrays["draws"], arrays["cropped"]
            epochs = [json.loads(line) for line in (LOCAL / f"{name}_seed{seed}_epochs.jsonl").read_text().splitlines()]
            if draws.shape != (80, 14061) or cropped.shape != draws.shape or len(epochs) != 80:
                raise ValueError("Incomplete simulation budget")
            lesion_cohort_ratios, double_patient_counts, over_quarter, object_min = [], [], [], []
            for epoch, (selected, crop_flags, recorded) in enumerate(zip(draws, cropped, epochs)):
                if np.any(selected < 0) or np.any(selected >= len(rows)):
                    raise ValueError("Unknown simulation index")
                repeats = np.bincount(selected, minlength=len(rows))
                patients = np.bincount(patient[selected], minlength=len(population.patient_size))
                if not np.all(patients > 0):
                    raise ValueError("Patient coverage was lost")
                before = counts[selected].sum(axis=0)
                after = before.copy()
                images_after = present[selected].sum(axis=0)
                if present[selected, 0].sum() != 9307 or len(selected)-present[selected, 0].sum() != 4754:
                    raise ValueError("Fracture status quota mismatch")
                full_parent = np.bincount(selected[~crop_flags], minlength=len(rows)) > 0
                if not np.all(full_parent[repeats > 0]):
                    raise ValueError("A selected parent lacks its full-image view")
                if policy:
                    cap = np.minimum(policy["maximum_patient_extra_absolute"],
                                     np.ceil(population.patient_size*policy["maximum_patient_extra_fraction"]))
                    if repeats.max() > policy["maximum_image_draws"] or np.any(patients > population.patient_size+cap):
                        raise ValueError("Repetition cap mismatch")
                    if np.bincount(patient[selected[crop_flags]], minlength=len(patients)).max() > 1:
                        raise ValueError("Crop patient cap mismatch")
                if np.any(patients[population.lesion_patients] > population.patient_size[population.lesion_patients]):
                    raise ValueError("Bone-lesion patients gained repeated draws")
                if not np.all(repeats[present[:, 5]] == 1):
                    raise ValueError("Bone-lesion image coverage changed")
                for index in selected[crop_flags]:
                    index = int(index)
                    if index not in cache:
                        plan = propose_crop(rows[index], policy["crop"], config["target_classes"], population.weights(policy)[0])
                        if not plan["accepted"]:
                            raise ValueError("Recorded crop is not reproducible")
                        # Recompute retained labels directly from rectangle intersections, not output boxes.
                        left, top, right, bottom = plan["rect"]
                        retained = np.zeros(8, dtype=int)
                        for box in rows[index]["boxes"]:
                            x1, y1, x2, y2 = box["xyxy"]
                            visible = max(0, min(x2, right)-max(x1, left))*max(0, min(y2, bottom)-max(y1, top))
                            fraction = visible/((x2-x1)*(y2-y1))
                            if box["class_id"] == 0 and fraction < 1-1e-9:
                                raise ValueError("Fracture was cropped out")
                            if visible:
                                if fraction < .9:
                                    raise ValueError("A visible fragment lost its annotation")
                                retained[box["class_id"]] += 1
                        cache[index] = retained
                    retained = cache[index]
                    after += retained-counts[index]
                    images_after += (retained > 0).astype(int)-present[index].astype(int)
                if (before.tolist() != recorded["class_objects_before"] or after.tolist() != recorded["class_objects_after"] or
                    images_after.tolist() != recorded["class_image_after"] or after[0] != before[0] or after[5] != 35):
                    raise ValueError("Saved exposure summary does not reconcile with draw/crop arrays")
                # Reproduce boundary epochs with reversed metadata to check full-population order independence.
                if epoch in (0, 79):
                    reversed_population = DesignPopulation(rows[::-1], config)
                    replay, _, _ = reversed_population.sample(policy, seed, epoch)
                    if not np.array_equal(selected, replay):
                        raise ValueError("Simulation ordering is not reproducible")
                ratio = patients/population.patient_size
                double_patient_counts.append(int((ratio >= 2-1e-9).sum()))
                over_quarter.append(int((ratio > 1.25+1e-9).sum()))
                lesion_cohort_ratios.append(float(patients[population.lesion_patients].sum()/population.patient_size[population.lesion_patients].sum()))
                object_min.append(float(after[0]/12422))
                total_epochs += 1
            result["policies"][name][str(seed)] = {
                "epochs_checked": len(epochs), "draws_checked": int(draws.size),
                "patients_at_least_double_baseline_per_epoch": distribution(double_patient_counts),
                "patients_over_1_25_baseline_per_epoch": distribution(over_quarter),
                "lesion_patient_cohort_draw_ratio": distribution(lesion_cohort_ratios),
                "fracture_object_ratio": distribution(object_min),
                "all_selected_parents_have_full_image_view": True,
                "image_and_patient_order_independence": "boundary epochs reproduced with reversed records"}
    result.update(epochs_checked=total_epochs, draws_checked=total_epochs*14061,
                  source_images_written=0, test_images_opened=0, model_training_performed=False)
    save_json(PUBLIC / "simulation_verification.json", result)
    print({"epochs_checked": total_epochs, "draws_checked": result["draws_checked"], "status": "passed"})


if __name__ == "__main__":
    main()
