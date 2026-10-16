"""Training-only eligible-pool geometry and annotation effects for each candidate."""

from collections import Counter
import json

import numpy as np

from .common import CLASSES, PUBLIC, ROOT, distribution, save_json, training_records
from .crops import propose_crop
from .sampling import DesignPopulation


def main():
    config = json.loads((ROOT / "configs/class_aware_candidates.json").read_text())
    population = DesignPopulation(training_records(), config)
    result = {"scope": "unique eligible training parents; not draw-weighted simulation counts", "policies": {}}
    for name, policy in config["candidates"].items():
        weights, images = population.weights(policy)
        eligible = np.flatnonzero(images > 0)
        plans = [(i, propose_crop(population.rows[i], policy["crop"], config["target_classes"], weights)) for i in eligible]
        accepted = [(i, p) for i, p in plans if p["accepted"]]
        result["policies"][name] = {
            "eligible_images": len(eligible), "eligible_patients": len(np.unique(population.patient[eligible])),
            "eligible_per_class": {class_name: {"images": int(population.present[eligible, c].sum()),
                "patients": len(np.unique(population.patient[eligible[population.present[eligible, c]]]))}
                for c, class_name in enumerate(CLASSES)},
            "image_weight_range": [float(images[eligible].min()), float(images[eligible].max())],
            "accepted_unique_crop_parents": len(accepted),
            "accepted_crop_target_classes": dict(Counter(CLASSES[p["target_class"]] for _, p in accepted)),
            "crop_outcomes": dict(Counter(p["reason"] for _, p in plans)),
            "accepted_zoom": distribution([p["zoom"] for _, p in accepted]),
            "target_short_before_768": distribution([p["source_target_short_at_768"] for _, p in accepted]),
            "target_short_after_768": distribution([p["crop_target_short_at_768"] for _, p in accepted]),
            "retained_image_area_fraction": distribution([(p["rect"][2]-p["rect"][0])*(p["rect"][3]-p["rect"][1])/
                (population.rows[i]["width"]*population.rows[i]["height"]) for i, p in accepted]),
            "clipped_annotations": sum(p["clipped_annotations"] for _, p in accepted),
            "omitted_objects": dict(zip(CLASSES, np.sum([p["omitted_objects"] for _, p in accepted], axis=0).tolist()))}
    lesion_objects = np.bincount(population.patient, weights=population.counts[:, 5])
    lesion_images = np.bincount(population.patient, weights=population.present[:, 5])
    support = population.lesion_patients
    result["bonelesion"] = {"images_per_positive_patient": distribution(lesion_images[support]),
                            "objects_per_positive_patient": distribution(lesion_objects[support]),
                            "largest_patient_image_share": float(lesion_images.max()/lesion_images.sum()),
                            "largest_patient_object_share": float(lesion_objects.max()/lesion_objects.sum())}
    save_json(PUBLIC / "policy_geometry.json", result)
    print({name: {key: data[key] for key in ("eligible_images", "eligible_patients", "accepted_unique_crop_parents")}
           for name, data in result["policies"].items()})


if __name__ == "__main__":
    main()
