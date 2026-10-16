"""Training support, co-occurrence and geometry; never reads holdout images."""

import numpy as np

from .common import CLASSES, PUBLIC, ROOT, arrays, distribution, save_json, training_records


def analyze(rows):
    counts, patient, patients = arrays(rows)
    present = counts > 0
    cooccurrence = present.astype(np.int32).T @ present.astype(np.int32)
    support, geometry = {}, {}
    for c, name in enumerate(CLASSES):
        selected = present[:, c]
        support[name] = {"objects": int(counts[:, c].sum()), "images": int(selected.sum()),
                         "patients": len(set(patient[selected])),
                         "also_fracture": int((selected & present[:, 0]).sum()),
                         "multiple_categories": int((selected & (present.sum(axis=1) > 1)).sum())}
        area, short, relative_short = [], [], []
        for row in rows:
            for box in row["boxes"]:
                if box["class_id"] == c:
                    x1, y1, x2, y2 = box["xyxy"]
                    area.append((x2-x1)*(y2-y1)/(row["width"]*row["height"]))
                    relative_short.append(min(x2-x1, y2-y1)/max(row["width"], row["height"]))
                    short.append(relative_short[-1]*768)
        geometry[name] = {"relative_area": distribution(area), "short_side_at_768": distribution(short),
                          "relative_short_to_long_edge": distribution(relative_short)}
    return {"scope": "training only", "images": len(rows), "patients": len(patients),
            "fracture_positive": int(present[:, 0].sum()), "fracture_negative": int((~present[:, 0]).sum()),
            "empty_labels": int((present.sum(axis=1) == 0).sum()), "classes": support,
            "cooccurrence_order": list(CLASSES), "cooccurrence_images": cooccurrence.tolist(),
            "geometry": geometry, "images_per_patient": distribution(np.bincount(patient)),
            "aspect_ratio": distribution([r["width"]/r["height"] for r in rows])}


def main():
    rows = training_records()
    result = analyze(rows)
    counts, patient, _ = arrays(rows)
    present = counts > 0
    result["replacement_references"] = {}
    # These exact expectations explain replacement risks; they are not proposed training policies.
    for name, weights in (("uniform_replacement", np.ones(len(rows))),
                          ("sqrt_rarity_sum", present @ (1/np.sqrt(present.sum(axis=0)))),
                          ("inverse_frequency_sum", present @ (1/present.sum(axis=0)))):
        probabilities = weights/weights.sum()
        patient_probability = np.bincount(patient, weights=probabilities)
        result["replacement_references"][name] = {
            "expected_unique_images": float((-np.expm1(len(rows)*np.log1p(-probabilities))).sum()),
            "expected_unique_patients": float((-np.expm1(len(rows)*np.log1p(-patient_probability))).sum()),
            "expected_image_exposure": dict(zip(CLASSES, (len(rows)*(present.T @ probabilities)).tolist())),
            "expected_object_exposure": dict(zip(CLASSES, (len(rows)*(counts.T @ probabilities)).tolist())),
            "expected_empty_label_draws": float(len(rows)*probabilities[~present.any(axis=1)].sum())}
    import json
    freeze = json.loads((ROOT / "outputs/dataset_preparation/freeze.json").read_text())
    result["split_manifest_sha256"] = freeze["split_manifest_sha256"]
    result["records_sha256"] = freeze["records_sha256"]
    save_json(PUBLIC / "training_analysis.json", result)
    print({name: row for name, row in result["classes"].items()})


if __name__ == "__main__":
    main()
