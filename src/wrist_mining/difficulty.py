"""One-to-one descriptive training errors; no changes to the pinned evaluator."""

import numpy as np

from wrist_eval.analyze import error_types, truth
from wrist_eval.metrics import size_bin


# Localization-only misses receive half weight; a .50 miss receives full weight.
def object_difficulty(matched50, matched75):
    a, b = np.asarray(matched50), np.asarray(matched75)
    if a.shape != b.shape:
        raise ValueError("Object match arrays differ")
    return np.where(a < 0, 1., np.where(b < 0, .5, 0.))


def characterize(row, predictions):
    if row["split"] != "train":
        raise ValueError("Difficulty may only use training annotations")
    gt = truth(row)
    points = {}
    for threshold in (.5, .75):
        g, p, matrix, reasons, fp_reasons = error_types(gt, predictions, .25, threshold)
        points[str(threshold)] = {"matched_gt": g.tolist(), "matched_pred": p.tolist(),
            "gt_reasons": reasons.tolist(), "potential_fp_reasons": fp_reasons.tolist(),
            "matched_gt_iou": [float(matrix[i, j]) if j >= 0 else None for i, j in enumerate(g)]}
    scores = object_difficulty(points["0.5"]["matched_gt"], points["0.75"]["matched_gt"])
    class_scores = np.zeros(8, dtype=float)
    objects = []
    for c in range(8):
        selected = gt[:, 4] == c
        if selected.any():
            class_scores[c] = scores[selected].mean()
    for i, box in enumerate(gt):
        short = min(box[2]-box[0], box[3]-box[1])*768/max(row["width"], row["height"])
        objects.append({"class_id": int(box[4]), "difficulty": float(scores[i]),
                        "short_768": float(short), "size_bin": size_bin(short)})
    return {"operating": points, "class_difficulty": class_scores.tolist(), "objects": objects}


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start+end-1)/2
        start = end
    return result


def rank_correlation(a, b):
    if len(a) != len(b):
        raise ValueError("Paired ranking populations differ")
    a, b = average_ranks(a), average_ranks(b)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() and b.std() else None
