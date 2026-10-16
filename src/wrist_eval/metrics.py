"""Native AP and separate, deterministic one-to-one error accounting."""

import numpy as np
from .common import CLASSES


def validate_predictions(pred, width, height):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite prediction")
    if len(pred) and (np.any(pred[:, 2:4] <= pred[:, :2]) or np.any(pred[:, :4] < 0)
                      or np.any(pred[:, [0, 2]] > width + 1e-3) or np.any(pred[:, [1, 3]] > height + 1e-3)
                      or np.any((pred[:, 4] < 0) | (pred[:, 4] > 1))
                      or np.any((pred[:, 5] < 0) | (pred[:, 5] >= 8) | (pred[:, 5] != np.floor(pred[:, 5])))):
        raise ValueError("Invalid prediction coordinates, confidence or class ID")
    return pred[np.argsort(-pred[:, 4], kind="stable")]


def remove_padding_only(pred):
    # A native network-space box can lie wholly in letterbox padding. Clipping then gives zero area.
    # Keep it in native AP, but count and omit it from valid original-image prediction exports.
    pred=np.asarray(pred,dtype=np.float32).reshape(-1,6)
    if not np.isfinite(pred).all():raise ValueError("Non-finite original-image prediction")
    keep=np.all(pred[:,2:4]>pred[:,:2],axis=1)
    return pred[keep],int((~keep).sum())


def ious(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match(gt, pred, threshold=0.5, confidence=0.25, class_aware=True):
    # Confidence-ordered predictions claim the best unclaimed target; ties keep source order.
    matrix = ious(gt[:, :4], pred[:, :4])
    g_to_p = np.full(len(gt), -1, dtype=int)
    p_to_g = np.full(len(pred), -1, dtype=int)
    for p in np.argsort(-pred[:, 4], kind="stable"):
        if pred[p, 4] < confidence:
            continue
        eligible = g_to_p < 0
        if class_aware:
            eligible &= gt[:, 4] == pred[p, 5]
        scores = np.where(eligible, matrix[:, p], -1)
        if len(scores) and scores.max() >= threshold:
            g = int(scores.argmax())
            g_to_p[g], p_to_g[p] = p, g
    return g_to_p, p_to_g, matrix


def native_stats(gt, pred):
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou
    # Use the pinned framework matcher, including its float32 IoU thresholds.
    holder = type("Thresholds", (), {"iouv": torch.linspace(0.5, 0.95, 10)})()
    tp = np.zeros((len(pred), 10), dtype=bool)
    if len(gt) and len(pred):
        tp = BaseValidator.match_predictions(holder, torch.tensor(pred[:, 5]), torch.tensor(gt[:, 4]),
                  box_iou(torch.tensor(gt[:, :4], dtype=torch.float32), torch.tensor(pred[:, :4], dtype=torch.float32))).numpy()
    return {"tp": tp, "conf": pred[:, 4], "pred_cls": pred[:, 5], "target_cls": gt[:, 4]}


def ap_summary(stats):
    from ultralytics.utils.metrics import ap_per_class
    values = {key: np.concatenate([s[key] for s in stats]) for key in ("tp", "conf", "pred_cls", "target_cls")}
    result = ap_per_class(**values, names=dict(enumerate(CLASSES)))
    _, _, p, r, _, ap, classes = result[:7]
    return {"map50": float(ap[:, 0].mean()), "map75": float(ap[:, 5].mean()), "map50_95": float(ap.mean()),
            "classes": {CLASSES[int(c)]: {"ap50": float(ap[i, 0]), "ap75": float(ap[i, 5]), "ap50_95": float(ap[i].mean()),
                        "ap_by_iou": ap[i].tolist(), "native_best_f1_precision": float(p[i]), "native_best_f1_recall": float(r[i])}
                        for i, c in enumerate(classes)}}


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"n": 0}
    return {"n": len(values), "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
            **{key: float(np.quantile(values, q)) for key, q in (("median", .5), ("p05", .05), ("p95", .95), ("p99", .99))}}


def size_bin(short):
    return int(np.searchsorted([16, 32, 64, 128], short, side="right"))


def paired_recovery(gt, a, b, confidence=.25, threshold=.5):
    ga, _, ia = match(gt, a, threshold, confidence)
    gb, _, ib = match(gt, b, threshold, confidence)
    return {"both": (ga >= 0) & (gb >= 0), "recovered": (ga < 0) & (gb >= 0),
            "lost": (ga >= 0) & (gb < 0), "neither": (ga < 0) & (gb < 0),
            "iou_a": np.array([ia[g, p] if p >= 0 else np.nan for g, p in enumerate(ga)]),
            "iou_b": np.array([ib[g, p] if p >= 0 else np.nan for g, p in enumerate(gb)])}
