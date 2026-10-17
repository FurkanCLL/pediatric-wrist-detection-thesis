"""Independent saved-output invariants in addition to exact frozen-core replay."""

import numpy as np

from wrist_compare.common import require
from wrist_eval.adaptive import plan_rois, fuse_predictions, overlap, shift_local
from wrist_eval.metrics import validate_predictions


def equal_global(actual, expected):
    actual = np.asarray(actual, np.float32).reshape(-1, 6)
    expected = np.asarray(expected, np.float32).reshape(-1, 6)
    require(actual.shape == expected.shape and actual.tobytes() == expected.tobytes(),
            "F global predictions are not bitwise equal to D")


def valid_timing(timing, expected_passes):
    require(timing["forward_passes"] == expected_passes and
        all(np.isfinite(timing[k]) and timing[k] >= 0 for k in ("e2e_seconds", "gpu_forward_ms", "decode_seconds")) and
        timing["gpu_forward_ms"] > 0, "Invalid timing or physical model-call count")


def verify_f(item, expected, cfg):
    original = np.asarray(item["global"], np.float32).reshape(-1, 6)
    equal_global(original, expected)
    plan = item["plan"]
    h, w = plan["image_hw"]
    require(plan == plan_rois(original, (h, w), cfg), "F saved ROI plan differs from frozen planning")
    views = item["local_views"]
    require(len(views) == len(plan["rois"]) <= 2, "F ROI budget differs")
    counters = [item["global_view"]["counters"], *[v["counters"] for v in views]]
    calls = sum(v["forward_passes"] for v in counters)
    require(calls == 1+len(views) <= 3 and all(v["forward_passes"] == 1 for v in counters), "F physical model calls differ")
    require(set(item["variants"]) == {"V2"}, "F must contain only V2")
    variant = item["variants"]["V2"]
    pred = np.asarray(variant["pred"], np.float32).reshape(-1, 6)
    require(np.array_equal(pred, validate_predictions(pred, w, h)), "F output geometry/confidence ordering differs")
    diagnostics = variant["diagnostics"]
    ids = diagnostics["output_global_ids"]
    require(sorted(ids) == list(range(len(original))) and len(pred) == len(original), "F added or removed global rows")
    restored = pred[np.argsort(ids)]
    candidates = {c["global_id"] for c in plan["candidates"]}
    protected = [i for i in range(len(original)) if i not in candidates]
    require(np.array_equal(restored[protected], original[protected]) and np.array_equal(restored[:, 5], original[:, 5]),
            "F changed protected fields/class IDs")
    require(np.array_equal(pred[pred[:, 4] >= .5], original[original[:, 4] >= .5]), "F changed confidence-.50 output")
    matches = diagnostics["matches"]
    require(len({p["global_id"] for p in matches}) == len(matches) and
            len({p["evidence_id"] for p in matches}) == len(matches), "F evidence assignment is not one-to-one")
    for index in candidates:
        require(original[index, 4] <= restored[index, 4] <= min(float(original[index, 4])+.1, .35), "F confidence cap failed")
        if np.array_equal(restored[index, :4], original[index, :4]):
            continue
        pair = next(p for p in matches if p["global_id"] == index and p["mutable"])
        local = np.asarray(diagnostics["evidence"][pair["evidence_id"]]["row"], np.float64)
        require(local[4] >= .5 and pair["iou"] >= .5, "F geometry evidence is ineligible")
        blended = (.75*original[index, :4].astype(np.float64)+.25*local[:4]).astype(np.float32)
        blended[[0, 2]] = np.clip(blended[[0, 2]], 0, w)
        blended[[1, 3]] = np.clip(blended[[1, 3]], 0, h)
        require(np.array_equal(blended, restored[index, :4]), "F geometry did not use frozen quarter blend")
        for j in range(len(original)):
            if j != index and original[j, 5] == original[index, 5]:
                require(overlap(restored[index, :4], restored[j, :4]) <= max(.7, overlap(original[index, :4], original[j, :4])),
                        "F introduced/increased an above-.70 collision")
    for roi, view in zip(plan["rois"], views):
        require(roi["id"] == view["roi_id"], "Local view order differs")
        shift_local(view["predictions"], roi, plan["image_hw"])
    # Offline replay checks every deterministic filtering, matching, veto and tie decision without a model call.
    replay = fuse_predictions(original, views, plan, cfg, "V2")
    require(np.array_equal(replay["predictions"], pred) and replay["diagnostics"] == diagnostics, "Frozen V2 replay differs")
    return {"rois": len(views), "calls": calls, "eligible_local_evidence": len(diagnostics["evidence"]),
        "matches": len(matches), "mutable_matches": sum(p["mutable"] for p in matches),
        "protected_matches": diagnostics["protected_matches"], "confidence_updates": len(diagnostics["score_changes"]),
        "geometry_changes": sum(g["reason"] == "accepted" for g in diagnostics["geometry"]),
        "collision_vetoes": sum(g["reason"] == "collision_veto" for g in diagnostics["geometry"])}
