"""Approved CG-ALR planning and fusion; no model, filesystem or ground-truth inputs."""

import math
import time
import numpy as np

from .metrics import ious, validate_predictions


# Validate without reordering: original row IDs are part of deterministic fusion.
def checked_rows(predictions, width, height):
    rows = np.asarray(predictions, dtype=np.float32)
    if rows.size == 0:
        rows = rows.reshape(0, 6)
    if rows.ndim != 2 or rows.shape[1] != 6 or len(rows) > 300:
        raise ValueError("Expected at most 300 prediction rows with six columns")
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive integers")
    validate_predictions(rows, width, height)
    return rows.copy()


def overlap(a, b):
    return float(ious(np.asarray(a, dtype=np.float64).reshape(1, 4),
                      np.asarray(b, dtype=np.float64).reshape(1, 4))[0, 0])


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def candidate_score(confidence, short_side, class_id, settings):
    cfg = settings["candidates"]
    low, high = cfg["confidence_min_inclusive"], cfg["confidence_max_exclusive"]
    q = float(np.clip((confidence-low)/(high-low), 0, 1))
    z = float(np.clip(1-short_side/cfg["short_side_reference"][str(class_id)], 0, 1))
    score = settings["scoring"]["confidence_weight"]*q + settings["scoring"]["size_weight"]*z
    return q, z, score


# Outward integer bounds avoid losing a fractional anchor at a crop edge.
def crop_geometry(box, width, height, settings):
    margin = settings["crop"]["anchor_margin_global_768_pixels"]*max(width, height)/768
    envelope = [math.floor(max(0, float(box[0])-margin)), math.floor(max(0, float(box[1])-margin)),
                math.ceil(min(width, float(box[2])+margin)), math.ceil(min(height, float(box[3])+margin))]
    fraction = max(settings["crop"]["base_dimension_fraction"],
                   (envelope[2]-envelope[0])/width, (envelope[3]-envelope[1])/height)
    cw, ch = min(width, math.ceil(fraction*width)), min(height, math.ceil(fraction*height))
    lx, ux = max(0, envelope[2]-cw), min(envelope[0], width-cw)
    ly, uy = max(0, envelope[3]-ch), min(envelope[1], height-ch)
    if min(cw, ch) <= 0 or lx > ux or ly > uy:
        return envelope, None, "infeasible_geometry"
    if max(cw, ch) >= max(width, height):
        return envelope, None, "no_zoom"
    x = min(ux, max(lx, math.floor((float(box[0])+float(box[2])-cw)/2+.5)))
    y = min(uy, max(ly, math.floor((float(box[1])+float(box[3])-ch)/2+.5)))
    return envelope, [x, y, x+cw, y+ch], None


def plan_rois(global_predictions, image_hw, settings):
    height, width = map(int, image_hw)
    global_rows = checked_rows(global_predictions, width, height)
    cfg = settings["candidates"]
    candidates, excluded = [], []
    for index, row in enumerate(global_rows):
        confidence, class_id = float(row[4]), int(row[5])
        reason = None
        if class_id not in cfg["class_ids"]:
            reason = "nontarget_class"
        elif not cfg["confidence_min_inclusive"] <= confidence < cfg["confidence_max_exclusive"]:
            reason = "confidence_outside_band"
        short = min(float(row[2])-float(row[0]), float(row[3])-float(row[1]))*768/max(width, height)
        if reason is None and short < cfg["minimum_effective_short_side"]:
            reason = "subpixel_anchor"
        if reason:
            excluded.append({"global_id": index, "reason": reason})
            continue
        q, z, score = candidate_score(confidence, short, class_id, settings)
        envelope, crop, failure = crop_geometry(row[:4], width, height, settings)
        candidates.append({"global_id": index, "class_id": class_id, "confidence": confidence,
                           "short_768": short, "Q": q, "Z": z, "score": score,
                           "envelope": envelope, "crop": crop, "geometry_rejection": failure})
    key = lambda c: (-c["score"], -c["confidence"], c["short_768"], c["global_id"])
    feasible = sorted((c for c in candidates if c["crop"] is not None), key=key)
    heads = []
    for class_id in cfg["class_ids"]:
        first = next((c for c in feasible if c["class_id"] == class_id), None)
        if first is not None:
            heads.append(first)
    head_ids = {c["global_id"] for c in heads}
    sequence = sorted(heads, key=key) + [c for c in feasible if c["global_id"] not in head_ids]
    rois, decisions = [], []
    for candidate in sequence:
        shared = [(overlap(roi["box"], candidate["crop"]), roi["id"]) for roi in rois
                  if contains(roi["box"], candidate["envelope"])]
        shared = sorted((v for v in shared if v[0] >= settings["selection"]["roi_overlap_iou_min"]),
                        key=lambda v: (-v[0], v[1]))
        if shared:
            roi_id = shared[0][1]
            rois[roi_id]["seed_ids"].append(candidate["global_id"])
            action = "coalesced"
        elif len(rois) < settings["selection"]["max_local_rois"]:
            roi_id = len(rois)
            box = candidate["crop"]
            rois.append({"id": roi_id, "box": box, "seed_ids": [candidate["global_id"]],
                         "ideal_zoom": max(width, height)/max(box[2]-box[0], box[3]-box[1])})
            action = "selected"
        else:
            roi_id, action = None, "budget_exhausted"
        decisions.append({"global_id": candidate["global_id"], "action": action, "roi_id": roi_id})
    if len(rois) > 2:
        raise ValueError("Approved two-ROI budget exceeded")
    return {"image_hw": [height, width], "candidates": candidates, "excluded": excluded,
            "rois": rois, "selection_decisions": decisions}


# Local rows already use crop pixels, restored by the verified inference adapter.
def shift_local(rows, roi, image_hw):
    height, width = map(int, image_hw)
    x1, y1, x2, y2 = roi["box"]
    result = checked_rows(rows, x2-x1, y2-y1)
    result[:, :4] += np.array([x1, y1, x1, y1], dtype=np.float32)
    return checked_rows(result, width, height)


def boundary_sides(box, roi, image_hw, gains, settings):
    height, width = image_hw
    x1, y1, x2, y2 = roi["box"]
    gy, gx = gains
    if not np.isfinite([gy, gx]).all() or min(gy, gx) <= 0:
        raise ValueError("Invalid resize gains")
    margin = settings["crop"]["artificial_boundary_guard_local_pixels"]
    distances = [("left", x1 > 0, box[0]*gx), ("top", y1 > 0, box[1]*gy),
                 ("right", x2 < width, (x2-x1-box[2])*gx),
                 ("bottom", y2 < height, (y2-y1-box[3])*gy)]
    return [name for name, artificial, distance in distances if artificial and distance <= margin]


def local_evidence(local_views, plan, settings):
    evidence, decisions = [], []
    roi_by_id = {roi["id"]: roi for roi in plan["rois"]}
    if len(local_views) != len(roi_by_id) or {v["roi_id"] for v in local_views} != set(roi_by_id):
        raise ValueError("Local views do not match the ROI plan")
    for view in local_views:
        roi = roi_by_id[view["roi_id"]]
        cw, ch = roi["box"][2]-roi["box"][0], roi["box"][3]-roi["box"][1]
        rows = checked_rows(view["predictions"], cw, ch)
        shifted = shift_local(rows, roi, plan["image_hw"])
        for index, (row, original) in enumerate(zip(rows, shifted)):
            sides = boundary_sides(row[:4], roi, plan["image_hw"], view["gains"], settings)
            reason = "nontarget_class" if int(row[5]) not in settings["matching"]["local_classes"] else (
                "low_local_confidence" if row[4] < settings["matching"]["local_confidence_min"] else (
                "artificial_boundary" if sides else "eligible"))
            item = {"roi_id": roi["id"], "local_id": index, "reason": reason, "boundary_sides": sides}
            decisions.append(item)
            if reason == "eligible":
                evidence.append({**item, "row": original.tolist()})
    ordered = sorted(evidence, key=lambda e: (-e["row"][4], e["roi_id"], e["local_id"]))
    kept = []
    for item in ordered:
        suppressor = next((other for other in kept if other["row"][5] == item["row"][5]
                           and overlap(other["row"][:4], item["row"][:4]) > settings["matching"]["local_union_nms_iou"]), None)
        if suppressor is None:
            kept.append(item)
        else:
            decisions.append({"roi_id": item["roi_id"], "local_id": item["local_id"],
                              "reason": "local_union_suppressed", "by": [suppressor["roi_id"], suppressor["local_id"]]})
    return kept, decisions


def match_evidence(global_rows, evidence, plan, settings):
    candidates = {c["global_id"]: c for c in plan["candidates"]}
    roi_by_id = {r["id"]: r for r in plan["rois"]}
    pairs = []
    for g, row in enumerate(global_rows):
        mutable = g in candidates
        protected = int(row[5]) in settings["candidates"]["class_ids"] and row[4] >= .25
        if not (mutable or protected):
            continue
        for local_index, item in enumerate(evidence):
            local = item["row"]
            if local[5] != row[5]:
                continue
            if mutable and not contains(roi_by_id[item["roi_id"]]["box"], candidates[g]["envelope"]):
                continue
            value = overlap(row[:4], local[:4])
            if value >= settings["matching"]["global_local_iou_min"]:
                pairs.append({"global_id": g, "evidence_id": local_index, "roi_id": item["roi_id"],
                              "local_id": item["local_id"], "iou": value, "mutable": mutable,
                              "local_confidence": local[4], "global_confidence": float(row[4])})
    pairs.sort(key=lambda p: (-p["iou"], -p["local_confidence"], -p["global_confidence"],
                              p["global_id"], p["roi_id"], p["local_id"]))
    used_global, used_local, matches = set(), set(), []
    for pair in pairs:
        accepted = pair["global_id"] not in used_global and pair["evidence_id"] not in used_local
        pair["accepted"] = accepted
        if accepted:
            used_global.add(pair["global_id"])
            used_local.add(pair["evidence_id"])
            matches.append(pair)
    return matches, pairs


def confidence_update(original, local, settings):
    cap = min(float(original)+settings["fusion"]["maximum_positive_confidence_delta"],
              float(local), settings["fusion"]["maximum_updated_confidence"])
    value = np.float32(cap)
    if float(value) > cap:
        value = np.nextafter(value, np.float32(-np.inf))
    return value


def fuse_predictions(global_predictions, local_views, plan, settings, variant):
    if variant not in ("V1", "V2"):
        raise ValueError("Only the two approved fusion variants are permitted")
    height, width = plan["image_hw"]
    original = checked_rows(global_predictions, width, height)
    output = original.copy()
    evidence, local_decisions = local_evidence(local_views, plan, settings)
    matches, pairs = match_evidence(original, evidence, plan, settings)
    score_changes, geometry = [], []
    for pair in matches:
        if pair["mutable"]:
            g = pair["global_id"]
            output[g, 4] = confidence_update(original[g, 4], pair["local_confidence"], settings)
            if output[g, 4] != original[g, 4]:
                score_changes.append({"global_id": g, "before": float(original[g, 4]), "after": float(output[g, 4])})
    if variant == "V2":
        cfg = settings["fusion"]["V2"]
        for pair in sorted(matches, key=lambda p: p["global_id"]):
            if not pair["mutable"] or pair["local_confidence"] < cfg["local_confidence_for_geometry_min"]:
                continue
            g = pair["global_id"]
            local = np.asarray(evidence[pair["evidence_id"]]["row"][:4], dtype=np.float64)
            alpha = cfg["local_coordinate_weight"]
            proposed = ((1-alpha)*original[g, :4].astype(np.float64)+alpha*local).astype(np.float32)
            proposed[[0, 2]] = np.clip(proposed[[0, 2]], 0, width)
            proposed[[1, 3]] = np.clip(proposed[[1, 3]], 0, height)
            reason = "unchanged" if np.array_equal(proposed, original[g, :4]) else "accepted"
            if not np.isfinite(proposed).all() or np.any(proposed[2:] <= proposed[:2]):
                reason = "invalid_geometry"
            else:
                for j in range(len(output)):
                    if j != g and output[j, 5] == output[g, 5] and overlap(proposed, output[j, :4]) > max(
                            cfg["collision_iou"], overlap(output[g, :4], output[j, :4])):
                        reason = "collision_veto"
                        break
            if reason == "accepted":
                output[g, :4] = proposed
            geometry.append({"global_id": g, "reason": reason, "before": original[g, :4].tolist(),
                             "proposed": proposed.tolist(), "after": output[g, :4].tolist()})
    mutable_ids = {c["global_id"] for c in plan["candidates"]}
    protected_ids = [i for i in range(len(output)) if i not in mutable_ids]
    if not np.array_equal(output[protected_ids], original[protected_ids]):
        raise AssertionError("Protected global predictions changed")
    if variant == "V1" and not np.array_equal(output[:, :4], original[:, :4]):
        raise AssertionError("V1 changed global coordinates")
    ids = list(range(len(output)))
    if not np.array_equal(output, original):
        ids.sort(key=lambda i: (-float(output[i, 4]), i))
    output = checked_rows(output[ids], width, height)
    if not np.array_equal(output[output[:, 4] >= .5], original[original[:, 4] >= .5]):
        raise AssertionError("Confidence-.50 output changed")
    return {"predictions": output, "diagnostics": {"output_global_ids": ids, "evidence": evidence,
            "local_decisions": local_decisions, "pairs": pairs, "matches": matches,
            "score_changes": score_changes, "geometry": geometry,
            "protected_matches": sum(not p["mutable"] for p in matches)}}


# The callback performs one verified view; branching fusion never triggers another forward pass.
def predict_adaptive(image, infer_view, global_shape, settings, variants=None):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a prepared three-channel uint8 image")
    requested = settings["fusion"]["variants"] if variants is None else variants
    if not requested or any(v not in settings["fusion"]["variants"] for v in requested):
        raise ValueError("Unknown or empty fusion variant selection")
    start = time.perf_counter()
    global_view = infer_view(image, global_shape)
    global_rows = global_view["predictions"]
    began = time.perf_counter()
    plan = plan_rois(global_rows, image.shape[:2], settings)
    planning_seconds = time.perf_counter()-began
    views = []
    for roi in plan["rois"]:
        x1, y1, x2, y2 = roi["box"]
        view = infer_view(image[y1:y2, x1:x2], None)
        views.append({**view, "roi_id": roi["id"]})
    common_seconds = time.perf_counter()-start
    results, fusion_seconds = {}, {}
    for variant in requested:
        began = time.perf_counter()
        results[variant] = fuse_predictions(global_rows, views, plan, settings, variant)
        fusion_seconds[variant] = time.perf_counter()-began
    calls = 1+len(views)
    if calls > settings["selection"]["max_total_model_passes"] or calls > 3:
        raise AssertionError("Approved model-pass bound exceeded")
    return {"global": global_rows, "global_view": global_view, "plan": plan, "local_views": views,
            "variants": results, "passes": calls, "common_seconds": common_seconds,
            "planning_seconds": planning_seconds, "fusion_seconds": fusion_seconds}
