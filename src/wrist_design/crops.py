"""Context-preserving crop geometry for training-only offline prototypes."""

import math


class CropRejected(ValueError):
    pass


def validate_boxes(row):
    width, height = row["width"], row["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid source dimensions")
    for box in row["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
            raise ValueError("Non-finite source box")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and 0 <= box["class_id"] < 8):
            raise ValueError("Degenerate or out-of-bounds source annotation")


def visible_fraction(box, rect):
    x1, y1, x2, y2 = box
    left, top, right, bottom = rect
    area = max(0, min(x2, right)-max(x1, left)) * max(0, min(y2, bottom)-max(y1, top))
    return area / ((x2-x1)*(y2-y1))


def transform_boxes(row, rect, settings, target=None):
    validate_boxes(row)
    left, top, right, bottom = rect
    if not (0 <= left < right <= row["width"] and 0 <= top < bottom <= row["height"]):
        raise ValueError("Invalid crop rectangle")
    result, indices, visibility, omitted = [], [], [], [0]*8
    scale = 768 / max(right-left, bottom-top)
    clipped = 0
    for i, box in enumerate(row["boxes"]):
        fraction = visible_fraction(box["xyxy"], rect)
        # Retain every fracture in full; never turn a positive source into a nominal negative crop.
        if (box["class_id"] == 0 or i == target) and fraction < 1-1e-9:
            raise CropRejected("fracture_or_target_not_fully_retained")
        if fraction == 0:
            omitted[box["class_id"]] += 1
            continue
        # Reject the whole crop rather than deleting an annotation on a visible object fragment.
        if fraction < settings["minimum_visible_fraction"]:
            raise CropRejected("partially_visible_annotation")
        x1, y1, x2, y2 = box["xyxy"]
        coords = [max(x1, left)-left, max(y1, top)-top, min(x2, right)-left, min(y2, bottom)-top]
        if min(coords[2]-coords[0], coords[3]-coords[1])*scale < settings["minimum_retained_side_at_768"]:
            raise CropRejected("retained_annotation_too_small")
        result.append({"class_id": box["class_id"], "xyxy": coords})
        indices.append(i)
        visibility.append(fraction)
        clipped += int(fraction < 1-1e-9)
    return {"boxes": result, "source_indices": indices, "visible_fractions": visibility,
            "omitted_objects": omitted, "clipped_annotations": clipped}


def full_image(row, reason):
    return {"accepted": False, "reason": reason, "rect": [0, 0, row["width"], row["height"]],
            "boxes": [{"class_id": b["class_id"], "xyxy": list(b["xyxy"])} for b in row["boxes"]],
            "source_indices": list(range(len(row["boxes"]))), "visible_fractions": [1.]*len(row["boxes"]),
            "omitted_objects": [0]*8, "clipped_annotations": 0}


def propose_crop(row, settings, targets, weights):
    if row["split"] != "train":
        raise ValueError("Ground-truth crop proposals are training-only")
    validate_boxes(row)
    if any(b["class_id"] == 5 for b in row["boxes"]):
        return full_image(row, "bonelesion_coverage_protected")
    width, height = row["width"], row["height"]
    eligible = []
    for i, box in enumerate(row["boxes"]):
        x1, y1, x2, y2 = box["xyxy"]
        short = min(x2-x1, y2-y1)*768/max(width, height)
        if box["class_id"] in targets and short < settings["target_short_side_at_768"]:
            eligible.append((-weights[box["class_id"]], short, tuple(box["xyxy"]), i))
    if not eligible:
        return full_image(row, "no_small_eligible_target")
    _, short, _, index = min(eligible)
    x1, y1, x2, y2 = row["boxes"][index]["xyxy"]
    fraction = max(settings["minimum_dimension_fraction"], short/settings["target_short_side_at_768"],
                   settings["target_context_multiple"]*max((x2-x1)/width, (y2-y1)/height))
    if fraction > settings["maximum_dimension_fraction"]:
        return full_image(row, "insufficient_zoom_with_context")
    # Keep approximately the source aspect ratio, rounding outward so context is not reduced.
    crop_w, crop_h = math.ceil(width*fraction), math.ceil(height*fraction)
    left = min(max(0, math.floor((x1+x2-crop_w)/2)), width-crop_w)
    top = min(max(0, math.floor((y1+y2-crop_h)/2)), height-crop_h)
    rect = [left, top, left+crop_w, top+crop_h]
    try:
        labels = transform_boxes(row, rect, settings, index)
    except CropRejected as error:
        return full_image(row, str(error))
    return {"accepted": True, "reason": "accepted", "rect": rect, "target_index": index,
            "target_class": row["boxes"][index]["class_id"], "zoom": max(width, height)/max(crop_w, crop_h),
            "source_target_short_at_768": short, "crop_target_short_at_768": short*max(width, height)/max(crop_w, crop_h),
            **labels}


def crop_pixels(image, plan):
    # Return independent pixel storage; the caller's source array is never changed.
    left, top, right, bottom = plan["rect"]
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError("Crop rectangle does not fit source pixels")
    return image[top:bottom, left:right].copy()
