"""Strict source geometry validation and approved detector label conversion."""

import math
from numbers import Real

from wrist_audit.annotations import object_box, validate_box
from .common import CLASSES


def target_boxes(annotation):
    width, height = annotation["size"]["width"], annotation["size"]["height"]
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Invalid source dimensions")
    result = []
    for index, obj in enumerate(annotation["objects"]):
        name = obj["classTitle"]
        if name in ("axis", "foreignbody"):
            continue
        if name not in CLASSES:
            raise ValueError("Unknown source category")
        points = obj.get("points", {}).get("exterior", [])
        polygon = name in ("bonelesion", "periostealreaction")
        if (polygon and len(points) < 3) or (not polygon and len(points) != 2):
            raise ValueError("Unexpected source geometry")
        if any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                                 for v in p) for p in points):
            raise ValueError("Nonfinite or malformed source points")
        if polygon:
            area_twice = sum(points[i][0] * points[(i + 1) % len(points)][1]
                             - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
            if area_twice == 0:
                raise ValueError("Degenerate source polygon")
        box = object_box(obj)
        issue = validate_box(box, width, height)
        if issue:
            raise ValueError("Invalid source box: " + issue)
        result.append({"source_object_index": index, "class_id": CLASSES.index(name),
                       "source_class": name, "xyxy": box})
    return result


def yolo_bytes(record):
    lines = []
    width, height = record["width"], record["height"]
    for obj in record["boxes"]:
        left, top, right, bottom = obj["xyxy"]
        values = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
        # Round-trip double precision keeps border coordinates without clipping.
        lines.append(str(obj["class_id"]) + " " + " ".join(format(v, ".17g") for v in values))
    return (("\n".join(lines) + "\n") if lines else "").encode("ascii")
