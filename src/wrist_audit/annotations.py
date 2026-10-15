"""Source geometry checks and comparisons; no training labels are produced."""

from collections import Counter
import math
from pathlib import Path
import xml.etree.ElementTree as ET


# These are the source export IDs, not the planned eight-class detector IDs.
SOURCE_CLASSES = (
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
)
PROPOSED_CLASSES = (
    "fracture", "metal", "text", "periostealreaction", "pronatorsign",
    "bonelesion", "softtissue", "boneanomaly",
)
PROPOSED_IDS = {name: i for i, name in enumerate(PROPOSED_CLASSES)}


def validate_box(box, width, height):
    """Use continuous image-edge coordinates; do not silently clip source boxes."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return "nonfinite_or_malformed"
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        return "nonpositive_area"
    if left < 0 or top < 0 or right > width or bottom > height:
        return "out_of_bounds"
    return None


def object_box(obj):
    points = obj.get("points", {}).get("exterior", [])
    if not points or any(len(p) != 2 for p in points):
        raise ValueError("Missing or malformed exterior points")
    if not all(math.isfinite(float(v)) for p in points for v in p):
        raise ValueError("Nonfinite exterior points")
    xs, ys = zip(*points)
    return tuple(map(float, (min(xs), min(ys), max(xs), max(ys))))


def read_voc(path):
    root = ET.parse(path).getroot()
    width, height = root.findtext("size/width"), root.findtext("size/height")
    # The original export has an empty <size/> element in every file.
    size = None if width is None and height is None else (int(width), int(height))
    objects = []
    for obj in root.findall("object"):
        box = tuple(float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objects.append(((obj.findtext("name") or "").strip(), box))
    flags = Counter(
        (flag, (obj.findtext(flag) or "").strip())
        for obj in root.findall("object") for flag in ("difficult", "truncated")
    )
    return size, objects, flags


def read_yolo(path, width, height):
    return parse_yolo(Path(path).read_text(encoding="utf-8"), width, height)


def parse_yolo(text, width, height):
    objects, errors = [], []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("field_count")
            class_id = int(fields[0])
            if not 0 <= class_id < len(SOURCE_CLASSES):
                raise ValueError("class_id")
            cx, cy, bw, bh = map(float, fields[1:])
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                raise ValueError("nonfinite")
            box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height)
            objects.append((SOURCE_CLASSES[class_id], box))
            # Six-decimal export rounding can move an edge by a tiny fraction of a pixel.
            issue = validate_box(box, width, height)
            if issue == "out_of_bounds" and validate_box(
                tuple(max(0, v) if i < 2 else min((width, height)[i - 2], v)
                      for i, v in enumerate(box)), width, height
            ) is None and all(-0.002 <= v <= limit + 0.002 for v, limit in zip(box, (width, height, width, height))):
                issue = "rounding_at_border"
            if issue:
                errors.append((line_number, issue))
        except (ValueError, OverflowError) as exc:
            errors.append((line_number, str(exc)))
    return objects, errors


def compare_objects(primary, other, tolerance):
    """Match by class and coordinates without depending on the export's object order."""
    unmatched = list(other)
    failures = 0
    maximum_delta = 0.0
    for name, box in primary:
        candidates = [(max(abs(a - b) for a, b in zip(box, other_box)), i)
                      for i, (other_name, other_box) in enumerate(unmatched) if name == other_name]
        if not candidates:
            failures += 1
            continue
        delta, index = min(candidates)
        maximum_delta = max(maximum_delta, delta)
        if delta <= tolerance:
            unmatched.pop(index)
        else:
            failures += 1
    return failures + len(unmatched), maximum_delta
