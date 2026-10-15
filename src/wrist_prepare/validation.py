"""Independent decoding, pixel arithmetic and source-to-label verification."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math

import cv2
import numpy as np

from .common import CLASSES, SPLITS, digest, json_bytes, protected_write


def check_pair(record, raw, prepared):
    stem, split = record["filestem"], record["split"]
    original_bytes = (raw / "images" / (stem + ".png")).read_bytes()
    output_bytes = (prepared / "images" / split / (stem + ".png")).read_bytes()
    if digest(original_bytes) != record["source_image_sha256"]:
        raise ValueError("Source image hash changed")
    # OpenCV provides a decoder independent of the Pillow conversion path.
    original = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    converted = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    shape = (record["height"], record["width"])
    if original is None or original.dtype != np.uint16 or original.shape != shape:
        raise ValueError("Invalid original pixel representation")
    if converted is None or converted.dtype != np.uint8 or converted.shape != shape:
        raise ValueError("Invalid prepared shape, bit depth or channel count")
    if output_bytes[:8] != b"\x89PNG\r\n\x1a\n" or output_bytes[24:26] != b"\x08\x00":
        raise ValueError("Prepared file is not an 8-bit grayscale PNG")
    # Floating-point reference arithmetic independently checks every converted pixel.
    expected = np.rint(original.astype(np.float64) * (255.0 / 65535.0)).astype(np.uint8)
    if not np.array_equal(converted, expected):
        raise ValueError("Prepared pixels differ from the fixed linear mapping")
    annotation_bytes = (raw / "supervisely/wrist/ann" / (stem + ".json")).read_bytes()
    if digest(annotation_bytes) != record["source_annotation_sha256"]:
        raise ValueError("Source annotation hash changed")
    annotation = json.loads(annotation_bytes)
    if (annotation["size"]["height"], annotation["size"]["width"]) != shape:
        raise ValueError("Source annotation dimensions changed")
    # Reconstruct envelopes directly from source points, without the label generator.
    reference = []
    for obj in annotation["objects"]:
        if obj["classTitle"] in ("axis", "foreignbody"):
            continue
        points = np.asarray(obj["points"]["exterior"], dtype=np.float64)
        reference.append((CLASSES.index(obj["classTitle"]),
                          np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])))
    lines = (prepared / "labels" / split / (stem + ".txt")).read_text(encoding="ascii").splitlines()
    if len(lines) != len(reference):
        raise ValueError("Prepared label count differs from source targets")
    counts, maximum_delta = Counter(), 0.0
    for line, (class_id, box) in zip(lines, reference):
        fields = line.split()
        if len(fields) != 5 or fields[0] != str(class_id):
            raise ValueError("Incorrect detector class or label structure")
        cx, cy, width, height = map(float, fields[1:])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, width, height)) or min(width, height) <= 0:
            raise ValueError("Invalid normalized label coordinates")
        actual = np.array([(cx - width / 2) * shape[1], (cy - height / 2) * shape[0],
                           (cx + width / 2) * shape[1], (cy + height / 2) * shape[0]])
        delta = float(np.max(np.abs(actual - box)))
        if delta > 1e-8 or np.any(actual[:2] < -1e-8) or actual[2] > shape[1] + 1e-8 or actual[3] > shape[0] + 1e-8:
            raise ValueError("Prepared box does not match original source geometry")
        maximum_delta = max(maximum_delta, delta)
        counts[CLASSES[class_id]] += 1
    return {"objects": len(reference), "empty": not reference, "counts": counts,
            "maximum_coordinate_delta_pixels": maximum_delta,
            "prepared_sha256": digest(output_bytes), "minimum": int(converted.min()), "maximum": int(converted.max())}


def validate_dataset(records, raw, prepared, workers=4):
    stems = [r["filestem"] for r in records]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate original image assignments")
    patients = {}
    for record in records:
        previous = patients.setdefault(record["patient_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("A patient occurs in multiple splits")
    for split in SPLITS:
        expected = {r["filestem"] for r in records if r["split"] == split}
        images = {p.stem for p in (prepared / "images" / split).glob("*.png")}
        labels = {p.stem for p in (prepared / "labels" / split).glob("*.txt")}
        if images != expected or labels != expected:
            raise ValueError("Prepared image/label census differs from frozen assignments")
    totals, hashes, maximum_delta = Counter(), [], 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda r: check_pair(r, raw, prepared), records)
        for index, (record, result) in enumerate(zip(records, results), 1):
            totals["objects"] += result["objects"]
            totals["empty_labels"] += result["empty"]
            totals.update({"class:" + k: v for k, v in result["counts"].items()})
            maximum_delta = max(maximum_delta, result["maximum_coordinate_delta_pixels"])
            hashes.append((record["filestem"], result["prepared_sha256"]))
            if index % 1000 == 0:
                print(f"Independent image/pixel/label checks: {index}/{len(records)}", flush=True)
    return {"images_checked": len(records), "patients": len(patients), "patient_overlap": 0,
            "objects_checked": totals["objects"], "empty_labels": totals["empty_labels"],
            "class_counts": {c: totals["class:" + c] for c in CLASSES},
            "maximum_coordinate_delta_pixels": maximum_delta,
            "coordinate_tolerance_pixels": 1e-8, "pixel_mapping_mismatches": 0,
            "original_image_hashes_verified": len(records), "original_annotation_hashes_verified": len(records),
            "decoder": "OpenCV IMREAD_UNCHANGED; uint16 source and uint8 single-channel output",
            "pixel_reference": "float64 rint(source * 255 / 65535), independently of integer conversion",
            "prepared_image_manifest_sha256": digest("".join(f"{s}\t{h}\n" for s, h in hashes).encode())}
