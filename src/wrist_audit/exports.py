"""Recheck all exports from the archive already proven identical to local files."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

from .annotations import compare_objects, object_box, parse_yolo, read_voc, validate_box
from .audit import validate_output_path, write_csv, write_json


def check_exports(output):
    receipt = json.loads((output / "source_release.json").read_text(encoding="utf-8"))
    provenance = receipt["folder_archive"]
    counts = provenance["counts"]
    if not provenance["official_md5_match"] or counts.get("identical_local") != counts["archive_files"]:
        raise ValueError("First verify archive equivalence with the extracted local source")
    data = (output / "folder_structure.zip").read_bytes()
    if hashlib.md5(data).hexdigest() != provenance["md5"]:
        raise ValueError("Annotation archive changed after provenance verification")
    checks, flags, issues = Counter(), Counter(), []
    class_counts = {name: Counter() for name in ("supervisely", "voc", "yolo")}
    deltas = {"voc": 0.0, "yolo": 0.0}
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        names = sorted(n for n in source.namelist() if n.startswith("supervisely/wrist/ann/") and n.endswith(".json"))
        for name in names:
            stem = Path(name).stem
            annotation = json.loads(source.read(name))
            width, height = annotation["size"]["width"], annotation["size"]["height"]
            primary = [(o["classTitle"], object_box(o)) for o in annotation["objects"] if o["classTitle"] != "axis"]
            checks["primary_detection_objects"] += len(primary)
            size, voc, voc_flags = read_voc(io.BytesIO(source.read("pascalvoc/" + stem + ".xml")))
            flags.update(voc_flags)
            checks["voc_images_checked"] += 1
            checks["voc_missing_sizes"] += size is None
            checks["voc_size_mismatches"] += size is not None and size != (width, height)
            checks["voc_objects"] += len(voc)
            for _, box in voc:
                problem = validate_box(box, width, height)
                if problem:
                    issues.append({"filestem": stem, "check": "voc_box", "detail": problem})
            yolo, errors = parse_yolo(source.read("yolov5/labels/" + stem + ".txt").decode("utf-8"), width, height)
            for format_name, objects in (("supervisely", primary), ("voc", voc), ("yolo", yolo)):
                class_counts[format_name].update(n for n, _ in objects)
            checks["yolo_images_checked"] += 1
            checks["yolo_objects"] += len(yolo)
            for line, error in errors:
                issues.append({"filestem": stem, "check": "yolo_box", "detail": f"line {line}: {error}"})
            for format_name, objects in (("voc", voc), ("yolo", yolo)):
                failures, delta = compare_objects(primary, objects, 1.0)
                checks[format_name + "_mismatched_objects"] += failures
                deltas[format_name] = max(deltas[format_name], delta)
                if failures:
                    issues.append({"filestem": stem, "check": format_name + "_comparison", "detail": str(failures)})
    result = {
        "source": "Official folder_structure.zip, previously compared with every extracted counterpart",
        "archive_md5": provenance["md5"], "equivalent_local_files": counts["identical_local"],
        "primary_dimensions": "Supervisely; verified against all original PNG headers by the full audit",
        "comparison_tolerance_pixels": 1.0, "crosschecks": dict(checks),
        "maximum_export_delta_pixels": deltas,
        "voc_flags": {f"{k}:{v}": n for (k, v), n in flags.items()},
        "class_counts_by_format": {k: dict(v) for k, v in class_counts.items()},
        "issue_counts": dict(Counter(x["check"] for x in issues)),
        "issue_details": dict(Counter(x["check"] + ":" + x["detail"] for x in issues)),
    }
    write_json(output / "export_crosscheck.json", result)
    write_csv(output / "export_issues.csv", issues, ["filestem", "check", "detail"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit"))
    args = parser.parse_args()
    validate_output_path(Path("data/raw"), args.output)
    print(json.dumps(check_exports(args.output), indent=2))


if __name__ == "__main__":
    main()
