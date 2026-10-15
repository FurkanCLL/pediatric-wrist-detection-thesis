"""Run the source audit; detailed records stay in the ignored local output folder."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np
import PIL

from .annotations import (SOURCE_CLASSES, PROPOSED_IDS, compare_objects, object_box,
                          read_voc, read_yolo, validate_box)
from .images import distribution, inspect_image
from .metadata import read_metadata


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def index_files(folder, suffix):
    paths = sorted(folder.rglob("*" + suffix))
    stems = Counter(p.stem for p in paths)
    return {p.stem: p for p in paths}, {"files": len(paths), "duplicate_stems": sum(v - 1 for v in stems.values())}


def validate_output_path(raw, output):
    # A mistyped CLI argument must never overwrite any part of the raw source.
    raw, output = raw.resolve(), output.resolve()
    # Protect sibling source folders too when the supplied dataset is under data/raw.
    source_root = next((p for p in (raw, *raw.parents) if p.name == "raw"), raw)
    if source_root == output or source_root in output.parents or output in raw.parents:
        raise ValueError("Output must be separate from the raw dataset and its ancestors")


def run_audit(raw, output, decode_sample=512, workers=4):
    validate_output_path(raw, output)
    output.mkdir(parents=True, exist_ok=True)
    print("Inspecting source inventory and metadata", flush=True)
    source_files = sorted(p for p in raw.rglob("*") if p.is_file())
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_files}
    metadata, metadata_summary, metadata_problems = read_metadata(raw / "dataset.csv")
    issues = [{"filestem": s, "check": "metadata", "detail": e} for s, e in metadata_problems]
    images, image_index = index_files(raw / "images", ".png")
    json_paths, json_index = index_files(raw / "supervisely/wrist/ann", ".json")
    voc_paths, voc_index = index_files(raw / "pascalvoc", ".xml")
    yolo_paths, yolo_index = index_files(raw / "yolov5/labels", ".txt")
    source_meta = json.loads((raw / "supervisely/meta.json").read_text(encoding="utf-8"))
    shapes = {x["title"]: x["shape"] for x in source_meta["classes"]}
    # Read this small YAML subset without adding a general YAML dependency.
    yolo_names = tuple(line[2:].strip() for line in (raw / "yolov5/meta.yaml").read_text().splitlines() if line.startswith("- "))
    if yolo_names != SOURCE_CLASSES or set(shapes) != set(SOURCE_CLASSES) | {"axis"}:
        raise ValueError("Source taxonomy changed; review the mapping before auditing")
    joins = {}
    for name, mapping in (("metadata", metadata), ("supervisely", json_paths), ("voc", voc_paths), ("yolo", yolo_paths)):
        missing, extra = set(images) - set(mapping), set(mapping) - set(images)
        joins[name] = {"missing_for_image": len(missing), "without_image": len(extra)}
        issues.extend({"filestem": s, "check": name + "_join", "detail": "missing"} for s in sorted(missing))
        issues.extend({"filestem": s, "check": name + "_join", "detail": "orphan"} for s in sorted(extra))

    instances, class_images, class_patients = Counter(), defaultdict(set), defaultdict(set)
    box_values = defaultdict(list)
    image_annotations, object_records = {}, []
    geometry_counts, tags, object_tags, descriptions, flags = Counter(), Counter(), Counter(), Counter(), Counter()
    crosschecks = Counter()
    max_deltas = {"voc": 0.0, "yolo": 0.0}
    annotation_sizes = {}
    for number, (stem, path) in enumerate(sorted(json_paths.items()), 1):
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
            width, height = annotation["size"]["width"], annotation["size"]["height"]
            annotation_sizes[stem] = (width, height)
            primary, axes = [], 0
            for tag in annotation.get("tags", []):
                tags[tag if isinstance(tag, str) else tag["name"]] += 1
            for obj in annotation["objects"]:
                name = obj["classTitle"]
                if name not in shapes:
                    issues.append({"filestem": stem, "check": "unknown_class", "detail": name})
                    continue
                geometry_counts[name + ":" + shapes[name]] += 1
                object_tags.update(str(t) for t in obj.get("tags", []))
                if obj.get("description"):
                    descriptions["nonempty_object_description"] += 1
                try:
                    box = object_box(obj)
                except (ValueError, TypeError) as exc:
                    issues.append({"filestem": stem, "check": "source_geometry", "detail": str(exc)})
                    continue
                if obj.get("points", {}).get("interior"):
                    crosschecks["objects_with_interior_points"] += 1
                if name == "axis":
                    axes += 1
                    points = obj["points"]["exterior"]
                    if len(points) != 2 or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height or points[0] == points[-1]:
                        issues.append({"filestem": stem, "check": "axis_geometry", "detail": "invalid_line"})
                    continue
                shape = shapes[name]
                if (shape == "rectangle" and len(obj["points"]["exterior"]) != 2) or (shape == "polygon" and len(obj["points"]["exterior"]) < 3):
                    issues.append({"filestem": stem, "check": "source_geometry", "detail": "unexpected_point_count"})
                issue = validate_box(box, width, height)
                if issue:
                    issues.append({"filestem": stem, "check": "source_box", "detail": issue})
                primary.append((name, box))
                instances[name] += 1
                class_images[name].add(stem)
                if stem in metadata:
                    class_patients[name].add(metadata[stem]["patient_id"])
                bw, bh = box[2] - box[0], box[3] - box[1]
                relative_area = bw * bh / (width * height)
                short_side = min(bw, bh)
                box_values[name].append((relative_area, short_side, bw / bh if bh > 0 else 0, short_side * 1024 / max(width, height)))
                object_records.append({"filestem": stem, "class": name, "source_shape": shape,
                                       "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3],
                                       "relative_area": relative_area, "valid": issue is None})
            image_annotations[stem] = {"boxes": len(primary), "eight_class_boxes": sum(n in PROPOSED_IDS for n, _ in primary),
                                       "axes": axes, "classes": sorted({n for n, _ in primary})}
            crosschecks["duplicate_source_boxes"] += sum(v - 1 for v in Counter(primary).values())
            if stem in voc_paths:
                size, objects, voc_flags = read_voc(voc_paths[stem])
                flags.update(voc_flags)
                failures, delta = compare_objects(primary, objects, 1.0)
                crosschecks["voc_images_checked"] += 1
                crosschecks["voc_mismatched_objects"] += failures
                crosschecks["voc_missing_sizes"] += size is None
                crosschecks["voc_size_mismatches"] += size is not None and size != (width, height)
                max_deltas["voc"] = max(max_deltas["voc"], delta)
                for _, box in objects:
                    problem = validate_box(box, width, height)
                    if problem:
                        issues.append({"filestem": stem, "check": "voc_box", "detail": problem})
                if failures:
                    issues.append({"filestem": stem, "check": "voc_comparison", "detail": str(failures)})
            if stem in yolo_paths:
                objects, errors = read_yolo(yolo_paths[stem], width, height)
                failures, delta = compare_objects(primary, objects, 1.0)
                crosschecks["yolo_images_checked"] += 1
                crosschecks["yolo_mismatched_objects"] += failures
                max_deltas["yolo"] = max(max_deltas["yolo"], delta)
                for line, error in errors:
                    issues.append({"filestem": stem, "check": "yolo_box", "detail": f"line {line}: {error}"})
                if failures:
                    issues.append({"filestem": stem, "check": "yolo_comparison", "detail": str(failures)})
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
            issues.append({"filestem": stem, "check": "annotation_parse", "detail": type(exc).__name__ + ": " + str(exc)})
        if number % 5000 == 0:
            print(f"Annotations checked: {number}", flush=True)

    # Stable hash order avoids a patient-prefix-biased sample; add rare-class coverage.
    sample_order = sorted(images, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    decode_stems = set(sample_order[:decode_sample])
    for name in SOURCE_CLASSES:
        candidates = sorted(class_images[name], key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        decode_stems.update(candidates[:8])
    empty_stems = [s for s in sample_order if image_annotations.get(s, {}).get("boxes") == 0]
    decode_stems.update(empty_stems[:8])
    image_records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda p: inspect_image(p, p.stem in decode_stems), (images[s] for s in sorted(images)))
        for number, record in enumerate(results, 1):
            stem = record["filestem"]
            if "error" in record:
                issues.append({"filestem": stem, "check": "image_integrity", "detail": record["error"]})
            if "width" in record and stem in annotation_sizes and annotation_sizes[stem] != (record["width"], record["height"]):
                issues.append({"filestem": stem, "check": "image_annotation_size", "detail": "mismatch"})
            for warning in record.get("warnings", []):
                issues.append({"filestem": stem, "check": "image_warning", "detail": warning})
            if record.get("constant_pixels"):
                issues.append({"filestem": stem, "check": "constant_image", "detail": "sampled_decode"})
            image_records.append(record)
            if number % 5000 == 0:
                print(f"PNG integrity and hash checks: {number}", flush=True)

    good = [r for r in image_records if "error" not in r]
    decoded = [r for r in good if "pixels" in r]
    hash_groups = defaultdict(list)
    for record in image_records:
        if "sha256" in record:
            hash_groups[record["sha256"]].append(record["filestem"])
    duplicate_groups = [stems for stems in hash_groups.values() if len(stems) > 1]
    for stems in duplicate_groups:
        issues.extend({"filestem": s, "check": "duplicate_image_bytes", "detail": "|".join(stems)} for s in stems)
    per_class = {}
    for name in SOURCE_CLASSES:
        values = box_values[name]
        per_class[name] = {
            "source_id": SOURCE_CLASSES.index(name), "proposed_id": PROPOSED_IDS.get(name),
            "source_shape": shapes[name], "instances": instances[name],
            "images": len(class_images[name]), "patients": len(class_patients[name]),
            "relative_area": distribution([v[0] for v in values]),
            "short_side_pixels": distribution([v[1] for v in values]),
            "aspect_ratio": distribution([v[2] for v in values]),
            "short_side_at_1024": distribution([v[3] for v in values]),
            "area_below_0_1_percent": sum(v[0] < .001 for v in values),
            "short_side_below_16_at_1024": sum(v[3] < 16 for v in values),
        }
    # Publish only aggregate data. Individual stems, patients, and file paths stay local.
    for key in ("images_per_patient", "images_per_study"):
        metadata_summary[key] = distribution(metadata_summary[key])
    csv_bytes = (raw / "dataset.csv").read_bytes()
    summary = {
        "schema_version": 1,
        "primary_annotations": "supervisely/wrist/ann; rectangle or polygon exterior envelopes; axis excluded",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
        "inventory": {"images": image_index, "supervisely": json_index, "voc": voc_index, "yolo": yolo_index,
                      "raw_files": len(source_files), "raw_bytes": sum(s[0] for s in before.values()),
                      "metadata_md5": hashlib.md5(csv_bytes).hexdigest(), "metadata_sha256": hashlib.sha256(csv_bytes).hexdigest()},
        "joins": joins, "metadata": metadata_summary,
        "annotations": {"detection_objects": sum(instances.values()), "eight_class_objects": sum(instances[n] for n in PROPOSED_IDS),
                        "axis_lines": sum(a["axes"] for a in image_annotations.values()),
                        "images_without_detection_boxes": sum(a["boxes"] == 0 for a in image_annotations.values()),
                        "images_without_eight_class_boxes": sum(a["eight_class_boxes"] == 0 for a in image_annotations.values()),
                        "images_with_foreignbody": len(class_images["foreignbody"]),
                        "geometry_counts": dict(geometry_counts), "image_tags": dict(tags),
                        "object_tag_counts": dict(object_tags), "description_counts": dict(descriptions),
                        "voc_flags": {f"{k}:{v}": n for (k, v), n in flags.items()},
                        "crosschecks": dict(crosschecks), "maximum_export_delta_pixels": max_deltas,
                        "comparison_tolerance_pixels": 1.0},
        "classes": per_class,
        "images": {
            "integrity_checked": len(image_records), "integrity_passed": len(good),
            "pixel_decoded": len(decoded), "decode_requested_base": decode_sample,
            "sample_policy": "SHA256(filestem) ascending, plus up to eight images per source class and eight empty-box images",
            "bit_depths": dict(Counter(str(r["bit_depth"]) for r in good)),
            "color_types": dict(Counter(str(r["color_type"]) for r in good)),
            "pillow_modes": dict(Counter(r["mode"] for r in good)),
            "width": distribution([r["width"] for r in good]),
            "height": distribution([r["height"] for r in good]),
            "width_height_ratio": distribution([r["width"] / r["height"] for r in good]),
            "distinct_dimensions": len({(r["width"], r["height"]) for r in good}),
            "sampled_pixel_statistics": {key: distribution([r["pixels"][key] for r in decoded])
                                        for key in ("min", "max", "mean", "p01", "p50", "p99", "zero_fraction", "maximum_code_fraction")},
            "sampled_dtypes": dict(Counter(r["pixels"]["dtype"] for r in decoded)),
            "exact_duplicate_groups": len(duplicate_groups),
            "exact_duplicate_excess_files": sum(len(g) - 1 for g in duplicate_groups),
            "duplicate_groups_across_patients": sum(len({metadata[s]["patient_id"] for s in g if s in metadata}) > 1 for g in duplicate_groups),
        },
        "issue_counts": dict(Counter(x["check"] for x in issues)),
        "blocking_issue_counts": dict(Counter(x["check"] for x in issues
            if x["check"] != "duplicate_image_bytes" and not (
                x["check"] == "yolo_box" and x["detail"].endswith(": rounding_at_border")))),
    }
    # Include every raw file in a local content manifest, not just the selected format.
    image_hashes = {images[r["filestem"]]: r["sha256"] for r in image_records if "sha256" in r}
    manifest = []
    for path in source_files:
        digest = image_hashes.get(path)
        if digest is None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"path": path.relative_to(raw).as_posix(), "bytes": before[path][0], "sha256": digest})
    summary["inventory"]["content_manifest_sha256"] = hashlib.sha256(
        "".join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in manifest).encode()
    ).hexdigest()
    after_files = sorted(p for p in raw.rglob("*") if p.is_file())
    summary["inventory"]["raw_stat_snapshot_unchanged"] = source_files == after_files and all(
        before[p] == (p.stat().st_size, p.stat().st_mtime_ns) for p in source_files
    )
    write_csv(output / "source_manifest.csv", manifest, ["path", "bytes", "sha256"])
    write_csv(output / "issues.csv", issues, ["filestem", "check", "detail"])
    write_csv(output / "objects.csv", object_records,
              ["filestem", "class", "source_shape", "xmin", "ymin", "xmax", "ymax", "relative_area", "valid"])
    write_csv(output / "image_records.csv", (
        dict(r, **{k: metadata.get(r["filestem"], {}).get(k, "") for k in ("patient_id", "study_number", "timehash")},
             **image_annotations.get(r["filestem"], {})) for r in image_records),
        ["filestem", "patient_id", "study_number", "timehash", "sha256", "bytes", "width", "height", "bit_depth", "color_type", "mode", "boxes", "eight_class_boxes", "axes", "classes", "error"])
    write_json(output / "pixel_sample.json", decoded)
    write_json(output / "summary.json", summary)
    print(json.dumps({"images": len(images), "patients": metadata_summary["patients"],
                      "detection_objects": sum(instances.values()), "issue_counts": summary["issue_counts"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/grazpedwri-dx"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--decode-sample", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.decode_sample < 0 or not 1 <= args.workers <= 8:
        parser.error("decode-sample must be nonnegative and workers must be between 1 and 8")
    summary = run_audit(args.raw, args.output, args.decode_sample, args.workers)
    # Findings are preserved even when the audit discovers a preparation blocker.
    return 2 if summary["blocking_issue_counts"] or not summary["inventory"]["raw_stat_snapshot_unchanged"] else 0


if __name__ == "__main__":
    sys.exit(main())
