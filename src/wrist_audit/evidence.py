"""Export compact aggregate evidence; individual audit records are never copied."""

import argparse
import copy
import csv
import json
from pathlib import Path

from .audit import validate_output_path, write_csv, write_json


# Keep a corrected export check separate from the expensive image/hash scan.
def reconcile_exports(public, local):
    path = local / "export_crosscheck.json"
    if not path.is_file():
        return None
    exports = json.loads(path.read_text(encoding="utf-8"))
    checks = exports["crosschecks"]
    image_count = public["inventory"]["images"]["files"]
    if checks["voc_images_checked"] != image_count or checks["yolo_images_checked"] != image_count:
        raise ValueError("Export recheck does not cover the full image census")
    if checks["primary_detection_objects"] != public["annotations"]["detection_objects"]:
        raise ValueError("Export recheck has different primary object counts")
    # Preserve the initial failed receipt locally and disclose the resolved parser defect.
    parse_errors = public["issue_counts"].get("annotation_parse", 0)
    if parse_errors:
        with (local / "issues.csv").open(encoding="utf-8", newline="") as stream:
            original = [r for r in csv.DictReader(stream) if r["check"] == "annotation_parse"]
        expected = "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
        if len(original) != image_count or any(r["detail"] != expected for r in original):
            raise ValueError("Unresolved annotation parser errors; cannot replace this coverage")
        public["resolved_tooling_findings"] = {
            "initial_voc_parser_errors": parse_errors,
            "cause": "Original VOC size elements are empty; the parser incorrectly required dimensions",
            "resolution": "Optional VOC dimensions tested; every export rechecked against Supervisely dimensions",
            "initial_full_scan_retained_locally": True,
        }
    public["annotations"]["crosschecks"].update(checks)
    for key in ("maximum_export_delta_pixels", "voc_flags"):
        public["annotations"][key] = exports[key]
    export_checks = {"annotation_parse", "yolo_box", "voc_box", "voc_comparison", "yolo_comparison"}
    for key in ("issue_counts", "blocking_issue_counts"):
        public[key] = {k: v for k, v in public[key].items() if k not in export_checks}
    public["issue_counts"].update(exports["issue_counts"])
    with (local / "export_issues.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["check"] == "yolo_box" and row["detail"].endswith(": rounding_at_border"):
                continue
            key = row["check"]
            public["blocking_issue_counts"][key] = public["blocking_issue_counts"].get(key, 0) + 1
    public["verification_batches"] = {
        "image_metadata_and_primary_geometry": "Full extracted-source audit; original receipt retained locally",
        "annotation_exports": "Complete recheck from the checksum-verified archive, identical to extracted files",
    }
    return exports


def export_evidence(local, destination):
    summary = json.loads((local / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((local / "source_release.json").read_text(encoding="utf-8"))
    public = copy.deepcopy(summary)
    exports = reconcile_exports(public, local)
    # Object tags can contain free text. Only their aggregate count is shareable.
    tag_counts = public["annotations"].pop("object_tag_counts")
    public["annotations"]["object_tags_total"] = sum(tag_counts.values())
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "aggregate_statistics.json", public)
    write_json(destination / "source_release.json", provenance)
    if exports is not None:
        write_json(destination / "export_crosscheck.json", exports)
    rows = []
    for name, stats in public["classes"].items():
        rows.append({"source_class": name, "source_id": stats["source_id"],
                     "proposed_id": stats["proposed_id"], "source_shape": stats["source_shape"],
                     "instances": stats["instances"], "images": stats["images"], "patients": stats["patients"],
                     "median_relative_area_percent": 100 * stats["relative_area"].get("median", 0),
                     "median_short_side_at_1024": stats["short_side_at_1024"].get("median", 0),
                     "short_side_below_16_at_1024": stats["short_side_below_16_at_1024"]})
    write_csv(destination / "class_statistics.csv", rows,
              ["source_class", "source_id", "proposed_id", "source_shape", "instances", "images", "patients",
               "median_relative_area_percent", "median_short_side_at_1024", "short_side_below_16_at_1024"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--destination", type=Path, default=Path("docs/evidence/dataset_audit"))
    args = parser.parse_args()
    validate_output_path(Path("data/raw"), args.destination)
    export_evidence(args.local, args.destination)


if __name__ == "__main__":
    main()
