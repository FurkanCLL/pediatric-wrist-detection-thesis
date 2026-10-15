"""Export compact aggregate evidence; individual audit records are never copied."""

import argparse
import copy
import json
from pathlib import Path

from .audit import validate_output_path, write_csv, write_json


def export_evidence(local, destination):
    summary = json.loads((local / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((local / "source_release.json").read_text(encoding="utf-8"))
    public = copy.deepcopy(summary)
    # Object tags can contain free text. Only their aggregate count is shareable.
    tag_counts = public["annotations"].pop("object_tag_counts")
    public["annotations"]["object_tags_total"] = sum(tag_counts.values())
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "aggregate_statistics.json", public)
    write_json(destination / "source_release.json", provenance)
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
