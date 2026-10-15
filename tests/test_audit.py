"""Small synthetic examples for source semantics and error-prone audit checks."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from wrist_audit.annotations import (PROPOSED_IDS, SOURCE_CLASSES, compare_objects,
                                     object_box, read_voc, read_yolo, validate_box)
from wrist_audit.audit import run_audit, validate_output_path
from wrist_audit.evidence import export_evidence
from wrist_audit.images import inspect_image
from wrist_audit.metadata import filename_disagreements, read_metadata
from wrist_audit.provenance import check_archive


def synthetic_row():
    # All identifiers in these fixtures are invented, not released patient records.
    return dict(filestem="0001_1234567890_01_WRI-L1_F010", patient_id="1",
                study_number="1", timehash="1234567890", gender="F", age="10",
                laterality="L", projection="1")


class GeometryTests(unittest.TestCase):
    def test_image_edges_are_valid_continuous_coordinates(self):
        self.assertIsNone(validate_box((0, 0, 100, 80), 100, 80))

    def test_invalid_boxes_are_not_clipped(self):
        for box, reason in [((0, 0, 0, 3), "nonpositive_area"),
                            ((4, 0, 3, 2), "nonpositive_area"),
                            ((-1, 0, 3, 2), "out_of_bounds"),
                            ((0, 0, 101, 80), "out_of_bounds"),
                            ((0, 0, float("nan"), 80), "nonfinite_or_malformed")]:
            with self.subTest(box=box):
                self.assertEqual(validate_box(box, 100, 80), reason)

    def test_polygon_envelope_preserves_subpixel_coordinates(self):
        obj = {"points": {"exterior": [[1.5, 5], [4, 2], [3, 7.25]]}}
        self.assertEqual(object_box(obj), (1.5, 2, 4, 7.25))

    def test_nonfinite_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            object_box({"points": {"exterior": [[0, 2], [float("inf"), 3]]}})

    def test_export_comparison_ignores_order_but_preserves_multiplicity(self):
        a = ("fracture", (0, 1, 10, 20))
        b = ("metal", (5, 5, 6, 6))
        self.assertEqual(compare_objects([a, b], [b, a], 0)[0], 0)
        self.assertEqual(compare_objects([a, a], [a], 0)[0], 1)

    def test_class_disagreement_is_not_a_match(self):
        self.assertEqual(compare_objects([("fracture", (0, 0, 1, 1))],
                                         [("metal", (0, 0, 1, 1))], 1)[0], 2)

    def test_source_and_proposed_ids_are_distinct(self):
        self.assertEqual(SOURCE_CLASSES[3], "fracture")
        self.assertEqual(PROPOSED_IDS["fracture"], 0)
        self.assertNotIn("foreignbody", PROPOSED_IDS)
        self.assertNotIn("axis", SOURCE_CLASSES)


class TemporaryFiles(unittest.TestCase):
    def setUp(self):
        # Keep test fixtures inside the ignored workspace output area.
        scratch = Path(__file__).resolve().parents[1] / "outputs/test_temp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class FileTests(TemporaryFiles):
    def test_voc_whitespace_and_flag_parsing(self):
        path = self.root / "annotation.xml"
        path.write_text("<annotation><size><width>100</width><height>80</height></size>"
                        "<object><name>\n fracture </name><difficult> 0 </difficult>"
                        "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>30</xmax><ymax>40</ymax>"
                        "</bndbox></object></annotation>")
        size, objects, flags = read_voc(path)
        self.assertEqual(size, (100, 80))
        self.assertEqual(objects, [("fracture", (1, 2, 30, 40))])
        self.assertEqual(flags["difficult", "0"], 1)

    def test_archive_checksum_and_content_match(self):
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "example.txt").write_bytes(b"synthetic")
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("example.txt", b"synthetic")
        expected = hashlib.md5(archive.read_bytes()).hexdigest()
        summary, errors = check_archive(raw, archive, expected)
        self.assertEqual(summary["counts"]["identical_local"], 1)
        self.assertEqual(errors, [])
        with self.assertRaises(ValueError):
            check_archive(raw, archive, "wrong")

    def test_archive_never_extracts_unsafe_members(self):
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("../unsafe.txt", b"synthetic")
        with self.assertRaises(ValueError):
            check_archive(self.root / "raw", archive, hashlib.md5(archive.read_bytes()).hexdigest())
        self.assertFalse((self.root / "unsafe.txt").exists())

    def test_yolo_class_mapping_and_original_coordinates(self):
        path = self.root / "label.txt"
        path.write_text("3 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        objects, errors = read_yolo(path, 100, 200)
        self.assertEqual(objects, [("fracture", (40, 60, 60, 140))])
        self.assertEqual(errors, [])

    def test_yolo_rejects_invalid_ids_nonfinite_and_wrong_fields(self):
        path = self.root / "label.txt"
        path.write_text("9 0 0 1 1\n3 nan 0 1 1\n3 0 1\n3 0.5 0.5 -1 1\n", encoding="utf-8")
        _, errors = read_yolo(path, 100, 100)
        self.assertEqual(len(errors), 4)

    def test_yolo_border_rounding_is_reported(self):
        path = self.root / "label.txt"
        path.write_text("3 0.05 0.5 0.100002 0.2\n", encoding="utf-8")
        self.assertEqual(read_yolo(path, 100, 100)[1], [(1, "rounding_at_border")])

    def test_empty_yolo_file_is_an_empty_annotation(self):
        path = self.root / "label.txt"
        path.write_text("\n", encoding="utf-8")
        self.assertEqual(read_yolo(path, 100, 100), ([], []))

    def test_metadata_bom_and_patient_study_key(self):
        path = self.root / "dataset.csv"
        rows = [synthetic_row(), dict(synthetic_row(), filestem="0001_1234567891_01_WRI-L2_F010",
                                     timehash="1234567891", projection="2")]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        records, summary, problems = read_metadata(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["patients"], 1)
        self.assertEqual(summary["patient_study_groups"], 1)
        self.assertEqual(summary["patient_timehash_groups"], 2)
        self.assertEqual(problems, [])

    def test_filename_disagreements(self):
        self.assertEqual(filename_disagreements(synthetic_row()), [])
        self.assertEqual(filename_disagreements(dict(synthetic_row(), patient_id="2")), ["patient_id"])
        self.assertEqual(filename_disagreements(dict(synthetic_row(), patient_id="")), ["patient_id"])
        self.assertEqual(filename_disagreements(dict(synthetic_row(), filestem="bad")), ["filename_pattern"])

    def test_fractional_ages_and_other_gender_are_valid(self):
        self.assertEqual(filename_disagreements(dict(synthetic_row(), age="10.9")), [])
        self.assertEqual(filename_disagreements(dict(synthetic_row(), gender="O",
                         filestem="0001_1234567890_01_WRI-L1_O010", age="10.5")), [])

    def test_duplicate_metadata_is_reported(self):
        path = self.root / "dataset.csv"
        row = synthetic_row()
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerows([row, row])
        _, summary, problems = read_metadata(path)
        self.assertEqual(summary["exact_duplicate_rows"], 1)
        self.assertIn((row["filestem"], "duplicate_filestem"), problems)

    def test_sixteen_bit_image_is_decoded_without_conversion(self):
        path = self.root / "image.png"
        pixels = np.array([[0, 65535], [256, 4095]], dtype=np.uint16)
        Image.fromarray(pixels).save(path)
        before = path.read_bytes()
        result = inspect_image(path, decode=True)
        self.assertEqual(result["bit_depth"], 16)
        self.assertEqual(result["color_type"], 0)
        self.assertEqual(result["pixels"]["dtype"], "uint16")
        self.assertEqual(result["pixels"]["max"], 65535)
        self.assertEqual(path.read_bytes(), before)

    def test_corrupt_png_is_reported(self):
        path = self.root / "bad.png"
        path.write_bytes(b"not a png")
        self.assertIn("error", inspect_image(path))

    def test_output_cannot_overlap_raw(self):
        raw = self.root / "raw"
        for output in (raw, raw / "child", self.root):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_output_path(raw, output)

    def test_output_cannot_write_to_sibling_raw_source(self):
        with self.assertRaises(ValueError):
            validate_output_path(self.root / "data/raw/dataset", self.root / "data/raw/other")


class IntegrationTests(TemporaryFiles):
    def test_minimal_audit_retains_empty_boxes_and_keeps_identities_local(self):
        raw, output = self.root / "raw", self.root / "outputs"
        for folder in ("images", "supervisely/wrist/ann", "pascalvoc", "yolov5/labels"):
            (raw / folder).mkdir(parents=True)
        row = synthetic_row()
        stem = row["filestem"]
        with (raw / "dataset.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        (raw / "supervisely/meta.json").write_text(json.dumps({"classes": [
            {"title": n, "shape": "line" if n == "axis" else "rectangle"}
            for n in (*SOURCE_CLASSES, "axis")]}))
        (raw / "yolov5/meta.yaml").write_text("names:\n" + "\n".join("- " + n for n in SOURCE_CLASSES))
        Image.fromarray(np.array([[0, 100], [200, 300]], dtype=np.uint16)).save(raw / "images" / (stem + ".png"))
        (raw / "supervisely/wrist/ann" / (stem + ".json")).write_text(json.dumps({
            "size": {"width": 2, "height": 2}, "objects": [{"classTitle": "axis", "points": {"exterior": [[0, 0], [1, 1]]}}], "tags": []}))
        (raw / "pascalvoc" / (stem + ".xml")).write_text("<annotation><size><width>2</width><height>2</height></size></annotation>")
        (raw / "yolov5/labels" / (stem + ".txt")).write_text("")
        summary = run_audit(raw, output, decode_sample=1, workers=1)
        self.assertEqual(summary["annotations"]["images_without_detection_boxes"], 1)
        self.assertEqual(summary["annotations"]["axis_lines"], 1)
        self.assertEqual(summary["images"]["pixel_decoded"], 1)
        self.assertTrue(summary["inventory"]["raw_stat_snapshot_unchanged"])
        self.assertEqual(summary["issue_counts"], {})
        self.assertNotIn(stem, json.dumps(summary))
        # A repeat must produce exactly the same aggregate evidence and source digest.
        self.assertEqual(summary, run_audit(raw, output, decode_sample=1, workers=1))
        summary["annotations"]["object_tag_counts"] = {"unreviewed free text": 1}
        (output / "summary.json").write_text(json.dumps(summary))
        (output / "source_release.json").write_text(json.dumps({"version": 2}))
        public = self.root / "public"
        export_evidence(output, public)
        shared = (public / "aggregate_statistics.json").read_text()
        self.assertNotIn(stem, shared)
        self.assertNotIn("unreviewed free text", shared)
        self.assertEqual(json.loads(shared)["annotations"]["object_tags_total"], 1)


if __name__ == "__main__":
    unittest.main()
