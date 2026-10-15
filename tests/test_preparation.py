"""Focused synthetic checks for split independence and deterministic preparation."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from wrist_prepare.common import CLASSES, digest, protected_write
from wrist_prepare.images import convert_one, convert_records, to_uint8
from wrist_prepare.labels import target_boxes, yolo_bytes
from wrist_prepare.records import training_statistics
from wrist_prepare.splits import allocate_patients
from wrist_prepare.validation import check_pair


def annotation(name="fracture", points=None):
    return {"size": {"width": 200, "height": 100}, "objects": [
        {"classTitle": name, "points": {"exterior": points or [[10, 20], [100, 80]]}}]}


def groups(count=100):
    # Invented patient names and views keep these tests independent of medical records.
    return [{"patient_id": f"synthetic-patient-{i}", "filestem": f"view-{i}-{view}",
             "width": 200, "height": 100,
             "boxes": [{"class_id": c, "xyxy": [10, 20, 100, 80]} for c in ([2, 5] if i < 24 else [0, 2])],
             "foreignbody_boxes": 0}
            for i in range(count) for view in range(1 + i % 3)]


class SplitTests(unittest.TestCase):
    def test_patient_capacities_and_repeated_views(self):
        records = groups()
        allocation = allocate_patients(records)
        self.assertEqual(Counter(allocation.values()), {"train": 70, "val": 15, "test": 15})
        self.assertEqual(len(allocation), 100)
        self.assertTrue(all(allocation[r["patient_id"]] in {"train", "val", "test"} for r in records))

    def test_same_seed_is_independent_of_record_order(self):
        records = groups()
        self.assertEqual(allocate_patients(records), allocate_patients(list(reversed(records))))

    def test_twenty_four_rare_patients_have_both_holdouts(self):
        allocation = allocate_patients(groups())
        actual = Counter(allocation[f"synthetic-patient-{i}"] for i in range(24))
        self.assertEqual(actual, {"train": 16, "val": 4, "test": 4})

    def test_missing_patient_is_rejected(self):
        records = groups()
        records[0]["patient_id"] = ""
        with self.assertRaises(ValueError):
            allocate_patients(records)

    def test_large_groups_do_not_bias_training_image_share(self):
        records = [{"patient_id": f"invented-{i}", "boxes": [{"class_id": 2}]}
                   for i in range(1000) for _ in range(1 + i % 30)]
        allocation = allocate_patients(records)
        counts = Counter(allocation[r["patient_id"]] for r in records)
        for name, target in (("train", .7), ("val", .15), ("test", .15)):
            self.assertLess(abs(counts[name] / len(records) - target), .03)

    def test_statistics_use_training_records_only(self):
        records = groups()
        for r in records:
            r["split"] = "test" if any(o["class_id"] == 5 for o in r["boxes"]) else "train"
        stats = training_statistics(records)
        self.assertEqual(stats["support"]["classes"]["bonelesion"]["objects"], 0)
        self.assertEqual(stats["geometry"]["bonelesion"], {"relative_area": {"n": 0}, "short_side_pixels": {"n": 0}})


class LabelTests(unittest.TestCase):
    def test_approved_ids_and_excluded_objects(self):
        objects = [annotation(c, [[10, 20], [50, 60], [100, 20]] if c in ("bonelesion", "periostealreaction") else None)["objects"][0] for c in CLASSES]
        objects.extend([{"classTitle": "foreignbody"}, {"classTitle": "axis"}])
        data = annotation()
        data["objects"] = objects
        self.assertEqual([o["class_id"] for o in target_boxes(data)], list(range(8)))
        self.assertEqual([o["source_object_index"] for o in target_boxes(data)], list(range(8)))

    def test_polygon_envelope_and_normalized_coordinates(self):
        data = annotation("bonelesion", [[3, 4], [13, 14], [20, 4]])
        boxes = target_boxes(data)
        self.assertEqual(boxes[0]["xyxy"], (3, 4, 20, 14))
        label = yolo_bytes({"width": 200, "height": 100, "boxes": boxes}).decode().split()
        self.assertEqual(label[0], "5")
        np.testing.assert_allclose(list(map(float, label[1:])), [.0575, .09, .085, .1], atol=1e-15)

    def test_invalid_geometry_is_rejected_without_clipping(self):
        for points in ([[-1, 0], [2, 3]], [[0, 0], [201, 100]], [[2, 2], [2, 3]], [[0, 0], [float("nan"), 3]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                target_boxes(annotation(points=points))
        with self.assertRaises(ValueError):
            target_boxes(annotation("bonelesion", [[1, 1], [2, 2], [3, 3]]))

    def test_empty_target_labels_are_zero_bytes(self):
        self.assertEqual(yolo_bytes({"width": 1, "height": 1, "boxes": []}), b"")


class PixelTests(unittest.TestCase):
    def test_entire_uint16_domain_matches_independent_reference(self):
        pixels = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        expected = np.rint(pixels.astype(np.float64) * 255 / 65535).astype(np.uint8)
        np.testing.assert_array_equal(to_uint8(pixels), expected)
        self.assertEqual(to_uint8(pixels).ravel()[[0, 128, 129, 65535]].tolist(), [0, 0, 1, 255])

    def test_unexpected_dtype_or_channels_are_rejected(self):
        for pixels in (np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.uint16)):
            with self.assertRaises(ValueError):
                to_uint8(pixels)


class OutputTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "outputs/preparation_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_source(self):
        raw = self.root / "raw"
        (raw / "images").mkdir(parents=True)
        (raw / "supervisely/wrist/ann").mkdir(parents=True)
        data = annotation()
        image = raw / "images/synthetic.png"
        Image.fromarray(np.arange(20000, dtype=np.uint16).reshape(100, 200)).save(image)
        source = raw / "supervisely/wrist/ann/synthetic.json"
        source.write_text(json.dumps(data))
        stat = image.stat()
        record = {"filestem": "synthetic", "patient_id": "invented", "split": "train", "width": 200, "height": 100,
                  "source_image_stat": [stat.st_size, stat.st_mtime_ns], "source_image_sha256": digest(image.read_bytes()),
                  "source_annotation_sha256": digest(source.read_bytes()), "boxes": target_boxes(data)}
        return raw, record

    def test_existing_output_is_preserved_on_conflict(self):
        path = self.root / "output.txt"
        protected_write(path, b"original")
        self.assertFalse(protected_write(path, b"original"))
        with self.assertRaises(FileExistsError):
            protected_write(path, b"different")
        self.assertEqual(path.read_bytes(), b"original")

    def test_conversion_resume_and_independent_validation(self):
        raw, record = self.make_source()
        prepared = self.root / "prepared"
        result = convert_records([record], raw, prepared, self.root / "completed.sqlite", {"policy": "synthetic"}, 1)
        self.assertEqual(result["newly_completed"], 1)
        self.assertEqual(check_pair(record, raw, prepared)["objects"], 1)
        result = convert_records([record], raw, prepared, self.root / "completed.sqlite", {"policy": "synthetic"}, 1)
        self.assertEqual(result["reused_completed_outputs"], 1)
        label = prepared / "labels/train/synthetic.txt"
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaises(ValueError):
            check_pair(record, raw, prepared)
        with self.assertRaises(ValueError):
            convert_records([record], raw, prepared, self.root / "completed.sqlite", {"policy": "synthetic"}, 1)

    def test_changed_raw_source_is_rejected(self):
        raw, record = self.make_source()
        record["source_image_sha256"] = "not the source hash"
        with self.assertRaises(ValueError):
            convert_one(record, raw, self.root / "prepared")

    def test_restart_after_image_publication_before_label_or_journal(self):
        raw, record = self.make_source()
        prepared = self.root / "prepared"
        convert_one(record, raw, prepared)
        image = prepared / "images/train/synthetic.png"
        original_output_hash = digest(image.read_bytes())
        (prepared / "labels/train/synthetic.txt").unlink()
        convert_records([record], raw, prepared, self.root / "new_journal.sqlite", {"policy": "synthetic"}, 1)
        self.assertEqual(digest(image.read_bytes()), original_output_hash)
        self.assertEqual(check_pair(record, raw, prepared)["objects"], 1)

    def test_incomplete_private_file_is_replaced_by_a_complete_publication(self):
        target = self.root / "output.txt"
        target.with_name("output.txt.partial").write_bytes(b"interrupted")
        protected_write(target, b"complete")
        self.assertEqual(target.read_bytes(), b"complete")
        self.assertFalse(target.with_name("output.txt.partial").exists())


if __name__ == "__main__":
    unittest.main()
