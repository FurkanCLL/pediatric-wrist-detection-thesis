"""Synthetic geometry and sampling checks without medical images or model fitting."""

from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from wrist_design.common import ROOT
from wrist_design.crops import CropRejected, crop_pixels, propose_crop, transform_boxes, visible_fraction
from wrist_design.sampling import DesignPopulation
from wrist_design.simulate import run_policy


def configuration():
    return json.loads((ROOT / "configs/class_aware_candidates.json").read_text())


def row(boxes=None, width=400, height=800):
    return {"split": "train", "filestem": "synthetic", "patient_id": 0,
            "width": width, "height": height, "boxes": boxes or [
                {"class_id": 0, "xyxy": [180, 300, 225, 380]},
                {"class_id": 3, "xyxy": [190, 320, 210, 360]},
                {"class_id": 2, "xyxy": [0, 0, 50, 20]}]}


def population_rows():
    result = []
    for i in range(400):
        labels = [2] + ([0] if i % 2 == 0 else [])
        labels += [c for c, divisor in ((1, 19), (3, 10), (4, 11), (6, 13), (7, 17), (5, 101)) if i % divisor == 0]
        result.append({**row(), "filestem": f"synthetic_{i:04}", "patient_id": i//4,
                       "boxes": [{"class_id": c, "xyxy": [180, 300, 200, 330]} for c in labels]})
    return result


class CropGeometry(unittest.TestCase):
    def setUp(self):
        self.crop = configuration()["candidates"]["B"]["crop"]

    def test_coordinates_visibility_and_all_retained_classes(self):
        source = row([{"class_id": 3, "xyxy": [90, 250, 190, 350]},
                      {"class_id": 0, "xyxy": [180, 300, 225, 380]},
                      {"class_id": 2, "xyxy": [0, 0, 50, 20]}])
        rect = [100, 200, 400, 800]
        self.assertAlmostEqual(visible_fraction(source["boxes"][0]["xyxy"], rect), .9)
        result = transform_boxes(source, rect, self.crop)
        self.assertEqual(result["boxes"], [{"class_id": 3, "xyxy": [0, 50, 90, 150]},
                                           {"class_id": 0, "xyxy": [80, 100, 125, 180]}])
        self.assertEqual(result["source_indices"], [0, 1])
        self.assertEqual(result["omitted_objects"][2], 1)
        self.assertEqual(result["clipped_annotations"], 1)

    def test_partial_visible_objects_trigger_rejection_not_label_deletion(self):
        source = row([{"class_id": 2, "xyxy": [0, 250, 110, 350]}])
        with self.assertRaisesRegex(CropRejected, "partially_visible"):
            transform_boxes(source, [100, 200, 400, 800], self.crop)

    def test_fracture_and_target_require_full_visibility(self):
        for class_id, target in ((0, None), (3, 0)):
            source = row([{"class_id": class_id, "xyxy": [90, 250, 190, 350]}])
            with self.assertRaisesRegex(CropRejected, "fully_retained"):
                transform_boxes(source, [100, 200, 400, 800], self.crop, target)

    def test_degenerate_nonfinite_and_outside_annotations_are_rejected(self):
        for box in ([10, 10, 10, 20], [0, 0, float("nan"), 20], [-1, 0, 10, 20]):
            with self.assertRaises(ValueError):
                propose_crop(row([{"class_id": 3, "xyxy": box}]), self.crop, [3], [1.]*8)

    def test_border_crop_and_aspect_ratio_are_safe(self):
        source = row([{"class_id": 3, "xyxy": [0, 0, 10, 20]}])
        plan = propose_crop(source, self.crop, [3], [1.]*8)
        self.assertTrue(plan["accepted"])
        self.assertEqual(plan["rect"][:2], [0, 0])
        left, top, right, bottom = plan["rect"]
        self.assertAlmostEqual((right-left)/(bottom-top), .5, delta=.002)
        self.assertEqual(plan["visible_fractions"], [1.])

    def test_wide_images_keep_aspect_and_no_fabricated_padding(self):
        source = row([{"class_id": 3, "xyxy": [350, 180, 370, 200]}], width=800, height=400)
        plan = propose_crop(source, self.crop, [3], [1.]*8)
        self.assertTrue(plan["accepted"])
        left, top, right, bottom = plan["rect"]
        self.assertAlmostEqual((right-left)/(bottom-top), 2., delta=.005)

    def test_minimum_retained_size_and_invalid_pixel_rectangles_are_rejected(self):
        source = row([{"class_id": 3, "xyxy": [180, 300, 180.5, 330]}])
        with self.assertRaisesRegex(CropRejected, "too_small"):
            transform_boxes(source, [0, 0, 400, 800], self.crop)
        with self.assertRaises(ValueError):
            crop_pixels(np.zeros((8, 4)), {"rect": [-1, 0, 4, 8]})

    def test_empty_labels_fall_back_without_creating_annotations(self):
        source = row()
        source["boxes"] = []
        plan = propose_crop(source, self.crop, [3], [1.]*8)
        self.assertFalse(plan["accepted"])
        self.assertEqual(plan["boxes"], [])

    def test_fracture_outside_causes_full_image_fallback(self):
        source = row([{"class_id": 3, "xyxy": [0, 0, 10, 20]},
                      {"class_id": 0, "xyxy": [350, 700, 390, 790]}])
        plan = propose_crop(source, self.crop, [3], [1.]*8)
        self.assertFalse(plan["accepted"])
        self.assertEqual(plan["reason"], "fracture_or_target_not_fully_retained")
        self.assertEqual(plan["rect"], [0, 0, 400, 800])
        self.assertEqual(plan["boxes"], source["boxes"])

    def test_bonelesion_is_coverage_only_and_holdouts_are_blocked(self):
        source = row([{"class_id": 5, "xyxy": [180, 300, 200, 330]}])
        self.assertEqual(propose_crop(source, self.crop, [5], [1.]*8)["reason"], "bonelesion_coverage_protected")
        source["split"] = "val"
        with self.assertRaises(ValueError):
            propose_crop(source, self.crop, [5], [1.]*8)

    def test_deterministic_proposal_and_independent_pixels_do_not_mutate_sources(self):
        source = row()
        saved = deepcopy(source)
        image = np.arange(400*800, dtype=np.uint8).reshape(800, 400)
        original = image.copy()
        first = propose_crop(source, self.crop, [3], [1.]*8)
        self.assertEqual(first, propose_crop(source, self.crop, [3], [1.]*8))
        self.assertTrue(first["accepted"])
        pixels = crop_pixels(image, first)
        pixels[:] = 0
        np.testing.assert_array_equal(image, original)
        self.assertEqual(source, saved)


class SamplingDesign(unittest.TestCase):
    def test_frozen_baseline_lock_stays_valid(self):
        from wrist_train.protocol import load_protocol
        approved, identity = load_protocol()
        self.assertEqual(approved["training"]["epochs"], 80)
        self.assertEqual(identity["training_epochs"], 80)

    def test_prototype_attempt_cannot_escape_local_outputs(self):
        from wrist_design.prototype import main
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--attempt", "../outside"])

    def test_order_independence_exact_draws_and_seed_reproducibility(self):
        config = configuration()
        rows = population_rows()
        first, second = DesignPopulation(rows, config), DesignPopulation(rows[::-1], config)
        policy = config["candidates"]["B"]
        draw, bonus, _ = first.sample(policy, 42, 1)
        other, other_bonus, _ = second.sample(policy, 42, 1)
        np.testing.assert_array_equal(draw, other)
        np.testing.assert_array_equal(bonus, other_bonus)
        self.assertEqual(len(draw), len(rows))
        self.assertFalse(np.array_equal(draw, first.sample(policy, 1042, 1)[0]))

    def test_caps_status_patients_and_lesion_preservation_across_policies(self):
        config = configuration()
        pop = DesignPopulation(population_rows(), config)
        for policy in config["candidates"].values():
            rows, cumulative, draws, crops = run_policy(pop, policy, 42, 3)
            # Short simulations need not cover every common image; report the measured union honestly.
            self.assertEqual(cumulative["distinct_images_over_budget"], len(np.unique(draws)))
            self.assertEqual(cumulative["distinct_patients_over_budget"], len(pop.patient_size))
            for record in rows:
                self.assertEqual(record["draws"], len(pop.rows))
                self.assertEqual(record["unique_patients"], len(pop.patient_size))
                self.assertEqual(record["fracture_positive_draws"], int(pop.present[:, 0].sum()))
                self.assertEqual(record["class_objects_before"][0], record["class_objects_after"][0])
                self.assertEqual(record["class_objects_after"][5], int(pop.counts[:, 5].sum()))
                self.assertLessEqual(record["image_repetition"]["max"], policy["maximum_image_draws"])
                self.assertLessEqual(record["patient_relative_repetition"]["max"], 2.)

    def test_standard_visits_every_image_once_and_rejects_holdout_or_duplicate_rows(self):
        config = configuration()
        rows = population_rows()
        pop = DesignPopulation(rows, config)
        draws, flags, _ = pop.sample(None, 42, 0)
        np.testing.assert_array_equal(np.sort(draws), np.arange(len(rows)))
        self.assertFalse(flags.any())
        with self.assertRaises(ValueError):
            DesignPopulation([*rows, rows[0]], config)
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError):
            DesignPopulation(rows, config)

    def test_max_weight_saturation_does_not_sum_cooccurring_classes(self):
        config = configuration()
        pop = DesignPopulation(population_rows(), config)
        for policy in config["candidates"].values():
            classes, images = pop.weights(policy)
            self.assertLessEqual(images.max(), policy["weight_cap"])
            self.assertTrue(np.all(images[np.isin(pop.patient, pop.lesion_patients)] == 0))
            eligible = ~np.isin(pop.patient, pop.lesion_patients)
            np.testing.assert_array_equal(images[eligible], (pop.present[eligible]*classes).max(axis=1))


if __name__ == "__main__":
    unittest.main()
