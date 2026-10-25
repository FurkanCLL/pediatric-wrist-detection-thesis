"""Synthetic checks for immutable identities and the two evaluation tracks."""

import unittest
import numpy as np

from wrist_v3.metrics import image_stats, deterministic_ap, historical_ap


class V3MetricTests(unittest.TestCase):
    def test_no_ties_reproduces_pinned_AP_for_all_eight_classes(self):
        gt = np.array([[i*30, 0, i*30+10, 10, i] for i in range(8)], np.float32)
        pred = np.array([[*box[:4], .9-i*.01, i] for i, box in enumerate(gt)], np.float32)
        stat = image_stats(gt, pred, list(range(8)), 0, canonical=True)
        self.assertEqual(deterministic_ap([stat]), historical_ap([stat]))

    def test_equal_scores_do_not_depend_on_final_row_order(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        pred = np.array([[0, 0, 10, 10, .3, 0], [20, 20, 30, 30, .3, 0]], np.float32)
        a = image_stats(gt, pred, [0, 1], 0, canonical=True)
        b = image_stats(gt, pred[::-1], [1, 0], 0, canonical=True)
        self.assertEqual(deterministic_ap([a]), deterministic_ap([b]))
        self.assertEqual(a['conf'].tolist(), b['conf'].tolist())

    def test_present_class_without_predictions_has_zero_AP(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        value = deterministic_ap([image_stats(gt, np.empty((0, 6), np.float32), [], 0, True)])
        self.assertEqual(value['fracture']['ap50_95'], 0.)
        self.assertNotIn('bonelesion', value)
        self.assertNotIn('overall', value)

    def test_empty_gt_has_no_invented_class_or_overall_AP(self):
        value = deterministic_ap([image_stats(np.empty((0, 5), np.float32), np.empty((0, 6), np.float32), [], 0, True)])
        self.assertEqual(value, {})

    def test_repeated_patient_view_occurrences_do_not_use_mutable_sample_order(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        tp = np.array([[0, 0, 10, 10, .3, 0]], np.float32)
        fp = np.array([[20, 20, 30, 30, .3, 0]], np.float32)
        a, b = image_stats(gt, tp, [0], 0, True), image_stats(gt, fp, [0], 1, True)
        self.assertEqual(deterministic_ap([a, b, a]), deterministic_ap([b, a, a]))

    def test_original_ids_must_be_complete_unique_and_in_range(self):
        gt = np.empty((0, 5), np.float32); pred = np.zeros((2, 6), np.float32)
        for ids in ([0, 0], [0, 2], [0]):
            with self.assertRaises(ValueError):
                image_stats(gt, pred, ids, 0)


if __name__ == '__main__':
    unittest.main()
