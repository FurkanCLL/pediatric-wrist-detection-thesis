"""Training-boundary and one-to-one difficulty safeguards."""

import copy
import unittest
import numpy as np

from wrist_mining.common import training_path, stability_subset
from wrist_mining.difficulty import object_difficulty, characterize, average_ranks, rank_correlation


def row(split="train"):
    return {"split": split, "filestem": "synthetic", "patient_id": "fixture", "width": 100, "height": 200,
            "boxes": [{"class_id": 0, "xyxy": [10, 20, 30, 60]}]}


class MiningTests(unittest.TestCase):
    def test_nontraining_paths_rejected(self):
        for split in ("val", "test"):
            with self.assertRaises(ValueError):
                training_path(row(split))
            with self.assertRaises(ValueError):
                characterize(row(split), np.empty((0, 6)))

    def test_path_traversal_and_unknown_kind_rejected(self):
        r = row(); r["filestem"] = "../outside"
        with self.assertRaises(ValueError): training_path(r)
        with self.assertRaises(ValueError): training_path(row(), "other")

    def test_difficulty_endpoints(self):
        self.assertEqual(object_difficulty([-1, 0, 1], [-1, -1, 1]).tolist(), [1, .5, 0])
        self.assertEqual(object_difficulty([], []).tolist(), [])
        with self.assertRaises(ValueError): object_difficulty([1], [])

    def test_exact_correct_annotation(self):
        d = characterize(row(), np.array([[10, 20, 30, 60, .8, 0]], np.float32))
        self.assertEqual(d["class_difficulty"][0], 0)
        self.assertEqual(d["operating"]["0.5"]["gt_reasons"], ["TP"])

    def test_weak_prediction_is_a_hard_positive(self):
        d = characterize(row(), np.array([[10, 20, 30, 60, .2, 0]], np.float32))
        self.assertEqual(d["class_difficulty"][0], 1)
        self.assertEqual(d["operating"]["0.5"]["gt_reasons"], ["low_confidence"])

    def test_localization_only_half_difficulty(self):
        d = characterize(row(), np.array([[10, 20, 39, 60, .8, 0]], np.float32))
        self.assertEqual(d["class_difficulty"][0], .5)
        self.assertEqual(d["operating"]["0.75"]["gt_reasons"], ["localization"])

    def test_wrong_class_is_not_correct_detection(self):
        d = characterize(row(), np.array([[10, 20, 30, 60, .8, 3]], np.float32))
        self.assertEqual(d["class_difficulty"][0], 1)
        self.assertEqual(d["operating"]["0.5"]["gt_reasons"], ["classification"])

    def test_one_prediction_cannot_explain_two_annotations(self):
        r = row(); r["boxes"].append(copy.deepcopy(r["boxes"][0]))
        d = characterize(r, np.array([[10, 20, 30, 60, .8, 0]], np.float32))
        self.assertEqual(d["operating"]["0.5"]["gt_reasons"], ["TP", "no_residual_candidate"])
        self.assertEqual(d["class_difficulty"][0], .5)

    def test_unmatched_prediction_is_only_potential_negative(self):
        r = row(); r["boxes"] = []
        d = characterize(r, np.array([[10, 20, 30, 60, .8, 0]], np.float32))
        self.assertEqual(d["class_difficulty"], [0]*8)
        self.assertEqual(d["operating"]["0.5"]["potential_fp_reasons"], ["background_or_unannotated"])

    def test_ranks_account_for_ties_and_constants(self):
        self.assertEqual(average_ranks([1, 1, 3, 2]).tolist(), [.5, .5, 3, 2])
        self.assertAlmostEqual(rank_correlation([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(rank_correlation([1, 1], [1, 2]))

    def test_hash_subset_is_order_independent(self):
        rows = []
        for i in range(1100):
            r = row(); r["filestem"] = f"fixture_{i:04}"; rows.append(r)
        a = {r["filestem"] for r in stability_subset(rows)}
        b = {r["filestem"] for r in stability_subset(list(reversed(rows)))}
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1024)


if __name__ == "__main__":
    unittest.main()
