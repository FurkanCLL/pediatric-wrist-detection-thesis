"""Synthetic contracts for A/D reuse, validation geometry and paired uncertainty."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wrist_compare.common import checked_checkpoint, load_predictions, study_path, read, write_new, require
from wrist_compare.infer import assert_metrics, verify_item
from wrist_compare.analysis import compact_ap, paired_values, count_metrics, metric_delta
from wrist_eval.common import CLASSES, image_path, sha
from wrist_eval.metrics import ap_summary, native_stats
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_eval.sliced import resize_and_pad


class RarityComparisonTests(unittest.TestCase):
    def test_reject_altered_checkpoint_before_pickle_load(self):
        cfg = {"baseline_checkpoint_sha256": "expected"}
        with patch("wrist_compare.common.settings", return_value=cfg), patch("wrist_compare.common.sha", return_value="altered"), \
             patch("torch.load") as loader:
            with self.assertRaises(ValueError):
                checked_checkpoint("A")
            loader.assert_not_called()

    def test_only_A_and_D_conditions(self):
        with patch("wrist_compare.common.settings", return_value={}):
            for condition in ("E", "F", "test", "unknown"):
                with self.subTest(condition=condition), self.assertRaises(ValueError):
                    checked_checkpoint(condition)

    def test_strict_prediction_hash_membership_and_linkage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"predictions.jsonl"
            row = {"filestem": "generated", "width": 10, "height": 20, "patient_id": "fixture"}
            path.write_text('{"stem":"generated","patient_id":"fixture","pred":[[0,0,10,20,0.9,0]]}\n')
            result, _ = load_predictions(path, [row], sha(path))
            self.assertEqual(result["generated"].shape, (1, 6))
            with self.assertRaises(ValueError):
                load_predictions(path, [row], "wrong")
            for bad in ({**row, "filestem": "missing"}, {**row, "patient_id": "different"}, {**row, "width": 9}):
                with self.assertRaises(ValueError):
                    load_predictions(path, [bad], sha(path))
            path.write_text(path.read_text()*2)
            with self.assertRaises(ValueError):
                load_predictions(path, [row], sha(path))

    def test_traversal_and_heldout_image_rejected(self):
        for name in ("../bad", "bad/path", "", ".."):
            with self.assertRaises(ValueError):
                study_path(name)
        for split in ("train", "test"):
            with self.assertRaises(ValueError):
                image_path({"filestem": "fixture", "split": split})

    def test_no_tiled_or_extra_pass_accounting(self):
        row = {"width": 10, "height": 20}
        pred = np.array([[0, 0, 10, 20, .9, 0]], np.float32)
        timing = {"forward_passes": 1, "slices": 0, "post_merge_predictions": 1, "seconds": .02,
            "e2e_seconds": .03, "decode_seconds": .01, "gpu_forward_ms": 2., "preprocess_seconds": .001,
            "model_wall_seconds": .003, "nms_restore_seconds": .001}
        verify_item(row, pred, timing)
        for key, value in (("slices", 1), ("forward_passes", 2), ("gpu_forward_ms", float("nan")),
                           ("e2e_seconds", 0), ("post_merge_predictions", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_item(row, pred, {**timing, key: value})

    def test_original_coordinate_roundtrip_using_preserved_geometry(self):
        import torch
        from ultralytics.utils import ops
        image = np.zeros((1024, 576, 3), np.uint8)
        padded, ratio = resize_and_pad(image, (800, 448))
        gt = torch.tensor([[0., 0., 576., 1024.], [100., 200., 300., 500.]])
        net = gt*.75+torch.tensor([*ratio[1], *ratio[1]])
        restored = ops.scale_boxes(padded.shape[:2], net, image.shape[:2], ratio)
        self.assertTrue(torch.equal(restored, gt))

    def test_common_AP_and_difference_sign(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        a = ap_summary([native_stats(gt, np.array([[0, 0, 10, 10, .9, 0]], np.float32))])
        d = ap_summary([native_stats(gt, np.array([[20, 20, 30, 30, .9, 0]], np.float32))])
        self.assertAlmostEqual(a["map50_95"], .995)
        self.assertAlmostEqual(metric_delta(a, d)["map50_95"], -.995)
        assert_metrics(a, a)
        with self.assertRaises(ValueError):
            assert_metrics(a, d)

    def test_absent_class_is_undefined_and_overall_requires_eight(self):
        value = {"classes": {CLASSES[0]: {"ap50": 0., "ap75": 0., "ap50_95": 0.}},
                 "map50": 0., "map75": 0., "map50_95": 0.}
        compact = compact_ap(value)
        self.assertNotIn("overall", compact)
        self.assertNotIn("bonelesion", compact)
        self.assertEqual(compact["fracture"]["ap50"], 0.)
        self.assertEqual(paired_values(compact, {}), {})

    def test_patient_multiplicity_and_paired_draws(self):
        rows = [{"patient_id": "p1"}, {"patient_id": "p2"}, {"patient_id": "p1"}]
        groups = patient_groups(rows)
        rng = np.random.default_rng(314159)
        for _ in range(20):
            indices = resample_indices(groups, rng)
            self.assertEqual(indices.count(0), indices.count(2))
            counts = np.array([[1, 1, 0], [3, 2, 1], [2, 1, 2]])
            # The same index vector is applied to A and D; an identical detector has zero paired change.
            a = count_metrics(np.tile(counts[indices].sum(axis=0), (8, 1)))
            self.assertTrue(all(v == 0 for m in paired_values(a, a).values() for v in m.values()))

    def test_undefined_operating_recall_is_not_zero(self):
        empty = count_metrics(np.zeros((8, 3), dtype=int))
        self.assertNotIn("recall", empty["bonelesion"])
        self.assertNotIn("precision", empty["bonelesion"])
        self.assertEqual(interval([]), {"replicates_present": 0})

    def test_existing_attempt_is_untouched(self):
        from wrist_compare.infer import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_new(path/"summary.json", {"original": True})
            with patch("sys.argv", ["infer", "pilot"]), patch("wrist_compare.infer.study_path", return_value=path.parent), \
                 patch("wrist_compare.infer.install_access_guard"), patch("wrist_compare.infer.run") as runner:
                # Name the existing directory exactly as the stage destination.
                stage = path.parent/"pilot"
                with patch("pathlib.Path.exists", return_value=True):
                    with self.assertRaises(FileExistsError):
                        main()
                runner.assert_not_called()
            self.assertEqual(read(path/"summary.json"), {"original": True})
            self.assertFalse((path/"failure.json").exists())


if __name__ == "__main__":
    unittest.main()
