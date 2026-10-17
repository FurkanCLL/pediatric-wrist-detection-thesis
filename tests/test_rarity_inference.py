"""Additive E/F contracts, using synthetic predictions rather than GPU inference."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wrist_inference_comparison import common
from wrist_inference_comparison.checks import equal_global, valid_timing, verify_f
from wrist_inference_comparison.analysis import contrast, CONTRASTS
from wrist_eval.adaptive import plan_rois, fuse_predictions, shift_local
from wrist_eval.adaptive_study import private_record
from wrist_eval.bootstrap import patient_groups, resample_indices
from wrist_eval.sliced import tile_dimensions


class RarityInferenceTests(unittest.TestCase):
    def setUp(self):
        self.executions, self.cfg = common.manifests()

    def fixture(self, values=None):
        original = np.asarray(values if values is not None else
            [[20, 20, 40, 40, .9, 2], [250, 450, 350, 550, .2, 0]], np.float32).reshape(-1, 6)
        plan = plan_rois(original, (1000, 600), self.cfg)
        counter = {"forward_passes": 1}
        views = []
        for roi in plan["rois"]:
            x, y, right, bottom = roi["box"]
            local = np.asarray([[260-x, 450-y, 360-x, 550-y, .8, 0]], np.float32)
            views.append({"predictions": local, "roi_id": roi["id"], "gains": (1., 1.), "counters": counter})
        fused = fuse_predictions(original, views, plan, self.cfg, "V2")
        result = {"global": original, "global_view": {"predictions": original, "gains": (1., 1.), "counters": counter},
            "local_views": views, "plan": plan, "variants": {"V2": fused}, "passes": 1+len(views),
            "common_seconds": .1, "planning_seconds": .01, "fusion_seconds": {"V2": .01}}
        return original, private_record("synthetic", result, .01, .12, {})

    def test_execution_scope_and_historical_bindings(self):
        self.assertEqual(self.executions["E"]["fixed_calls_per_image"], 5)
        self.assertEqual(self.executions["F"]["variant"], "V2")
        self.assertEqual(self.executions["F"]["historical_source_sha256"], common.adaptive_sources())
        self.assertEqual(self.executions["F"]["checkpoint"]["epoch"], 26)
        self.assertFalse(self.cfg["execution_authorized"])

    def test_reject_manifest_identity_drift(self):
        actual_read = common.read
        alterations = [("E", "historical_config_sha256", "bad"), ("F", "historical_execution_sha256", "bad"),
            ("F", "variant", "V1"), ("E", "fixed_calls_per_image", 4), ("F", "training_authorized", True),
            ("F", "checkpoint", {**self.executions["F"]["checkpoint"], "epoch": 80}),
            ("F", "historical_source_sha256", {})]
        for condition, key, value in alterations:
            changed = copy.deepcopy(self.executions)
            changed[condition][key] = value
            def reader(path):
                for c, name in zip(("E", "F"), common.CONFIGS):
                    if Path(path) == common.ROOT/name:
                        return changed[c]
                return actual_read(path)
            with self.subTest(condition=condition, key=key), patch.object(common, "read", side_effect=reader):
                with self.assertRaises(ValueError):
                    common.manifests()

    def test_wrong_checkpoint_rejected_before_deserialization(self):
        from wrist_compare.common import settings
        with patch("wrist_compare.common.settings", return_value=settings()), \
             patch("wrist_compare.common.sha", return_value="wrong"), patch("torch.load") as loader:
            with self.assertRaises(ValueError):
                common.checked_checkpoint("D")
            loader.assert_not_called()

    def test_four_context_slices_and_five_calls(self):
        from sahi.slicing import get_slice_bboxes
        for h, w in ((1000, 600), (600, 1000), (1231, 999), (2048, 2048)):
            sh, sw = tile_dimensions(h, w, .667)
            self.assertEqual(len(get_slice_bboxes(h, w, slice_height=sh, slice_width=sw,
                overlap_height_ratio=.2, overlap_width_ratio=.2, auto_slice_resolution=False)), 4)
        value = {"forward_passes": 5, "e2e_seconds": .2, "gpu_forward_ms": 20., "decode_seconds": .01}
        valid_timing(value, 5)
        for key, bad in (("forward_passes", 4), ("gpu_forward_ms", 0), ("e2e_seconds", float("nan"))):
            with self.assertRaises(ValueError):
                valid_timing({**value, key: bad}, 5)

    def test_exact_global_disallows_even_one_ulp_or_signed_zero(self):
        value = np.asarray([[0, 0, 10, 10, .2, 0]], np.float32)
        equal_global(value.tolist(), value)
        for column in (0, 4):
            changed = value.copy()
            changed[0, column] = -0. if column == 0 else np.nextafter(value[0, 4], np.float32(1.))
            with self.assertRaises(ValueError):
                equal_global(changed, value)

    def test_frozen_f_replay_and_original_coordinate_geometry(self):
        original, item = self.fixture()
        counts = verify_f(item, original, self.cfg)
        self.assertEqual((counts["rois"], counts["calls"], counts["confidence_updates"], counts["geometry_changes"]), (1, 2, 1, 1))
        np.testing.assert_array_equal(item["variants"]["V2"]["pred"][1][:4], [252.5, 450, 352.5, 550])
        shifted = shift_local(item["local_views"][0]["predictions"], item["plan"]["rois"][0], (1000, 600))
        np.testing.assert_array_equal(shifted[0, :4], [260, 450, 360, 550])

    def test_zero_roi_still_checks_saved_D(self):
        original, item = self.fixture([[20, 20, 40, 40, .9, 2]])
        self.assertEqual(verify_f(item, original, self.cfg)["calls"], 1)
        with self.assertRaises(ValueError):
            verify_f(item, np.empty((0, 6), np.float32), self.cfg)

    def test_reject_protected_class_score_geometry_and_count_corruption(self):
        original, item = self.fixture()
        for column, bad in ((0, 21), (4, .8), (5, 1)):
            changed = copy.deepcopy(item)
            changed["variants"]["V2"]["pred"][0][column] = bad
            with self.subTest(column=column), self.assertRaises(ValueError):
                verify_f(changed, original, self.cfg)
        for column, bad in ((0, 251), (4, .36)):
            changed = copy.deepcopy(item)
            changed["variants"]["V2"]["pred"][1][column] = bad
            with self.assertRaises(ValueError):
                verify_f(changed, original, self.cfg)
        changed = copy.deepcopy(item)
        changed["local_views"][0]["counters"]["forward_passes"] = 3
        with self.assertRaises(ValueError):
            verify_f(changed, original, self.cfg)
        changed = copy.deepcopy(item)
        changed["variants"]["V2"]["pred"].pop()
        with self.assertRaises(ValueError):
            verify_f(changed, original, self.cfg)

    def test_interactions_are_paired_and_absent_metrics_undefined(self):
        values = {c: {"overall": {"map50_95": value}, "fracture": {"ap50_95": value/2}}
            for c, value in zip("ABCDEF", (1., 2., 3., 4., 8., 10.))}
        self.assertEqual(contrast(values, CONTRASTS["(E-D)-(B-A)"])["overall"]["map50_95"], 3.)
        self.assertEqual(contrast(values, CONTRASTS["(F-D)-(C-A)"])["overall"]["map50_95"], 4.)
        values["E"].pop("overall")
        self.assertNotIn("overall", contrast(values, CONTRASTS["E-D"]))
        self.assertNotIn("bonelesion", contrast(values, CONTRASTS["E-D"]))
        groups = patient_groups([{"patient_id": "one"}, {"patient_id": "two"}, {"patient_id": "one"}])
        a, b = np.random.default_rng(314159), np.random.default_rng(314159)
        for _ in range(10):
            indices = resample_indices(groups, a)
            self.assertEqual(indices, resample_indices(groups, b))
            self.assertEqual(indices.count(0), indices.count(2))

    def test_prediction_membership_order_and_hash(self):
        from wrist_eval.common import sha
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic.jsonl"
            row = {"filestem": "synthetic", "width": 10, "height": 10, "patient_id": "fixture"}
            path.write_text('{"stem":"synthetic","pred":[[0,0,10,10,0.9,0]]}\n')
            common.load_predictions(path, [row], sha(path))
            for rows, digest in (([row], "wrong"), ([{**row, "filestem": "other"}], sha(path)), ([row, row], sha(path))):
                with self.assertRaises(ValueError):
                    common.load_predictions(path, rows, digest)

    def test_existing_attempt_does_not_write_failure_or_infer(self):
        from wrist_inference_comparison.run import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"pilot_F").mkdir()
            common.write_new(path/"pilot_F/summary.json", {"original": True})
            with patch("sys.argv", ["run", "pilot", "F"]), \
                 patch("wrist_inference_comparison.run.study_path", return_value=path), \
                 patch("wrist_inference_comparison.run.execute") as infer:
                with self.assertRaises(FileExistsError):
                    main()
                infer.assert_not_called()
            self.assertEqual(common.read(path/"pilot_F/summary.json"), {"original": True})
            self.assertFalse((path/"pilot_F/failure.json").exists())

    def test_failed_fresh_attempt_retains_partial_evidence(self):
        from wrist_inference_comparison.run import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            def fail(*args):
                (path/"pilot_F").mkdir()
                (path/"pilot_F/partial.txt").write_text("preserved")
                raise ValueError("synthetic integrity gate")
            with patch("sys.argv", ["run", "pilot", "F"]), \
                 patch("wrist_inference_comparison.run.study_path", return_value=path), \
                 patch("wrist_inference_comparison.run.install_access_guard"), \
                 patch("wrist_inference_comparison.run.execute", side_effect=fail):
                with self.assertRaises(ValueError):
                    main()
            self.assertTrue((path/"pilot_F/failure.json").exists())
            self.assertEqual((path/"pilot_F/partial.txt").read_text(), "preserved")

    def test_heldout_and_path_escape_rejected(self):
        from wrist_eval.common import image_path
        for split in ("train", "test"):
            with self.assertRaises(ValueError):
                image_path({"split": split, "filestem": "synthetic"})
        for name in ("../bad", "bad/path", "", ".."):
            with self.assertRaises(ValueError):
                common.study_path(name)

    def test_export_rejects_unverified_results_before_publication(self):
        from wrist_inference_comparison.export import export
        with patch("wrist_inference_comparison.export.frozen_study", return_value={}), \
             patch("wrist_inference_comparison.export.read", return_value={"passed": False}), \
             patch("wrist_inference_comparison.export.write_new") as writer:
            with self.assertRaises(ValueError):
                export(Path("synthetic"))
            writer.assert_not_called()

    def test_original_scientific_gates_do_not_reward_unchanged_AP(self):
        from wrist_inference_comparison.export import scientific_gate
        from wrist_eval.common import CLASSES
        point = {"total_fp": 100, "classes": {"fracture": {"fp": 20}}}
        delta = {"overall": {"map50": 0., "map75": 0., "map50_95": 0.},
            **{c: {"ap50": 0., "ap75": 0., "ap50_95": 0.} for c in CLASSES}}
        ci = {c: {metric: {"lower": 0., "upper": 0.} for metric in value} for c, value in delta.items()}
        result = {"contrasts": {"F-D": delta}, "metrics": {"D": {}, "F": {}},
                  "operating_points": {c: {"conf0.25_iou0.5": point} for c in ("D", "F")}}
        gate = scientific_gate(result, {"contrasts": {"F-D": ci}}, {"timing": {
            "e2e_seconds": {"mean": .08}, "paired_global_e2e_seconds": {"mean": .06}}}, self.cfg)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["primary_success"])
        self.assertFalse(gate["secondary_tradeoff_success"])


if __name__ == "__main__":
    unittest.main()
