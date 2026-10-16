"""Synthetic CG-ALR safeguards; none of these tests runs YOLO inference."""

import copy
import unittest
from unittest.mock import patch
import numpy as np

from wrist_eval.adaptive import (boundary_sides, candidate_score, checked_rows, confidence_update,
    contains, crop_geometry, fuse_predictions, local_evidence, overlap, plan_rois, predict_adaptive, shift_local)
from wrist_eval.adaptive_config import approved_settings


def rows(values=()):
    return np.asarray(values, dtype=np.float32).reshape(-1, 6)


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.manifest = approved_settings()
        self.hw = (1000, 600)
        self.global_rows = rows([[250, 450, 350, 550, .2, 0]])
        self.plan = plan_rois(self.global_rows, self.hw, self.cfg)

    def view(self, original_rows, plan=None, roi_id=0):
        plan = plan or self.plan
        roi = plan["rois"][roi_id]
        x1, y1, x2, y2 = roi["box"]
        values = rows(original_rows)
        values[:, :4] -= np.array([x1, y1, x1, y1], dtype=np.float32)
        cw, ch = x2-x1, y2-y1
        r = 768/max(cw, ch)
        return {"roi_id": roi_id, "predictions": values,
                "gains": (min(768, np.ceil(ch*r))/ch, min(768, np.ceil(cw*r))/cw)}

    def fused(self, local=None, variant="V1", global_rows=None, plan=None):
        source = self.global_rows if global_rows is None else global_rows
        plan = self.plan if plan is None else plan
        local = [[250, 450, 350, 550, .8, 0]] if local is None else local
        views = [self.view(local, plan)] if plan["rois"] else []
        return fuse_predictions(source, views, plan, self.cfg, variant)

    def test_approved_identity_preserves_proposal_status(self):
        self.assertTrue(self.manifest["execution_authorized"])
        self.assertFalse(self.cfg["execution_authorized"])
        self.assertEqual(self.cfg["status"], "PROPOSED_UNAPPROVED")

    def test_score_equations_and_monotonicity(self):
        self.assertAlmostEqual(candidate_score(.20, 32, 0, self.cfg)[2], .65)
        self.assertAlmostEqual(candidate_score(.20, 16, 3, self.cfg)[2], .65)
        self.assertGreater(candidate_score(.20, 64, 0, self.cfg)[2], candidate_score(.05, 1, 0, self.cfg)[2])
        self.assertGreater(candidate_score(.24, 20, 0, self.cfg)[2], candidate_score(.06, 20, 0, self.cfg)[2])
        self.assertEqual(candidate_score(.2, 128, 0, self.cfg)[1], 0)

    def test_confidence_band_boundaries(self):
        conf = [np.nextafter(np.float32(.05), np.float32(-np.inf)), .05,
                np.nextafter(np.float32(.25), np.float32(-np.inf)), .25, .5]
        source = rows([[10, 10, 50, 50, p, 0] for p in conf])
        plan = plan_rois(source, self.hw, self.cfg)
        self.assertEqual([c["global_id"] for c in plan["candidates"]], [1, 2])

    def test_target_classes_only(self):
        source = rows([[10, 10, 50, 50, .2, c] for c in range(8)])
        self.assertEqual([c["class_id"] for c in plan_rois(source, self.hw, self.cfg)["candidates"]], [0, 3])

    def test_one_effective_pixel_minimum(self):
        source = rows([[10, 10, 11, 20, .2, 0], [20, 20, 20.9, 30, .2, 0]])
        self.assertEqual(len(plan_rois(source, (768, 768), self.cfg)["candidates"]), 1)

    def test_large_size_reference_is_not_exclusion(self):
        self.assertGreater(self.plan["candidates"][0]["short_768"], 64)
        self.assertEqual(len(self.plan["rois"]), 1)

    def test_stable_ties(self):
        source = np.repeat(self.global_rows, 3, axis=0)
        first = plan_rois(source, self.hw, self.cfg)
        self.assertEqual(first, plan_rois(source.copy(), self.hw, self.cfg))
        self.assertEqual(first["rois"][0]["seed_ids"], [0, 1, 2])

    def test_class_diversity_precedes_second_same_class(self):
        source = rows([[10, 10, 60, 60, .24, 0], [15, 20, 65, 70, .23, 0], [550, 950, 590, 990, .06, 3]])
        plan = plan_rois(source, self.hw, self.cfg)
        self.assertEqual([d["global_id"] for d in plan["selection_decisions"][:2]], [0, 2])
        self.assertEqual(len(plan["rois"]), 2)

    def test_single_class_can_fill_both_slots(self):
        source = rows([[10, 10, 60, 60, .24, 0], [550, 950, 590, 990, .2, 0]])
        self.assertEqual(len(plan_rois(source, self.hw, self.cfg)["rois"]), 2)

    def test_coalescing_and_budget(self):
        source = rows([[10, 10, 60, 60, .24, 0], [15, 20, 65, 70, .23, 3],
                       [550, 950, 590, 990, .2, 0], [540, 10, 590, 60, .1, 0]])
        plan = plan_rois(source, self.hw, self.cfg)
        self.assertEqual(len(plan["rois"]), 2)
        self.assertIn("coalesced", [d["action"] for d in plan["selection_decisions"]])
        self.assertIn("budget_exhausted", [d["action"] for d in plan["selection_decisions"]])

    def test_overlap_without_containment_is_not_coalesced(self):
        source = rows([[10, 10, 60, 60, .2, 0], [550, 950, 590, 990, .2, 3]])
        with patch("wrist_eval.adaptive.overlap", return_value=.9):
            self.assertEqual(len(plan_rois(source, self.hw, self.cfg)["rois"]), 2)

    def test_containment_crop_borders_and_rounding(self):
        for w, h in [(600, 1000), (333, 1001), (1001, 333), (5, 5)]:
            for box in ([0, 0, .2*w, .2*h], [.8*w, .8*h, w, h], [.4*w, .4*h, .6*w, .6*h]):
                envelope, crop, reason = crop_geometry(box, w, h, self.cfg)
                self.assertIsNone(reason)
                self.assertTrue(contains(crop, envelope))
                self.assertTrue(contains([0, 0, w, h], crop))
                self.assertTrue(all(isinstance(v, int) for v in crop))
                self.assertLessEqual(max(w, h)/max(crop[2]-crop[0], crop[3]-crop[1]), 4/3+1e-12)

    def test_no_zoom_is_skipped(self):
        for w, h, box in [(600, 1000, [0, 0, 600, 1000]), (1, 1000, [0, 400, 1, 500])]:
            self.assertEqual(crop_geometry(box, w, h, self.cfg)[2], "no_zoom")

    def test_shift_once_and_original_bounds(self):
        roi = self.plan["rois"][0]
        view = self.view([[250, 450, 350, 550, .8, 0]])
        np.testing.assert_array_equal(shift_local(view["predictions"], roi, self.hw), rows([[250, 450, 350, 550, .8, 0]]))

    def test_actual_resize_gains_roundtrip(self):
        import torch
        from ultralytics.utils.ops import scale_boxes
        from wrist_eval.sliced import resize_and_pad
        image = np.zeros((751, 451, 3), np.uint8)
        padded, ratio = resize_and_pad(image)
        gy, gx = ratio[0]
        left, top = ratio[1]
        source = np.array([[12.5, 20.25, 210.5, 450.25]], dtype=np.float32)
        transformed = source*np.array([gx, gy, gx, gy])+[left, top, left, top]
        restored = scale_boxes(padded.shape[:2], torch.tensor(transformed, dtype=torch.float32), image.shape[:2], ratio).numpy()
        np.testing.assert_allclose(source, restored, atol=5e-5)

    def test_artificial_boundary_inclusive_and_physical_exemption(self):
        roi = {"box": [75, 125, 525, 875]}
        self.assertIn("left", boundary_sides([2, 20, 30, 50], roi, self.hw, (1, 1), self.cfg))
        self.assertNotIn("left", boundary_sides([2.1, 20, 30, 50], roi, self.hw, (1, 1), self.cfg))
        self.assertNotIn("left", boundary_sides([0, 20, 30, 50], {"box": [0, 0, 450, 750]}, self.hw, (1, 1), self.cfg))

    def test_wrong_class_cannot_confirm(self):
        result = self.fused([[250, 450, 350, 550, .9, 3]])
        np.testing.assert_array_equal(result["predictions"], self.global_rows)

    def test_local_confidence_threshold(self):
        self.assertEqual(len(self.fused([[250, 450, 350, 550, .25, 0]])["diagnostics"]["matches"]), 1)
        p = np.nextafter(np.float32(.25), np.float32(-np.inf))
        self.assertEqual(len(self.fused([[250, 450, 350, 550, p, 0]])["diagnostics"]["matches"]), 0)

    def test_protected_global_competes_for_evidence(self):
        source = rows([[250, 450, 350, 550, .9, 0], [260, 450, 360, 550, .2, 0]])
        plan = plan_rois(source, self.hw, self.cfg)
        result = self.fused(global_rows=source, plan=plan)
        self.assertEqual(result["diagnostics"]["protected_matches"], 1)
        np.testing.assert_array_equal(result["predictions"], source)

    def test_union_evidence_one_boost_only(self):
        plan = copy.deepcopy(self.plan)
        plan["rois"] = [{"id": 0, "box": [0, 0, 450, 750]}, {"id": 1, "box": [150, 250, 600, 1000]}]
        views = [self.view([[250, 450, 350, 550, .8, 0]], plan, i) for i in range(2)]
        result = fuse_predictions(self.global_rows, views, plan, self.cfg, "V1")
        self.assertEqual(len(result["diagnostics"]["evidence"]), 1)
        self.assertEqual(len(result["diagnostics"]["score_changes"]), 1)
        self.assertLessEqual(float(result["predictions"][0, 4])-float(self.global_rows[0, 4]), .10)

    def test_one_local_cannot_confirm_two_globals(self):
        source = rows([[250, 450, 350, 550, .2, 0], [255, 450, 355, 550, .19, 0]])
        result = self.fused(global_rows=source, plan=plan_rois(source, self.hw, self.cfg))
        self.assertEqual(len(result["diagnostics"]["matches"]), 1)

    def test_v1_geometry_and_unrelated_rows_immutable(self):
        source = rows([[10, 10, 30, 30, .9, 2], [250, 450, 350, 550, .2, 0], [30, 30, 50, 50, .1, 6]])
        result = self.fused(global_rows=source, plan=plan_rois(source, self.hw, self.cfg))
        ids = result["diagnostics"]["output_global_ids"]
        np.testing.assert_array_equal(result["predictions"][:, :4], source[ids, :4])
        for index, original_id in enumerate(ids):
            if original_id != 1:
                np.testing.assert_array_equal(result["predictions"][index], source[original_id])

    def test_float32_caps_never_round_up(self):
        for p in np.linspace(.05, .24999, 301, dtype=np.float32):
            for local in np.array([.25, .5, .9], np.float32):
                updated = confidence_update(p, local, self.cfg)
                self.assertGreaterEqual(updated, p)
                self.assertLessEqual(float(updated), min(float(p)+.1, float(local), .35))

    def test_v2_blend_and_shared_scores(self):
        local = [[260, 450, 360, 550, .8, 0]]
        a, b = self.fused(local, "V1"), self.fused(local, "V2")
        np.testing.assert_array_equal(a["predictions"][:, 4:], b["predictions"][:, 4:])
        np.testing.assert_allclose(b["predictions"][0, :4], [252.5, 450, 352.5, 550])

    def test_v2_below_point_five_local_score_keeps_geometry(self):
        result = self.fused([[260, 450, 360, 550, .49, 0]], "V2")
        np.testing.assert_array_equal(result["predictions"][:, :4], self.global_rows[:, :4])

    def test_v2_collision_veto(self):
        source = rows([[200, 400, 300, 500, .9, 0], [218, 400, 318, 500, .2, 0]])
        result = self.fused([[210, 400, 310, 500, .8, 0]], "V2", source, plan_rois(source, self.hw, self.cfg))
        self.assertEqual(result["diagnostics"]["geometry"][0]["reason"], "collision_veto")
        by_id = {i: row for i, row in zip(result["diagnostics"]["output_global_ids"], result["predictions"])}
        np.testing.assert_array_equal(by_id[1][:4], source[1, :4])

    def test_adversarial_blend_can_reduce_gt_iou(self):
        result = self.fused([[260, 450, 360, 550, .8, 0]], "V2")
        self.assertLess(overlap(result["predictions"][0, :4], self.global_rows[0, :4]), 1)

    def test_adversarial_confirmation_can_worsen_ranking(self):
        source = rows([[100, 100, 200, 200, .28, 0], [300, 500, 400, 600, .2, 0]])
        result = self.fused([[300, 500, 400, 600, .9, 0]], global_rows=source,
                            plan=plan_rois(source, self.hw, self.cfg))
        self.assertEqual(result["diagnostics"]["output_global_ids"], [1, 0])

    def test_empty_and_no_candidate_exact_return(self):
        for source in [rows(), rows([[10, 10, 20, 20, .8, 0]])]:
            plan = plan_rois(source, self.hw, self.cfg)
            result = fuse_predictions(source, [], plan, self.cfg, "V2")
            np.testing.assert_array_equal(result["predictions"], source)

    def test_empty_local_output_unchanged(self):
        np.testing.assert_array_equal(self.fused([])["predictions"], self.global_rows)

    def test_invalid_input_fails_before_filtering(self):
        for source in [rows([[1, 1, 1, 2, .2, 0]]), rows([[1, 1, 2, 2, float("nan"), 2]]),
                       rows([[1, 1, 2, 2, .2, 8]])]:
            with self.assertRaises(ValueError):
                plan_rois(source, self.hw, self.cfg)
        with self.assertRaises(ValueError):
            checked_rows(np.zeros((301, 6)), 600, 1000)

    def test_wrong_local_bundle_fails(self):
        with self.assertRaises(ValueError):
            fuse_predictions(self.global_rows, [], self.plan, self.cfg, "V1")

    def test_callback_pipeline_passes_and_source_immutability(self):
        image = np.zeros((1000, 600, 3), dtype=np.uint8)
        before = image.copy()
        calls = []
        local = self.view([[250, 450, 350, 550, .8, 0]])
        def callback(pixels, shape):
            calls.append(shape)
            return {"predictions": self.global_rows.copy(), "gains": (1, 1)} if len(calls) == 1 else local
        result = predict_adaptive(image, callback, (800, 480), self.cfg)
        self.assertEqual(result["passes"], 2)
        self.assertEqual(calls, [(800, 480), None])
        np.testing.assert_array_equal(image, before)
        self.assertEqual(set(result["variants"]), {"V1", "V2"})

    def test_no_candidate_callback_only_once(self):
        calls = []
        def callback(image, shape):
            calls.append(1)
            return {"predictions": rows(), "gains": (1, 1)}
        result = predict_adaptive(np.zeros((1000, 600, 3), np.uint8), callback, (800, 480), self.cfg)
        self.assertEqual(result["passes"], 1)
        self.assertEqual(len(calls), 1)

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            self.fused(variant="V3")

    def test_single_variant_uses_complete_view_path(self):
        calls = []
        def callback(image, shape):
            calls.append(1)
            return {"predictions": rows(), "gains": (1, 1)}
        result = predict_adaptive(np.zeros((1000, 600, 3), np.uint8), callback, (800, 480), self.cfg, ["V1"])
        self.assertEqual(list(result["variants"]), ["V1"])
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            predict_adaptive(np.zeros((1000, 600, 3), np.uint8), callback, (800, 480), self.cfg, ["V3"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
