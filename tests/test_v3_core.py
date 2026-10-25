"""Synthetic V3 ownership, score, geometry and physical-call safeguards."""

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

from wrist_eval.adaptive import plan_rois as legacy_plan
from wrist_v3.planning import plan_rois
from wrist_v3.fusion import confidence_update, fuse_predictions
from wrist_v3.core import predict

ROOT = Path(__file__).resolve().parents[1]
LEGACY = json.loads((ROOT/'configs/adaptive_refinement_proposed.json').read_text())
SETTINGS = json.loads((ROOT/'configs/cg_alr_v3.json').read_text())


def rows(values):
    return np.asarray(values, np.float32).reshape(-1, 6)


def fixture(global_rows, local_rows):
    plan = plan_rois(global_rows, (100, 100), LEGACY, SETTINGS)
    views = []
    for roi in plan['rois']:
        local = local_rows.copy()
        local[:, :4] -= np.array([*roi['box'][:2], *roi['box'][:2]], np.float32)
        views.append({'roi_id': roi['id'], 'predictions': local, 'gains': [768/75, 768/75]})
    return plan, views


class V3CoreTests(unittest.TestCase):
    def test_unique_confirmation_uses_midpoint_cap_and_original_coordinates(self):
        original = rows([[40, 40, 60, 60, .20, 0]])
        plan, views = fixture(original, rows([[40, 40, 60, 60, .32, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertAlmostEqual(float(out['predictions'][0, 4]), .26, places=6)
        np.testing.assert_array_equal(out['predictions'][:, :4], original[:, :4])
        self.assertEqual(len(out['diagnostics']['matches']), 1)

    def test_protected_overlap_candidate_requests_no_local_call(self):
        original = rows([[40, 40, 60, 60, .8, 0], [41, 40, 61, 60, .2, 0]])
        plan = plan_rois(original, (100, 100), LEGACY, SETTINGS)
        self.assertFalse(plan['rois'])
        self.assertEqual(plan['candidate_evaluation'][0]['reason'], 'protected_conflict')

    def test_protected_evidence_is_reserved_even_when_weak_global_boxes_do_not_overlap_enough(self):
        original = rows([[40, 40, 60, 60, .2, 0], [48, 40, 68, 60, .8, 0]])
        plan, views = fixture(original, rows([[44, 40, 64, 60, .9, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        np.testing.assert_array_equal(out['predictions'], original)
        self.assertEqual(out['diagnostics']['reserved_evidence'][0]['protected_ids'], [1])

    def test_one_local_row_cannot_validate_two_weak_duplicates(self):
        original = rows([[40, 40, 60, 60, .2, 0], [41, 40, 61, 60, .18, 0]])
        plan, views = fixture(original, rows([[40, 40, 60, 60, .8, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertFalse(out['diagnostics']['matches'])
        self.assertTrue(all(v['reason'] == 'ambiguous_local_degree' for v in out['diagnostics']['candidate_matching']))

    def test_multiple_local_owners_are_ambiguous_not_extra_confirmation(self):
        original = rows([[40, 40, 60, 60, .2, 0]])
        plan, views = fixture(original, rows([[36, 40, 56, 60, .8, 0], [44, 40, 64, 60, .7, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertEqual(out['diagnostics']['candidate_matching'][0]['reason'], 'ambiguous_global_degree')
        self.assertFalse(out['diagnostics']['score_changes'])

    def test_sequential_promotion_guard_prevents_two_overlapping_operating_rows(self):
        original = rows([[40, 40, 60, 60, .22, 0], [46, 40, 66, 60, .2, 0]])
        plan, views = fixture(original, rows([[37, 40, 57, 60, .8, 0], [49, 40, 69, 60, .8, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertEqual(len(out['diagnostics']['matches']), 2)
        self.assertEqual(sum(out['predictions'][:, 4] >= .25), 1)
        self.assertEqual(out['diagnostics']['arbitration'][1]['reason'], 'operating_conflict')

    def test_no_candidate_and_rare_class_predictions_are_exactly_preserved(self):
        original = rows([[20, 20, 40, 40, .2, 5], [50, 20, 70, 40, .8, 3]])
        plan = plan_rois(original, (100, 100), LEGACY, SETTINGS)
        self.assertFalse(plan['rois'])
        out = fuse_predictions(original, [], plan, LEGACY, SETTINGS)
        np.testing.assert_array_equal(out['predictions'], original)

    def test_empty_predictions_are_valid(self):
        original = rows([]); plan = plan_rois(original, (100, 100), LEGACY, SETTINGS)
        out = fuse_predictions(original, [], plan, LEGACY, SETTINGS)
        self.assertEqual(out['predictions'].shape, (0, 6))

    def test_no_local_evidence_does_not_update_or_reorder(self):
        original = rows([[40, 40, 60, 60, .2, 3]])
        plan, views = fixture(original, rows([]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        np.testing.assert_array_equal(out['predictions'], original)

    def test_wrong_class_and_low_local_score_cannot_confirm(self):
        original = rows([[40, 40, 60, 60, .2, 0]])
        plan, views = fixture(original, rows([[40, 40, 60, 60, .9, 3], [40, 40, 60, 60, .249, 0]]))
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertFalse(out['diagnostics']['score_changes'])

    def test_planner_matches_V2_exactly_when_no_protected_conflict_exists(self):
        original = rows([[20, 20, 40, 40, .2, 0], [50, 50, 70, 70, .18, 3]])
        old, new = legacy_plan(original, (100, 100), LEGACY), plan_rois(original, (100, 100), LEGACY, SETTINGS)
        self.assertEqual(new['rois'], old['rois'])
        self.assertEqual(new['selection_decisions'], old['selection_decisions'])

    def test_score_caps_quantize_conservatively_at_float_boundaries(self):
        for p in [np.float32(.05), np.nextafter(np.float32(.15), np.float32(0)), np.float32(.15), np.float32(.249999)]:
            for q in [np.float32(.25), np.float32(.35), np.float32(.9)]:
                value = confidence_update(p, q, SETTINGS)
                self.assertLessEqual(float(value), min(float(p)+.1, (float(p)+float(q))/2, .35))
                self.assertGreaterEqual(value, p)

    def test_replays_are_deterministic_and_input_rows_are_not_mutated(self):
        original = rows([[40, 40, 60, 60, .2, 0]]); copy = original.copy()
        plan, views = fixture(original, rows([[40, 40, 60, 60, .8, 0]]))
        a = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        b = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertEqual(a['diagnostics'], b['diagnostics'])
        np.testing.assert_array_equal(original, copy)

    def test_actual_callback_count_is_bounded_without_any_gt_parameter(self):
        calls = []
        def view(image, shape):
            calls.append(shape)
            return {'predictions': rows([]), 'gains': [1., 1.], 'counters': {'forward_passes': 1}}
        result = predict(np.zeros((100, 100, 3), np.uint8), view, [128, 128], LEGACY, SETTINGS)
        self.assertEqual(result['passes'], len(calls))
        self.assertEqual(len(calls), 1)
        self.assertNotIn('gt', inspect.signature(predict).parameters)
        self.assertNotIn('annotations', inspect.signature(fuse_predictions).parameters)

    def test_hidden_extra_view_forward_is_rejected(self):
        def view(image, shape):
            return {'predictions': rows([]), 'gains': [1., 1.], 'counters': {'forward_passes': 2}}
        with self.assertRaises(AssertionError):
            predict(np.zeros((100, 100, 3), np.uint8), view, [128, 128], LEGACY, SETTINGS)

    def test_two_local_views_make_exactly_three_physical_calls(self):
        calls = []
        original = rows([[100, 100, 120, 120, .2, 0], [850, 850, 870, 870, .19, 3]])
        def view(image, shape):
            calls.append(shape)
            return {'predictions': original.copy() if shape is not None else rows([]),
                    'gains': [1., 1.], 'counters': {'forward_passes': 1}}
        result = predict(np.zeros((1000, 1000, 3), np.uint8), view, [768, 768], LEGACY, SETTINGS)
        self.assertEqual((len(result['plan']['rois']), result['passes'], len(calls)), (2, 3, 3))

    def test_overlapping_views_restore_once_and_union_duplicate_evidence(self):
        original = rows([[45, 45, 55, 55, .2, 0]])
        plan = plan_rois(original, (100, 100), LEGACY, SETTINGS)
        plan['rois'] = [{'id': 0, 'box': [0, 0, 75, 75], 'seed_ids': [0], 'ideal_zoom': 4/3},
                        {'id': 1, 'box': [25, 25, 100, 100], 'seed_ids': [0], 'ideal_zoom': 4/3}]
        views = [{'roi_id': 0, 'predictions': rows([[45, 45, 55, 55, .8, 0]]), 'gains': [10., 10.]},
                 {'roi_id': 1, 'predictions': rows([[20, 20, 30, 30, .8, 0]]), 'gains': [10., 10.]}]
        out = fuse_predictions(original, views, plan, LEGACY, SETTINGS)
        self.assertEqual(len(out['diagnostics']['evidence']), 1)
        self.assertEqual(len(out['diagnostics']['matches']), 1)
        np.testing.assert_array_equal(out['predictions'][:, :4], original[:, :4])

    def test_invalid_geometry_and_nonfinite_global_outputs_stop(self):
        for value in ([[0, 0, 0, 2, .2, 0]], [[0, 0, 2, 2, float('nan'), 0]]):
            with self.assertRaises(ValueError):
                plan_rois(rows(value), (100, 100), LEGACY, SETTINGS)


if __name__ == '__main__':
    unittest.main()
