"""Synthetic final-matrix, content-order, bootstrap and execution-boundary checks."""

import inspect
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from wrist_final.protocol import ROOT, read, rectangular_shapes, phase5c_guard, agents_local_only
from wrist_final.protocol import verify_frozen
from wrist_final.metrics import (canonical_predictions, final_stats, operating_counts, contrasts,
                                 count_summary, paired_bootstrap)
from wrist_final.inference import FrozenPredictor, create_predictor
from wrist_final.verify import build_matrix
from wrist_v3.metrics import deterministic_ap, historical_ap


def rows(values):
    return np.asarray(values, np.float32).reshape(-1, 6)


class FinalMetricTests(unittest.TestCase):
    def test_all_six_labels_use_the_same_order_for_new_and_existing_rows(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        pred = rows([[20, 20, 30, 30, .3, 0], [0, 0, 10, 10, .3, 0],
                     [5, 5, 15, 15, .3, 3], [0, 0, 10, 10, .3, 0]])
        expected = deterministic_ap([final_stats(gt, pred, 100, 100, 0)])
        for _ in range(6):
            for seed in range(8):
                actual = pred[np.random.default_rng(seed).permutation(len(pred))]
                self.assertEqual(deterministic_ap([final_stats(gt, actual, 100, 100, 0)]), expected)

    def test_canonicalization_preserves_every_float_and_duplicate(self):
        pred = rows([[20, 20, 30, 30, .3, 0], [0, 0, 10, 10, .30000004, 0], [0, 0, 10, 10, .3, 0]])
        copy = pred.copy(); result = canonical_predictions(pred, 100, 100)
        np.testing.assert_array_equal(pred, copy)
        self.assertEqual(sorted(map(tuple, result)), sorted(map(tuple, pred)))
        self.assertEqual(result[0, 4], pred[1, 4])

    def test_no_tie_eight_class_AP_matches_pinned_integration(self):
        gt = np.array([[i*30, 0, i*30+10, 10, i] for i in range(8)], np.float32)
        pred = rows([[*g[:4], .95-i*.02, i] for i, g in enumerate(gt)])
        stat = final_stats(gt, pred, 400, 100, 0)
        self.assertEqual(deterministic_ap([stat]), historical_ap([stat]))

    def test_empty_predictions_present_class_and_absent_classes_are_distinct(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        ap = deterministic_ap([final_stats(gt, rows([]), 100, 100, 0)])
        self.assertEqual(ap['fracture']['ap50_95'], 0.)
        self.assertNotIn('bonelesion', ap); self.assertNotIn('overall', ap)
        self.assertEqual(deterministic_ap([final_stats([], rows([]), 100, 100, 0)]), {})

    def test_invalid_outputs_and_annotations_are_rejected(self):
        for pred in (rows([[0, 0, 10, 10, np.nan, 0]]), rows([[0, 0, 101, 10, .3, 0]]),
                     np.tile(rows([[0, 0, 10, 10, .3, 0]]), (301, 1))):
            with self.assertRaises(ValueError):
                canonical_predictions(pred, 100, 100)
        with self.assertRaises(ValueError):
            final_stats([[0, 0, 110, 10, 0]], rows([]), 100, 100, 0)

    def test_operating_counts_and_zero_denominators(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        counts = operating_counts(gt, rows([[0, 0, 10, 10, .25, 0], [20, 20, 30, 30, .3, 0]]), 100, 100)
        self.assertEqual(counts['fracture'], {'tp': 1, 'fp': 1, 'fn': 0})
        summary = count_summary([[v[k] for k in ('tp', 'fp', 'fn')] for v in counts.values()])
        self.assertEqual(summary['overall']['precision'], .5)
        self.assertNotIn('recall', summary['bonelesion'])
        self.assertNotIn('precision', summary['bonelesion'])

    def test_paired_bootstrap_keeps_missing_AP_undefined_and_reuses_draws(self):
        gt = np.array([[i*30, 0, i*30+10, 10, i] for i in range(8) if i != 5], np.float32)
        rare = np.array([[0, 0, 10, 10, 5]], np.float32)
        stats = [final_stats(gt, rows([]), 400, 100, 0), final_stats(rare, rows([]), 100, 100, 1)]
        population = [{'patient_id': 'a'}, {'patient_id': 'b'}]
        data = {f'final_{c}': stats for c in 'ABCDEF'}
        coefficients = {'primary': {'final_C': 1, 'final_A': -1}}
        counts = {m: np.zeros((2, 8, 3), int) for m in data}
        value = paired_bootstrap(population, data, coefficients, counts=counts)
        ci = value['contrasts']['primary']['overall']['map50_95']
        self.assertTrue(0 < ci['replicates_present'] < 500)
        self.assertEqual((ci['lower'], ci['upper']), (0., 0.))
        self.assertEqual(value['operating_contrasts']['primary']['overall']['tp']['replicates_present'], 500)
        # Independent draws keep every image of each sampled patient, including repeated patients.
        from wrist_eval.bootstrap import patient_groups, resample_indices
        from wrist_final.protocol import identity
        rng = np.random.default_rng(314159)
        expected = [identity([int(i) for i in rng.integers(0, 2, size=2)]) for _ in range(500)]
        self.assertEqual(value['draw_sequence_sha256'], identity(expected))
        self.assertEqual(patient_groups([{'patient_id': 'b'}, {'patient_id': 'a'}, {'patient_id': 'b'}]), [[1], [0, 2]])
        draw = resample_indices([[1], [0, 2]], np.random.default_rng(4))
        self.assertTrue(draw in ([1, 1], [1, 0, 2], [0, 2, 1], [0, 2, 0, 2]))

    def test_bootstrap_policy_cannot_be_changed(self):
        for kwargs in ({'replicates': 1}, {'seed': 42}):
            with self.assertRaises(ValueError):
                paired_bootstrap([], {}, {}, **kwargs)

    def test_interaction_sign_and_missing_metric_handling(self):
        values = {f'final_{c}': {'overall': {'map50_95': float(i)}} for i, c in enumerate('ABCDEF')}
        result = contrasts(values, {'interaction': {'final_F': 1, 'final_D': -1, 'final_C': -1, 'final_A': 1}})
        self.assertEqual(result['interaction']['overall']['map50_95'], 0.)
        self.assertEqual(contrasts({'a': {}, 'b': {'fracture': {'ap50': 0.}}}, {'x': {'a': 1, 'b': -1}})['x'], {})


class FakeModel:
    def register_forward_pre_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=lambda: None)

    def forward(self):
        self.callback(self, ())


class FinalRoutingTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.v3 = read(ROOT/'configs/cg_alr_v3.json')
        self.legacy = read(ROOT/'configs/adaptive_refinement_proposed.json')
        self.image = np.zeros((1000, 1000, 3), np.uint8)

    def predictor(self, global_rows=None, hidden=0, bound=None):
        original = rows([]) if global_rows is None else global_rows
        def view(image, shape):
            for _ in range(1+hidden):
                self.model.forward()
            return {'predictions': original.copy() if shape is not None else rows([]),
                    'gains': [1., 1.], 'counters': {'forward_passes': 1}}
        def sliced(image, shape):
            for _ in range(5):
                self.model.forward()
            return original.copy(), {'forward_passes': 5, 'slices': 4}
        return FrozenPredictor(self.model, view, sliced, self.v3, self.legacy, inference=bound)

    def test_standard_and_sahi_have_one_and_five_physical_calls(self):
        for method, expected in (('standard', 1), ('sahi_context', 5)):
            predictor = self.predictor()
            value = predictor(self.image, method, [768, 768], started_at=time.perf_counter())
            self.assertEqual(value['actual_model_calls'], expected)
            self.assertGreaterEqual(value['e2e_seconds'], 0.)
            predictor.close()

    def test_v3_retains_exact_three_call_budget(self):
        original = rows([[100, 100, 120, 120, .2, 0], [850, 850, 870, 870, .19, 3]])
        predictor = self.predictor(original)
        value = predictor(self.image, 'cg_alr_v3', [768, 768], started_at=time.perf_counter())
        self.assertEqual(value['actual_model_calls'], 3)
        self.assertGreater(value['paired_global_e2e_seconds'], 0.)
        np.testing.assert_array_equal(value['predictions'], original)
        predictor.close()

    def test_hidden_forward_is_detected(self):
        predictor = self.predictor(hidden=1)
        with self.assertRaises(ValueError):
            predictor(self.image, 'standard', [768, 768], started_at=time.perf_counter())
        predictor.close()

    def test_bound_condition_cannot_switch_inference(self):
        predictor = self.predictor(bound='cg_alr_v3')
        with self.assertRaises(ValueError):
            predictor(self.image, 'standard', [768, 768], started_at=time.perf_counter())
        self.assertEqual(predictor.physical_calls, 0)

    def test_factory_stops_before_model_loading_without_authorization(self):
        with self.assertRaises(PermissionError):
            create_predictor('final_C', ROOT/'outputs/final_evaluation/not_run')
        self.assertNotIn('gt', inspect.signature(FrozenPredictor.__call__).parameters)


class FinalProtocolTests(unittest.TestCase):
    def test_runtime_requires_completed_preflight_before_any_identity_scan(self):
        with patch('wrist_final.protocol.settings', return_value={}), patch('wrist_final.protocol.read', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                verify_frozen()

    def test_matrix_tampering_is_rejected_before_using_its_policy(self):
        with patch('wrist_final.protocol.settings', return_value={}), patch('wrist_final.protocol.sha', return_value='changed'), \
             patch('wrist_final.protocol.read', return_value={'status': 'READY', 'matrix_sha256': 'frozen'}):
            with self.assertRaisesRegex(ValueError, 'completed preflight'):
                verify_frozen()

    def test_six_mapping_comparisons_and_primary_are_exact(self):
        cfg = build_matrix()
        self.assertEqual([v['id'] for v in cfg['conditions']], ['final_'+v for v in 'ABCDEF'])
        self.assertEqual([v['inference'] for v in cfg['conditions']], ['standard', 'sahi_context', 'cg_alr_v3']*2)
        self.assertEqual(cfg['primary_proposed_configuration'], 'final_C')
        self.assertEqual(len(cfg['comparisons']), 11)
        self.assertTrue(all(sum(v.values()) == 0 for v in cfg['comparisons'].values()))
        self.assertFalse(cfg['execution']['test_authorized_in_phase5c'])

    def test_guard_blocks_test_data_and_mixed_annotation_containers(self):
        paths = ['data/prepared/grazpedwri_detection/images/test/example.png',
                 'data/prepared/grazpedwri_detection/labels/test/example.txt',
                 'outputs/dataset_preparation/source_records.jsonl', 'data/raw/example.png', '.env', '.env.local']
        for name in paths:
            with self.assertRaises(PermissionError):
                phase5c_guard('open', (ROOT/name, 'r', 0))
        phase5c_guard('open', (ROOT/'docs/evidence/dataset_preparation/aggregate_statistics.json', 'r', 0))

    def test_rectangular_shape_schedule_and_duplicate_guard(self):
        rows_ = [{'stem': str(i), 'height': 1000, 'width': 500} for i in range(13)]
        self.assertTrue(all(value == [800, 416] for value in rectangular_shapes(rows_).values()))
        with self.assertRaises(ValueError):
            rectangular_shapes(rows_+[rows_[0]])

    def test_local_agents_is_ignored_untracked_and_unstaged(self):
        result = agents_local_only()
        self.assertTrue(result['exists_locally'] and result['ignored'])
        self.assertFalse(result['tracked'] or result['staged'])


if __name__ == '__main__':
    unittest.main()
