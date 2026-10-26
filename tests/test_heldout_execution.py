"""Pre-access orchestration checks using synthetic arrays and temporary journals."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from wrist_final.protocol import ROOT, read, MATRIX
from wrist_final.execution import access_guard, PRIVATE
from wrist_final.test_data import population
from wrist_final.metrics import final_stats, paired_bootstrap, canonical_predictions
from wrist_final.test_analysis import bootstrap, gates
from wrist_final.test_verification import independent_flags, independent_ap, independent_operating, same_ap
from wrist_final.test_reports import render
from wrist_eval.metrics import match
from wrist_v3.metrics import deterministic_ap


def fixture():
    gt = np.array([[i*30, 0, i*30+10, 10, i] for i in range(8)], np.float32)
    pred = np.array([[*g[:4], .9-i*.01, i] for i, g in enumerate(gt)]+[[250, 20, 260, 30, .5, 0]], np.float32)
    pred = canonical_predictions(pred, 400, 100)
    return gt, pred, final_stats(gt, pred, 400, 100, 0)


class HeldoutExecutionTests(unittest.TestCase):
    def test_independent_assignment_and_integration_agree(self):
        gt, pred, stat = fixture()
        np.testing.assert_array_equal(independent_flags(gt, pred), stat['tp'])
        same_ap(independent_ap([stat]), deterministic_ap([stat]))
        a, b = independent_operating(gt, pred); x, y, _ = match(gt, pred)
        np.testing.assert_array_equal(a, x); np.testing.assert_array_equal(b, y)

    def test_native_assignment_with_empty_inputs_and_duplicates(self):
        for gt, pred in ((np.empty((0, 5), np.float32), np.empty((0, 6), np.float32)),
                         (np.array([[0, 0, 10, 10, 0]], np.float32),
                          np.array([[0, 0, 10, 10, .3, 0], [0, 0, 10, 10, .3, 0]], np.float32))):
            stat = final_stats(gt, pred, 100, 100, 0)
            np.testing.assert_array_equal(independent_flags(gt, pred), stat['tp'])
            same_ap(independent_ap([stat]), deterministic_ap([stat]))

    def test_journal_bootstrap_exactly_matches_frozen_helper(self):
        _, _, stat = fixture()
        rows = [{'patient_id': 'a'}, {'patient_id': 'b'}]
        stats = {'final_A': [stat, stat], 'final_C': [stat, stat]}
        counts = {m: np.ones((2, 8, 3), int) for m in stats}
        coefficients = {'primary': {'final_C': 1, 'final_A': -1}}
        expected = paired_bootstrap(rows, stats, coefficients, counts=counts)
        with tempfile.TemporaryDirectory() as folder:
            actual = bootstrap(rows, stats, counts, coefficients, Path(folder)/'bootstrap.jsonl')
        actual.pop('journal_sha256')
        self.assertEqual(actual, expected)

    def test_population_rejects_patient_leakage(self):
        records = [{'patient_id': 'same', 'split': 'train'}, {'patient_id': 'same', 'split': 'test'}]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            population(records)

    def test_population_requires_exact_frozen_counts(self):
        records = [{'patient_id': str(i % 913), 'split': 'test', 'filestem': str(i),
                    'boxes': [{}]*(3 if i < 1005 else 2)} for i in range(3178)]
        self.assertEqual(len(population(records)), 3178)
        with self.assertRaises(ValueError):
            population(records[:-1])

    def test_data_and_historical_writes_are_blocked(self):
        for path in (ROOT/'data/prepared/example', ROOT/'outputs/training/example'):
            with self.assertRaises(PermissionError):
                access_guard('open', (path, 'w', 0))
        access_guard('open', (PRIVATE/'synthetic_example', 'w', 0))

    def test_primary_interval_gate_is_not_replaced_by_a_point_gain(self):
        cfg = read(MATRIX); data = synthetic_results()['test']
        ci = data['bootstrap']; ci['contrasts']['v3_standard_training']['overall']['map50_95']['lower'] = -.01
        actual = gates(data['AP'], data['operating'], ci, data['runtime'], cfg)
        self.assertFalse(actual['final_C']['checks']['overall_lower95_positive'])
        self.assertFalse(actual['final_C']['strict_success'])

    def test_reports_cover_six_conditions_classes_and_limits(self):
        reports = render(synthetic_results())
        self.assertEqual(len(reports), 3)
        text = '\n'.join(reports.values())
        for value in ('final_A', 'final_F', 'bonelesion', 'fracture', 'seed 314159', 'Failed checks'):
            self.assertIn(value, text)


def synthetic_results():
    from wrist_eval.common import CLASSES
    from wrist_final.test_analysis import METHODS
    cfg = read(MATRIX); ap = deterministic_ap([fixture()[2]])
    result = {'images': 2, 'patients': 2, 'AP': {m: ap for m in METHODS},
        'operating': {m: {c: {'tp': 1, 'fp': 1, 'fn': 1, 'precision': .5, 'recall': .5} for c in [*CLASSES, 'overall']} for m in METHODS},
        'support': {c: {'objects': 1, 'patients': 1, 'images': 1} for c in CLASSES}, 'runtime': {}, 'recovery': {},
        'AP_contrasts': {}, 'bootstrap': {'contrasts': {}}, 'engineering_gates': {}}
    for method in METHODS:
        timing = {'e2e_seconds': {'mean': .1, 'median': .1, 'p95': .2}, 'forward_passes': {'mean': 1., 'max': 3},
                  'allocated_peak_bytes': {'max': 100}, 'reserved_peak_bytes': {'max': 200}}
        result['runtime'][method] = {'timing': timing, 'usage': {'calls': 2}, 'mean_paired_latency_ratio': 1.5}
    for name in cfg['comparisons']:
        result['AP_contrasts'][name] = {c: {k: 0. for k in ap[c]} for c in ap}
        result['bootstrap']['contrasts'][name] = {c: {k: {'lower': -.01, 'upper': .01, 'replicates_present': 500} for k in ap[c]} for c in ap}
    for pair in ('final_C-final_A', 'final_F-final_D'):
        result['recovery'][pair] = {c: {'both': 1, 'recovered': 0, 'lost': 0, 'neither': 0} for c in CLASSES}
    result['engineering_gates'] = gates(result['AP'], result['operating'], result['bootstrap'], result['runtime'], cfg)
    return {'test': result, 'validation': result}
