"""One-to-one oracle capacity and descriptive separation safeguards."""

import unittest

import numpy as np

from wrist_analysis.calculations import cardinality, oracle_capacity, summarize, superiority, feature_groups


class DiagnosticCalculationTests(unittest.TestCase):
    def test_oracle_cannot_recover_two_targets_with_one_proposal(self):
        self.assertEqual(cardinality({0: [10], 1: [10]}), 1)

    def test_oracle_reassigns_a_shared_proposal_instead_of_greedy_undercounting(self):
        self.assertEqual(cardinality({0: [10, 11], 1: [10]}), 2)

    def test_oracle_empty_edges_are_not_recoverable(self):
        self.assertEqual(cardinality({0: [], 1: []}), 0)
        self.assertEqual(cardinality({}), 0)

    def test_oracle_respects_the_stated_iou_and_available_proposal_set(self):
        matrix = np.array([[.50, .49, .8], [.49, .7, .8]])
        self.assertEqual(oracle_capacity(matrix, [0, 1], [0]), 1)
        self.assertEqual(oracle_capacity(matrix, [0, 1], [0, 1]), 2)
        self.assertEqual(oracle_capacity(matrix, [0, 1], []), 0)

    def test_separation_is_half_for_ties_and_complements_when_reversed(self):
        self.assertEqual(superiority([1, 1], [1]), .5)
        self.assertEqual(superiority([2], [1]), 1.)
        a, b = [1, 2, 3], [2, 3, 4]
        self.assertAlmostEqual(superiority(a, b)+superiority(b, a), 1.)

    def test_missing_outcome_group_has_no_invented_separation(self):
        self.assertIsNone(superiority([], [1]))
        self.assertIsNone(superiority([1], []))
        self.assertEqual(summarize([]), {'n': 0})

    def test_group_support_counts_independent_patients_not_events(self):
        values = [{'class': 'fracture', 'stem': 'a', 'patient_id': 'p', 'score': .1},
                  {'class': 'fracture', 'stem': 'b', 'patient_id': 'p', 'score': .3},
                  {'class': 'fracture', 'stem': 'b', 'patient_id': 'q', 'score': .5}]
        group = feature_groups(values, ('class',), ('score',))[0]
        self.assertEqual((group['events'], group['images'], group['patients']), (3, 2, 2))
        self.assertEqual(group['features']['score']['median'], .3)
        self.assertNotIn('patient_id', group)
        self.assertNotIn('stem', group)

    def test_oracle_matches_exhaustive_small_graph_capacity(self):
        # Enumerate every 3x3 adjacency graph with an independent assignment search.
        import itertools
        for mask in range(512):
            edges = {g: [p for p in range(3) if mask & (1 << (3*g+p))] for g in range(3)}
            brute = max(sum(p >= 0 for p in choices) for choices in itertools.product((-1, 0, 1, 2), repeat=3)
                if len([p for p in choices if p >= 0]) == len({p for p in choices if p >= 0})
                and all(p < 0 or p in edges[g] for g, p in enumerate(choices)))
            self.assertEqual(cardinality(edges), brute)

    def test_promoted_tp_is_not_necessarily_a_recovered_target(self):
        from wrist_eval.metrics import match
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        original = np.array([[0, 0, 10, 10, .26, 0], [0, 0, 10, 10, .20, 0]], np.float32)
        output = original.copy(); output[1, 4] = .30
        before_g, before_p, _ = match(gt, original)
        after_g, after_p, _ = match(gt, output)
        self.assertEqual(before_p.tolist(), [0, -1])
        self.assertEqual(after_p.tolist(), [-1, 0])
        self.assertFalse(np.any((before_g < 0) & (after_g >= 0)))
        self.assertEqual(int(((before_p >= 0) & (after_p < 0)).sum()), 1)

    def test_independent_check_preserves_score_ties_class_and_confidence_rules(self):
        from wrist_analysis.verify import independent_match
        from wrist_eval.metrics import match
        gt = np.array([[0, 0, 10, 10, 0], [20, 20, 30, 30, 3]], np.float32)
        pred = np.array([[0, 0, 10, 10, .3, 0], [0, 0, 10, 10, .3, 0],
                         [20, 20, 30, 30, .8, 0], [20, 20, 30, 30, .249, 3]], np.float32)
        g, p = independent_match(gt, pred)
        expected_g, expected_p, _ = match(gt, pred)
        self.assertEqual(g.tolist(), [0, -1])
        np.testing.assert_array_equal(g, expected_g)
        np.testing.assert_array_equal(p, expected_p)


if __name__ == '__main__':
    unittest.main()
