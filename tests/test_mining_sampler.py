"""Cap/quota feasibility and matched full-image sampling checks."""

import itertools
import json
import unittest
import numpy as np

from wrist_mining.sampling import offers, feasible, select_extras, MatchedPopulation
from wrist_mining.analysis import concentration
from wrist_mining.common import ROOT


class MiningSamplerTests(unittest.TestCase):
    def test_quota_reservation_prevents_greedy_patient_starvation(self):
        patients, strata = np.array([0, 0, 1]), np.array([0, 1, 0])
        weights = np.array([1000., 1., 1.])
        chosen = select_extras([0, 1, 2], weights, patients, strata, [1, 1], [1, 1], np.random.default_rng(42))
        self.assertEqual(set(chosen), {1, 2})

    def test_infeasible_matched_budget_stops(self):
        with self.assertRaises(ValueError):
            select_extras([0, 1], np.ones(2), np.array([0, 0]), np.array([0, 1]), [1], [1, 1], np.random.default_rng(42))

    def test_empty_budget_and_zero_weight_validation(self):
        self.assertEqual(len(select_extras([], np.ones(1), np.array([0]), np.array([0]), [1], [0, 0], np.random.default_rng(1))), 0)
        with self.assertRaises(ValueError):
            select_extras([0], np.zeros(1), np.array([0]), np.array([0]), [1], [1, 0], np.random.default_rng(1))

    def test_ticket_order_does_not_change_seeded_selection(self):
        args = (np.arange(1, 7)/6, np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1]), [1, 1, 1], [1, 1])
        a = select_extras(range(6), *args, np.random.default_rng(42))
        b = select_extras(list(reversed(range(6))), *args, np.random.default_rng(42))
        self.assertEqual(a.tolist(), b.tolist())

    def test_feasibility_matches_exhaustive_small_allocations(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            patients = rng.integers(0, 3, size=6)
            strata = rng.integers(0, 2, size=6)
            capacity = rng.integers(0, 3, size=3)
            quota = rng.integers(0, 3, size=2)
            available = np.zeros((3, 2), int)
            np.add.at(available, (patients, strata), 1)
            total = sum((offers(c, a) for c, a in zip(capacity, available)), np.zeros(3, int))
            solutions = []
            for flags in itertools.product((False, True), repeat=6):
                chosen = np.flatnonzero(flags)
                if np.array_equal(np.bincount(strata[chosen], minlength=2), quota) and np.all(np.bincount(patients[chosen], minlength=3) <= capacity):
                    solutions.append(chosen)
            self.assertEqual(feasible(total, quota), bool(solutions))
            if solutions:
                chosen = select_extras(range(6), np.ones(6), patients, strata, capacity, quota, np.random.default_rng(13))
                self.assertTrue(np.array_equal(np.bincount(strata[chosen], minlength=2), quota))

    def test_patient_concentration_and_zero_support(self):
        value = concentration([0, 0, 2, 2])
        self.assertEqual(value["patients_nonzero"], 2)
        self.assertAlmostEqual(value["gini"], .5)
        self.assertAlmostEqual(value["effective_patients"], 2)
        self.assertEqual(concentration([0, 0])["total"], 0)

    def population(self, alpha):
        rows = []
        for patient in range(50):
            for view in range(4):
                labels = ([0] if view < 2 else []) + ([3] if patient < 40 and view == 0 else []) + ([5] if patient == 49 and view == 1 else [])
                rows.append({"split": "train", "filestem": f"fixture_{patient:02}_{view}", "patient_id": f"p{patient}",
                             "boxes": [{"class_id": c} for c in labels]})
        spec = json.loads((ROOT/"configs/class_aware_candidates.json").read_text())
        return MatchedPopulation(rows, spec, np.full(len(rows), .5),
                                 {"replacement_fraction": .05, "unique_stratum_minimum": .95, "difficulty_weight": alpha})

    def test_zero_error_weight_is_an_exact_paired_control(self):
        pop = self.population(0)
        rare, rf, guided, gf, _ = pop.paired(42, 0)
        self.assertTrue(np.array_equal(rare, guided))
        self.assertTrue(np.array_equal(rf, gf))

    def test_matched_budget_coverage_and_whole_lesion_cohort(self):
        pop = self.population(.5)
        for epoch in range(4):
            rare, rf, guided, gf, _ = pop.paired(42, epoch)
            self.assertEqual(int(rf.sum()), int(gf.sum()))
            self.assertTrue(np.array_equal(np.unique(rare), np.unique(guided)))
            repeats = np.bincount(guided, minlength=pop.n)
            self.assertTrue(np.all(repeats[np.isin(pop.patient, pop.lesion_patients)] == 1))
            self.assertEqual(len(rare), 200)


if __name__ == "__main__":
    unittest.main()
