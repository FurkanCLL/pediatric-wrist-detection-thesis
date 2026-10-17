"""Keep the predefined decision gates separate from attractive point estimates."""

import copy
import unittest
import numpy as np

from wrist_eval.adaptive_analysis import operating_values, point_counts
from wrist_eval.adaptive_config import approved_settings
from wrist_eval.adaptive_validation import assess_variant


class AdaptiveResultTests(unittest.TestCase):
    def setUp(self):
        self.limits = approved_settings()[0]["phase_3c2_plan"]["proposed_success"]
        point = {"total_fp": 100, "classes": {"fracture": {"fp": 50}}}
        delta = {"overall_ap_delta": {"map50": .001, "map50_95": .001},
                 "per_class_ap_delta": {"fracture": {k: .001 for k in ("ap50", "ap75", "ap50_95")}}}
        self.metrics = {"methods": {m: {"operating_points": {"conf0.25_iou0.5": copy.deepcopy(point)}} for m in ("standard_serial", "V1")},
                        "comparisons": {"V1_minus_standard_serial": delta}}
        self.ci = {"comparisons": {"V1_minus_standard_serial": {
            "overall": {"map50_95": {"lower": .0001}},
            "fracture": {k: {"lower": .0001} for k in ("ap50", "ap75", "ap50_95")},
            "periostealreaction": {"ap50_95": {"lower": .0001}}}}}

    def assess(self, seconds=1.5):
        return assess_variant("V1", self.metrics, self.ci, seconds, 1, self.limits)

    def test_all_predefined_gates_pass(self):
        self.assertTrue(self.assess()["primary_success"])

    def test_small_uncertain_gain_is_not_primary_success(self):
        self.ci["comparisons"]["V1_minus_standard_serial"]["overall"]["map50_95"]["lower"] = 0
        self.assertFalse(self.assess()["primary_success"])
        for value in self.ci["comparisons"]["V1_minus_standard_serial"]["fracture"].values():
            value["lower"] = 0
        self.ci["comparisons"]["V1_minus_standard_serial"]["periostealreaction"]["ap50_95"]["lower"] = 0
        self.assertFalse(self.assess()["passes"])

    def test_secondary_requires_nonnegative_overall_point(self):
        self.ci["comparisons"]["V1_minus_standard_serial"]["overall"]["map50_95"]["lower"] = -.001
        self.assertTrue(self.assess()["secondary_tradeoff_success"])
        self.metrics["comparisons"]["V1_minus_standard_serial"]["overall_ap_delta"]["map50_95"] = -.000001
        self.assertFalse(self.assess()["secondary_tradeoff_success"])

    def test_fp_tolerance_is_exactly_ten_percent(self):
        point = self.metrics["methods"]["V1"]["operating_points"]["conf0.25_iou0.5"]
        point["total_fp"], point["classes"]["fracture"]["fp"] = 110, 55
        self.assertTrue(self.assess()["passes"])
        point["total_fp"] = 111
        self.assertFalse(self.assess()["passes"])

    def test_fracture_point_decline_or_cost_failure_blocks_success(self):
        self.assertFalse(self.assess(2.001)["passes"])
        self.metrics["comparisons"]["V1_minus_standard_serial"]["per_class_ap_delta"]["fracture"]["ap75"] = -.000001
        self.assertFalse(self.assess()["passes"])

    def test_counting_and_empty_support(self):
        gt = np.array([[0, 0, 10, 10, 0]], np.float32)
        pred = np.array([[0, 0, 10, 10, .8, 0], [20, 20, 30, 30, .7, 0]], np.float32)
        counts = point_counts(gt, pred)
        self.assertEqual(counts, [1, 1, 0, 1, 1, 0])
        self.assertEqual(operating_values(counts, 2)["fp_per_image"], .5)
        self.assertEqual(operating_values([0]*6, 2)["recall"], 0)


if __name__ == "__main__":
    unittest.main()
