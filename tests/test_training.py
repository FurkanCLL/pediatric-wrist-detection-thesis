"""Focused safeguards for the diagnostic training workflow."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wrist_train.common import candidate, select_training, write_data_yaml


class TrainingInputs(unittest.TestCase):
    def test_subset_is_train_only_and_order_independent(self):
        records = [{"filestem": str(i), "split": "train" if i < 30 else "test"} for i in range(40)]
        first = select_training(records, 15)
        self.assertEqual(first, select_training(list(reversed(records)), 15))
        self.assertTrue(all(r["split"] == "train" for r in first))
        self.assertEqual(len({r["filestem"] for r in first}), 15)

    def test_diagnostic_yaml_has_no_test_path(self):
        import yaml
        with tempfile.TemporaryDirectory() as name, patch("wrist_train.common.verify_frozen_data", return_value=([], {})):
            path, _ = write_data_yaml(Path(name), subset=False)
            data = yaml.safe_load(path.read_text())
            self.assertNotIn("test", data)
            self.assertEqual(data["nc"], 8)
            self.assertEqual(data["channels"], 3)

    def test_candidate_keeps_optional_augmentation_disabled(self):
        args = candidate()["training"]
        for key in ("mosaic", "mixup", "cutmix", "copy_paste", "fliplr", "flipud", "shear", "perspective", "cls_pw", "hsv_h", "hsv_s"):
            self.assertEqual(args[key], 0)
        self.assertEqual(args["augmentations"], [])
        self.assertFalse(args["cls_remap"])
        self.assertEqual(args["seed"], 42)
        self.assertEqual(args["optimizer"], "AdamW")

    def test_gradient_accumulation_is_explicit(self):
        args = candidate()["training"]
        self.assertEqual(args["nbs"], 24)
        self.assertEqual(max(round(args["nbs"] / 6), 1), 4)
        self.assertEqual(args["batch"], 6)


class RuntimeChecks(unittest.TestCase):
    def test_update_budget_accounts_for_warmup(self):
        from wrist_train.budget import optimizer_budget
        smoke = optimizer_budget(14061, 6, 24, 1, 1.0)
        official = optimizer_budget(14061, 6, 24, 30, 1.0)
        self.assertEqual(smoke["warmup_iterations"], 0)
        self.assertEqual(smoke["total_optimizer_updates"], 586)
        self.assertGreater(official["updates_per_epoch"][0], 586)
        self.assertEqual(official["updates_per_epoch"][1:], [586] * 29)
        with self.assertRaises(ValueError):
            optimizer_budget(0, 6, 24, 1, 0)

    def test_actual_grayscale_intensity_transform(self):
        from wrist_train.feasibility import augmentation_probe
        result = augmentation_probe()
        self.assertGreater(result["changed_trials"], 0)
        self.assertTrue(result["equal_channels_preserved"])

    def test_nested_nonfinite_predictions_are_detected(self):
        import torch
        from wrist_train.diagnostics import finite_tensors
        self.assertTrue(finite_tensors([torch.ones(2), {"loss": torch.tensor(0.)}]))
        self.assertFalse(finite_tensors((torch.ones(2), {"prediction": torch.tensor(float("nan"))})))
        self.assertFalse(finite_tensors(torch.tensor(float("inf"))))


if __name__ == "__main__":
    unittest.main()
