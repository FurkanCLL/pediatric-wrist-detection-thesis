"""Corruption fixtures for the independent completion audit; no data or GPU work."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wrist_completion.checks import csv_history, class_history, sampling_counts, amp_history, optimizer_coverage
from wrist_completion.original import write_new
from wrist_prepare.common import CLASSES
from wrist_train.progress import METRIC_KEYS, LOSS_KEYS


def histories():
    rows, classes = [], []
    for epoch, metric in enumerate((.1, .3, .3), 1):
        row = {"epoch": epoch, "time": epoch*10, **{k: metric for k in METRIC_KEYS},
               **{p+k: .2 for p in ("train/", "val/") for k in LOSS_KEYS},
               **{f"lr/pg{i}": .001 for i in range(3)}}
        rows.append(row)
        classes.append({"epoch": epoch, "classes": [{"class_id": i, "class": name,
            **{k: metric for k in ("precision", "recall", "ap50", "ap50_95")}} for i, name in enumerate(CLASSES)]})
    return rows, classes


class CompletionTests(unittest.TestCase):
    def test_ordered_history_and_latest_exact_tie(self):
        rows, classes = histories()
        self.assertEqual(class_history(classes, csv_history(rows, 3)), 3)

    def test_missing_duplicate_fractional_or_unordered_csv_epochs_rejected(self):
        rows, _ = histories()
        for bad in (rows[:-1], [rows[1], rows[0], rows[2]], [rows[0], rows[0], rows[2]],
                    [{**rows[0], "epoch": 1.5}, *rows[1:]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                csv_history(bad, 3)

    def test_nonfinite_missing_metric_and_reverse_time_rejected(self):
        rows, _ = histories()
        for key, value in ((METRIC_KEYS[0], float("nan")), (METRIC_KEYS[1], float("inf")), ("time", -1)):
            bad = deepcopy(rows)
            bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                csv_history(bad, 3)
        del rows[0][METRIC_KEYS[0]]
        with self.assertRaises(ValueError):
            csv_history(rows, 3)

    def test_class_coverage_names_and_macro_rejected(self):
        rows, classes = histories()
        for change in ("missing", "duplicate", "name", "macro", "nan"):
            bad = deepcopy(classes)
            if change == "missing":
                bad[0]["classes"].pop()
            elif change == "duplicate":
                bad[0]["classes"][0]["class_id"] = 1
            elif change == "name":
                bad[0]["classes"][0]["class"] = "wrong"
            else:
                bad[0]["classes"][0]["ap50_95"] = float("nan") if change == "nan" else .4
            with self.subTest(change=change), self.assertRaises(ValueError):
                class_history(bad, rows)

    def test_native_rounding_is_applied_before_selection(self):
        rows, classes = histories()
        # The later raw score is slightly lower, but both native fitness values round to .3.
        for record, score in zip(classes[1:], (.300004, .299996)):
            for c in record["classes"]:
                c["ap50_95"] = score
        self.assertEqual(class_history(classes, rows), 3)

    def test_invalid_consumed_indices_rejected(self):
        for draws in (np.array([0]), np.full(14061, -1), np.zeros(14061, dtype=float)):
            with self.subTest(shape=draws.shape), self.assertRaises(ValueError):
                sampling_counts(draws, np.arange(14061) % 4264, np.zeros(14061, dtype=bool), np.arange(16))

    def test_fixed_parameter_has_no_adam_state_but_trainable_parameters_do(self):
        import torch
        model = torch.nn.Linear(2, 1)
        model.weight.requires_grad_(False)
        state = {"exp_avg": torch.zeros_like(model.bias), "exp_avg_sq": torch.zeros_like(model.bias)}
        optimizer = {"param_groups": [{"params": [0]}, {"params": []}, {"params": [1]}], "state": {1: state}}
        self.assertEqual(optimizer_coverage(model, optimizer), ["weight"])
        for bad in ({**optimizer, "state": {}}, {**optimizer, "state": {0: state, 1: state}}):
            with self.assertRaises(ValueError):
                optimizer_coverage(model, bad)

    def test_sampling_coverage_and_lesion_guard_on_generated_population(self):
        patient = np.arange(14061) % 4264
        patient[-2:] = [0, 1]
        positive = np.arange(14061) < 9307
        lesion = np.isin(patient, np.arange(16))
        remove = np.r_[np.flatnonzero(positive & ~lesion)[-465:], np.flatnonzero(~positive & ~lesion)[-160:]]
        extra = np.r_[np.flatnonzero(positive & ~lesion)[:465], np.flatnonzero(~positive & ~lesion)[:160]]
        draws = np.r_[np.setdiff1d(np.arange(14061), remove), extra]
        result = sampling_counts(draws, patient, positive, np.arange(16))
        self.assertEqual(result["lesion_cohort_views_once"], 66)
        self.assertEqual(result["unique_fracture_images"], 8842)
        bad = draws.copy()
        bad[-1] = 0
        with self.assertRaises(ValueError):
            sampling_counts(bad, patient, positive, np.arange(16))

    def test_amp_replay_growth_skip_and_recovery(self):
        policy = {"amp_initial_scale": 128, "amp_growth_interval": 2, "amp_growth_factor": 2.,
                  "amp_backoff_factor": .5, "overflow_window": 100, "overflow_max_consecutive": 5,
                  "overflow_max_in_window": 10}
        events = [
            {"epoch": 1, "event": "amp", "attempt": 2, "successful_updates": 2, "skipped_updates": 0,
             "updated": True, "scale_before": 128., "scale_after": 256., "recovered": False, "persistent": False},
            {"epoch": 1, "event": "amp", "attempt": 3, "successful_updates": 2, "skipped_updates": 1,
             "updated": False, "scale_before": 256., "scale_after": 128., "recovered": False, "persistent": False},
            {"epoch": 1, "event": "amp", "attempt": 4, "successful_updates": 3, "skipped_updates": 1,
             "updated": True, "scale_before": 128., "scale_after": 128., "recovered": True, "persistent": False}]
        receipts = [{"epoch": 1, "scale_start": 128, "scale_end": 128, "optimizer_epoch": {"attempts": 4}}]
        states, summary = amp_history(events, receipts, policy)
        self.assertEqual(states[-1]["accounting"]["successful"], 3)
        self.assertEqual(summary["overflow_skips"], 1)
        for field, value in (("scale_after", 64), ("successful_updates", 99), ("recovered", False), ("persistent", True)):
            bad = deepcopy(events)
            bad[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                amp_history(bad, receipts, policy)
        with self.assertRaises(ValueError):
            amp_history(events[:-1], receipts, policy)

    def test_previous_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"
            write_new(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
