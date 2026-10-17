"""Rarity integration fixtures; real worker consumption without medical images or fitting."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wrist_train.common import ROOT, configure_runtime

configure_runtime()
import numpy as np
import torch
from PIL import Image
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer

from wrist_prepare.common import CLASSES
from wrist_train.budget import optimizer_budget
from wrist_train.official import OfficialTrainer
from wrist_train.protocol import load_protocol, canonical_hash
from wrist_train.launch import prepare_directory
from wrist_rarity.data import EpochSampler, FiniteEpochLoader, IndexedDataset, label_signature
from wrist_rarity.launch import main, check_condition_checkpoint, training_gate
from wrist_rarity.policy import RarityPlans, plan_hash
from wrist_rarity.protocol import condition_definition, load_condition, source_files_and_labels
from wrist_rarity.trainer import EpochAudit, RarityTrainer


# Native YOLO transforms on generated patterns exercise the same label/collation code as training.
def synthetic_dataset(root):
    image_dir, label_dir = root/"images/train", root/"labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for i in range(16):
        yy, xx = np.indices((80, 64))
        pattern = np.stack([(xx*3+i*7) % 256, (yy*2+i*11) % 256, (xx+yy+i*13) % 256], axis=-1).astype(np.uint8)
        Image.fromarray(pattern).save(image_dir/f"sample_{i:02}.png")
        (label_dir/f"sample_{i:02}.txt").write_text(f"{i % 8} 0.5 0.5 0.4 0.3\n")
    args = json.loads((ROOT/"configs/training_official.json").read_text())["training"]
    args.update(imgsz=64, batch=4, device="cpu")
    native = build_yolo_dataset(get_cfg(overrides=args), str(image_dir), 4,
                                {"nc": 8, "names": dict(enumerate(CLASSES)), "channels": 3},
                                mode="train", rect=False, stride=32)
    files = [str((image_dir/f"sample_{i:02}.png").resolve()) for i in range(16)]
    signatures = [label_signature([[i % 8, .5, .5, .4, .3]]) for i in range(16)]
    return native, files, signatures


class RarityWorkerTests(unittest.TestCase):
    def test_two_workers_replay_three_native_epochs_across_two_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, files, signatures = synthetic_dataset(root)
            dataset = IndexedDataset(native, files, signatures)
            # Two repeated sources and two omitted views give different ordered plans each epoch.
            plans = [np.array(list(range(14))+[0, 1], dtype=np.int32),
                     np.array(list(range(2, 16))+[2, 3], dtype=np.int32),
                     np.array(list(range(4, 16))+[0, 1, 4, 5], dtype=np.int32)]
            launches = []
            for launch in range(2):
                loader = FiniteEpochLoader(dataset, batch_size=4, workers=2, pin_memory=False)
                all_receipts = []
                for epoch, plan in enumerate(plans):
                    loader.select_epoch(epoch, plan)
                    generator_state = loader.generator.get_state().clone()
                    discarded = enumerate(loader)
                    self.assertIsNone(loader.active_iterator)
                    self.assertTrue(torch.equal(generator_state, loader.generator.get_state()))
                    audit = EpochAudit(epoch, plan, files, signatures, root/f"audit_{launch}", loader.worker_base_seed)
                    for batch in loader:
                        audit.consume(batch["rarity_receipt"], batch["im_file"])
                        # Every output box retains its source class after native augmentation/filtering.
                        for item, index in enumerate([r[2] for r in batch["rarity_receipt"]]):
                            labels = batch["cls"][batch["batch_idx"] == item]
                            self.assertTrue(torch.all(labels == index % 8))
                        all_receipts.extend(batch["rarity_receipt"])
                    summary = audit.finish()
                    self.assertEqual(summary["different_augmented_pairs"], 2)
                    self.assertEqual(summary["draws"], 16)
                    self.assertIsNone(loader.active_iterator)
                    self.assertEqual([r[0] for r in all_receipts[-16:]], [epoch]*16)
                    del discarded
                loader.close()
                launches.append(all_receipts)
            self.assertEqual(launches[0], launches[1])
            self.assertNotEqual(launches[0][0][4], launches[0][16][4])

    def test_native_image_order_and_cached_label_corruption_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            native, files, signatures = synthetic_dataset(Path(temporary))
            with self.assertRaisesRegex(ValueError, "image order"):
                IndexedDataset(native, list(reversed(files)), signatures)
            native.labels[0]["cls"][0, 0] = 7
            with self.assertRaisesRegex(ValueError, "annotations"):
                IndexedDataset(native, files, signatures)

    def test_worker_rechecks_labels_and_wrong_source_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            native, files, signatures = synthetic_dataset(Path(temporary))
            wrapped = IndexedDataset(native, files, signatures)
            native.labels[0]["bboxes"][0, 0] = .1
            with self.assertRaisesRegex(ValueError, "worker"):
                wrapped[0, 0, 0, False]


class RarityIntegrationTests(unittest.TestCase):
    def test_invalid_plan_lengths_types_indices_and_epochs_stop(self):
        sampler = EpochSampler(4)
        for epoch, plan in [(0, [0, 1]), (0, [0, 1, 2, 4]), (0, [0., 1., 2., 3.]), (-1, [0, 1, 2, 3])]:
            with self.assertRaises(ValueError):
                sampler.set_plan(epoch, plan)
        with self.assertRaises(RuntimeError):
            list(sampler)
        sampler.set_plan(3, np.array([3, 1, 1, 0]))
        self.assertEqual(list(sampler), [(3, 0, 3, False), (3, 1, 1, True), (3, 2, 1, True), (3, 3, 0, False)])

    def test_stale_epoch_wrong_order_and_wrong_label_receipts_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = [str((Path(temporary)/str(i)).resolve()) for i in range(4)]
            signatures = [str(i) for i in range(4)]
            for number, receipt in enumerate([(0, 0, 0, 0, 10, "0", None),
                                               (1, 0, 1, 0, 10, "1", None),
                                               (1, 0, 0, 0, 10, "bad", None),
                                               (1, 0, 0, 0, 99, "0", None)]):
                audit = EpochAudit(1, [0, 1, 2, 3], files, signatures, Path(temporary)/str(number), 10)
                with self.assertRaises(ValueError):
                    audit.consume([receipt], [files[receipt[2]]])

    def test_incomplete_consumption_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = EpochAudit(1, [0, 1], [], [], Path(temporary), 10)
            with self.assertRaises(ValueError):
                audit.finish()

    def test_model_optimizer_scaler_validation_and_native_training_loop_are_inherited(self):
        for name in ("optimizer_step", "_setup_train", "get_validator", "_load_checkpoint_state", "_do_train"):
            self.assertIs(getattr(RarityTrainer, name), getattr(OfficialTrainer, name))
        with patch.object(OfficialTrainer, "get_dataloader", return_value="native-validation") as native:
            trainer = object.__new__(RarityTrainer)
            self.assertEqual(trainer.get_dataloader("validation", 12, -1, "val"), "native-validation")
            native.assert_called_once_with("validation", 12, -1, "val")

    def test_standard_budget_and_warmup_accumulation_are_preserved(self):
        self.assertEqual(optimizer_budget(14061, 6, 24, 80, 1)["total_optimizer_updates"], 47433)
        base, _ = load_protocol()
        self.assertEqual((base["training"]["batch"], base["training"]["nbs"], base["runtime"]["amp_initial_scale"]), (6, 24, 128))

    def test_preflight_and_unconfirmed_start_never_train(self):
        with patch("wrist_rarity.launch.train") as train:
            with self.assertRaises(SystemExit):
                main(["start"])
            with self.assertRaises(SystemExit):
                main(["resume"])
            with patch("wrist_rarity.launch.preflight", return_value=({}, {}, {"training_started": False})), patch("wrist_rarity.launch.training_gate"):
                main(["preflight"])
            train.assert_not_called()

    def test_checkpoint_rejects_baseline_diagnostic_and_other_source_identity(self):
        identity = {"rarity_condition_sha256": "condition", "rarity_source_sha256": {"file": "sha"},
                    "diagnostic_only": False, "condition": "patient_aware_rarity_only"}
        payload = {"epoch": 4, "official": {**identity, "rarity_completed_epoch": {"epoch": 5, "consumed_indices_exact_match": True}}}
        check_condition_checkpoint(payload, identity)
        for key in identity:
            bad = deepcopy(payload)
            bad["official"][key] = "wrong"
            with self.assertRaises(ValueError):
                check_condition_checkpoint(bad, identity)
        payload["official"]["rarity_completed_epoch"]["epoch"] = 4
        with self.assertRaises(ValueError):
            check_condition_checkpoint(payload, identity)

    def test_output_claim_does_not_overwrite_an_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"run"
            prepare_directory(path, {"identity": "fixture"}, False)
            with self.assertRaises(FileExistsError):
                prepare_directory(path, {"identity": "fixture"}, False)

    def test_diagnostic_changes_only_epoch_budget_in_training_arguments(self):
        common, original, definition = condition_definition()
        with patch("pathlib.Path.read_text", return_value=json.dumps(definition)):
            with patch("wrist_rarity.protocol.condition_definition", return_value=(deepcopy(common), deepcopy(original), definition)):
                smoke, identity = load_condition(True)
        self.assertEqual(identity["training_epochs"], 2)
        self.assertEqual(identity["protocol_sha256"], original["protocol_sha256"])
        differences = {k for k in common["training"] if common["training"][k] != smoke["training"][k]}
        self.assertEqual(differences, {"epochs"})
        self.assertIs(identity["diagnostic_only"], True)

    def test_plan_hash_is_stable_across_integer_storage_width(self):
        self.assertEqual(plan_hash(np.arange(20, dtype=np.int64)), plan_hash(np.arange(20, dtype=np.int32)))

    def test_nontraining_metadata_is_rejected_before_any_file_is_opened(self):
        with self.assertRaises(ValueError):
            source_files_and_labels([{"split": "val", "filestem": "forbidden"}])
        with self.assertRaises(ValueError):
            source_files_and_labels([{"split": "train", "filestem": "../forbidden"}])

    def test_epoch_transition_cannot_replace_an_active_iterator(self):
        class Fixture:
            collate_fn = staticmethod(lambda values: values)
            def __len__(self):
                return 4
            def __getitem__(self, request):
                return request
        loader = FiniteEpochLoader(Fixture(), batch_size=2, workers=0, pin_memory=False)
        loader.select_epoch(0, np.arange(4))
        iterator = iter(loader)
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, "still active"):
            loader.select_epoch(1, np.arange(4))
        iterator.close()
        loader.select_epoch(1, np.arange(4))
        self.assertEqual([r[0] for batch in loader for r in batch], [1]*4)

    def test_source_change_after_condition_lock_is_rejected(self):
        common, original, definition = condition_definition()
        modified = deepcopy(definition)
        modified["source_sha256"]["src/wrist_rarity/data.py"] = "different"
        with patch("wrist_rarity.protocol.condition_definition", return_value=(common, original, modified)), patch("pathlib.Path.read_text", return_value=json.dumps(definition)):
            with self.assertRaisesRegex(ValueError, "changed after verification"):
                load_condition()


if __name__ == "__main__":
    unittest.main()
