"""Synthetic CPU fixtures for official-run safeguards; no YOLO epochs are started."""

from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from wrist_train.common import ROOT, configure_runtime

configure_runtime()
import torch
from ultralytics.utils.torch_utils import ModelEMA

from wrist_prepare.common import CLASSES
from wrist_train.amp import UpdateMonitor, controlled_step, guard_training_loss
from wrist_train.budget import optimizer_budget
from wrist_train.checkpoints import checked_bytes, check_payload, load_recovery, model_snapshot, publish_checkpoint, sidecar
from wrist_train.launch import checkpoint_path, main, prepare_directory, run_lock
from wrist_train.official import OfficialTrainer, restore_results_csv
from wrist_train.progress import METRIC_KEYS, display_epoch, epoch_record, per_class_record
from wrist_train.protocol import canonical_hash, environment_check, load_protocol, output_check, validate_protocol, verify_inputs


def protocol():
    return json.loads((ROOT / "configs/training_official.json").read_text())


def fixture(growth_interval=2000, fused=False):
    torch.manual_seed(42)
    model = torch.nn.Conv2d(3, 8, 1)
    model.names = dict(enumerate(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, fused=fused)
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=growth_interval)
    ema = ModelEMA(model)
    monitor = UpdateMonitor(protocol()["runtime"])
    return model, optimizer, scaler, ema, monitor


def step(objects, overflow=False):
    model, optimizer, scaler, ema, monitor = objects
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    scaler.scale(loss).backward()
    if overflow:
        next(model.parameters()).grad.fill_(float("inf"))
    events = []
    result = controlled_step(model, optimizer, scaler, ema, monitor, events.append)
    return result, events


IDENTITY = {"protocol_sha256": "protocol", "split_manifest_sha256": "split", "pretrained_sha256": "weights",
            "implementation_source_sha256": {}, "implementation_revision": "synthetic", "training_epochs": 80}


def checkpoint_fixture():
    objects = fixture()
    step(objects)
    step(objects, True)
    model, optimizer, scaler, ema, monitor = objects
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / protocol()["training"]["epochs"])
    payload = {"epoch": 0, "model": model_snapshot(model), "ema": model_snapshot(ema.ema),
        "best_ema": model_snapshot(ema.ema), "best_fitness": 0.2, "updates": ema.updates,
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
        "accounting": monitor.state_dict(), "train_args": protocol()["training"],
        "train_results": {"epoch": [1], "time": [1.0]}, "stopper": {}, "rng": None,
        "official": {**IDENTITY, "format_version": 1, "resumable": True, "best_epoch": 0, "pending_microbatches": 1}}
    return payload


class AmpRecovery(unittest.TestCase):
    def test_fused_adamw_does_not_count_a_skipped_kernel_update(self):
        objects = fixture(fused=True)
        calls = []
        handle = objects[1].register_step_post_hook(lambda *args: calls.append(True))
        step(objects)
        event, _ = step(objects, True)
        handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertFalse(event["updated"])
        self.assertEqual(objects[4].successful, 1)
        self.assertEqual(objects[3].updates, 1)

    def test_backoff_recovery_growth_and_ema_accounting(self):
        objects = fixture(growth_interval=2)
        first, _ = step(objects)
        self.assertTrue(first["updated"])
        original = next(objects[0].parameters()).detach().clone()
        skipped, events = step(objects, True)
        self.assertFalse(skipped["updated"])
        self.assertEqual(skipped["scale_after"], 64)
        self.assertTrue(torch.equal(original, next(objects[0].parameters())))
        self.assertEqual(objects[3].updates, 1)
        recovered, _ = step(objects)
        self.assertTrue(recovered["recovered"])
        grown, events = step(objects)
        self.assertEqual(grown["scale_after"], 128)
        self.assertEqual(objects[3].updates, objects[4].successful)
        self.assertEqual((objects[4].attempts, objects[4].successful, objects[4].skipped), (4, 3, 1))

    def test_persistent_consecutive_overflow_stops(self):
        objects = fixture()
        for _ in range(4):
            step(objects, True)
        with self.assertRaisesRegex(FloatingPointError, "Persistent"):
            step(objects, True)
        self.assertEqual(objects[4].skipped, 5)
        self.assertEqual(objects[3].updates, 0)

    def test_rolling_window_limit_survives_checkpoint_state(self):
        monitor = UpdateMonitor(protocol()["runtime"])
        for _ in range(9):
            monitor.record(False, 128, 64)
            monitor.record(True, 64, 64)
        restored = UpdateMonitor(protocol()["runtime"], monitor.state_dict())
        self.assertTrue(restored.record(False, 64, 32)["persistent"])

    def test_nonfinite_loss_stops_before_backward(self):
        model = torch.nn.Module()
        with self.assertRaisesRegex(FloatingPointError, "backward was not started"):
            guard_training_loss(model, ({},), (torch.tensor(float("nan")), {"box_loss": torch.tensor(1.)}))

    def test_corrupt_model_is_not_saved(self):
        model = fixture()[0]
        next(model.parameters()).data.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            model_snapshot(model)


class RecoveryFiles(unittest.TestCase):
    def test_periodic_saves_through_epoch_80_are_full_precision_and_not_stripped(self):
        model, optimizer, scaler, ema, monitor = fixture()
        step((model, optimizer, scaler, ema, monitor))
        trainer = OfficialTrainer.__new__(OfficialTrainer)
        trainer.model, trainer.optimizer, trainer.scaler, trainer.ema, trainer.monitor = model, optimizer, scaler, ema, monitor
        trainer.identity = IDENTITY
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / protocol()["training"]["epochs"])
        trainer.epochs = protocol()["training"]["epochs"]
        trainer.best_fitness = trainer.fitness = 0.2
        trainer.metrics = dict(zip(METRIC_KEYS, (0.1, 0.2, 0.3, 0.2)))
        trainer._best_seen_score = float('-inf')
        trainer._best_model = None
        trainer.stopper = SimpleNamespace(best_fitness=0.2, best_epoch=10)
        trainer.args = SimpleNamespace(**protocol()['training'])
        trainer.current_per_class = []
        trainer.pending_microbatches = 1
        with tempfile.TemporaryDirectory() as temporary:
            trainer.wdir = Path(temporary)/'weights'
            trainer.last, trainer.best = trainer.wdir/'last.pt', trainer.wdir/'best.pt'
            # Exercise the real save hook with synthetic completed-epoch positions, without a training loop.
            for completed in range(10, trainer.epochs + 1, 10):
                trainer.epoch = trainer.scheduler.last_epoch = completed - 1
                trainer.read_results_csv = lambda: {'epoch': list(range(1, completed + 1))}
                trainer.save_model()
                path = trainer.wdir / f'epoch{completed:03}.pt'
                self.assertTrue(path.exists())
                saved = torch.load(io.BytesIO(checked_bytes(path)), map_location='cpu', weights_only=False)
                check_payload(saved, IDENTITY, require_resume=False)
                self.assertIsNotNone(saved['optimizer'])
                self.assertEqual(saved['epoch'], completed - 1)
                if completed < trainer.epochs:
                    self.assertIsNotNone(load_recovery(path, IDENTITY)['optimizer'])
                else:
                    with self.assertRaisesRegex(ValueError, 'unfinished epoch'):
                        load_recovery(path, IDENTITY)
            self.assertFalse((trainer.wdir/'epoch009.pt').exists())
            with patch.object(trainer, '_event') as event:
                trainer.final_eval()
                event.assert_called_once()
                self.assertEqual(event.call_args.args[0]['epochs'], 80)
            self.assertIsNotNone(torch.load(io.BytesIO(checked_bytes(trainer.last)), weights_only=False)['optimizer'])

    def test_resume_boundaries_and_stale_budget_identity(self):
        payload = checkpoint_fixture()
        for completed in (60, 70, 79, 80):
            with self.subTest(completed=completed):
                payload['epoch'] = payload['scheduler']['last_epoch'] = completed - 1
                payload['train_results'] = {'epoch': list(range(1, completed + 1))}
                trainer = OfficialTrainer.__new__(OfficialTrainer)
                trainer.epochs, trainer.resume = 80, True
                trainer.args = SimpleNamespace(model='synthetic')
                trainer.model = SimpleNamespace(model=[None])
                with patch.object(trainer, '_load_checkpoint_state') as restore:
                    if completed < 80:
                        check_payload(payload, IDENTITY)
                        trainer.resume_training(payload)
                        self.assertEqual(trainer.start_epoch, completed)
                        restore.assert_called_once_with(payload)
                    else:
                        with self.assertRaisesRegex(ValueError, '80-epoch budget'):
                            check_payload(payload, IDENTITY)
                        with self.assertRaises(AssertionError):
                            trainer.resume_training(payload)
                        restore.assert_not_called()
        stale = deepcopy(payload)
        stale['official']['training_epochs'] = 60
        with self.assertRaisesRegex(ValueError, 'training_epochs'):
            check_payload(stale, IDENTITY)
        stale = deepcopy(payload)
        stale['train_args']['epochs'] = 60
        with self.assertRaisesRegex(ValueError, 'arguments disagree'):
            check_payload(stale, IDENTITY)

    def test_roundtrip_retains_fp32_state_and_scale(self):
        payload = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            publish_checkpoint(path, payload)
            loaded = load_recovery(path, IDENTITY)
            self.assertEqual(loaded["scaler"]["scale"], 64)
            self.assertTrue(all(value.dtype == torch.float32 for value in loaded["model"].parameters()))
            state = next(iter(loaded["optimizer"]["state"].values()))
            self.assertEqual(state["exp_avg"].dtype, torch.float32)
            self.assertEqual(loaded["accounting"]["skipped"], 1)

    def test_resume_restores_native_optimizer_ema_schedule_and_scaler(self):
        payload = checkpoint_fixture()
        objects = fixture()
        trainer = OfficialTrainer.__new__(OfficialTrainer)
        trainer.model, trainer.optimizer, trainer.scaler, trainer.ema, _ = objects
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda epoch: 1 - epoch / protocol()["training"]["epochs"])
        trainer.stopper = SimpleNamespace()
        trainer.protocol = protocol()
        with tempfile.TemporaryDirectory() as temporary:
            trainer.directory = Path(temporary)
            trainer._load_checkpoint_state(deepcopy(payload))
            self.assertEqual(trainer.scaler.get_scale(), 64)
            self.assertEqual(trainer.scheduler.state_dict(), payload["scheduler"])
            self.assertEqual(trainer.monitor.state_dict(), payload["accounting"])
            self.assertEqual(trainer.ema.updates, payload["updates"])
            for key, value in trainer.ema.ema.state_dict().items():
                self.assertTrue(torch.equal(value, payload["ema"].state_dict()[key]))
            state = next(iter(trainer.optimizer.state.values()))
            saved = next(iter(payload["optimizer"]["state"].values()))
            self.assertTrue(torch.equal(state["exp_avg"], saved["exp_avg"]))

    def test_setup_preserves_restored_scale_and_uses_live_weights(self):
        payload = checkpoint_fixture()
        trainer = OfficialTrainer.__new__(OfficialTrainer)
        trainer.protocol = protocol()
        trainer.resume = True
        trainer._resume_payload = deepcopy(payload)
        trainer.amp = True
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128)
        trainer.scaler.load_state_dict(payload["scaler"])
        trainer.setup_model()
        for key, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, payload["model"].state_dict()[key]))
        with patch("wrist_train.official.DetectionTrainer._setup_train"):
            trainer._setup_train()
        self.assertEqual(trainer.scaler.get_scale(), 64)

    def test_incomplete_pair_is_rejected_and_predecessor_survives(self):
        import wrist_train.checkpoints as module
        payload = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            publish_checkpoint(path, payload)
            old = checked_bytes(path)
            replacement = deepcopy(payload)
            replacement["epoch"] = 1
            replacement["scheduler"]["last_epoch"] = 1
            original = module.atomic_bytes
            def fail_sidecar(destination, content):
                if destination == sidecar(path):
                    raise OSError("simulated interruption between renames")
                return original(destination, content)
            with patch.object(module, "atomic_bytes", side_effect=fail_sidecar), self.assertRaises(OSError):
                publish_checkpoint(path, replacement, backup=True)
            with self.assertRaises(ValueError):
                checked_bytes(path)
            self.assertEqual(checked_bytes(path.with_name("last.previous.pt")), old)

    def test_wrong_identity_corrupt_state_and_finished_run_are_rejected(self):
        payload = checkpoint_fixture()
        with self.assertRaises(ValueError):
            check_payload(payload, {**IDENTITY, "protocol_sha256": "different"})
        corrupt = deepcopy(payload)
        next(iter(corrupt["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            check_payload(corrupt, IDENTITY)
        payload["epoch"] = protocol()["training"]["epochs"] - 1
        with self.assertRaises(ValueError):
            check_payload(payload, IDENTITY)

    def test_periodic_files_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "epoch010.pt"
            publish_checkpoint(path, checkpoint_fixture())
            with self.assertRaises(FileExistsError):
                publish_checkpoint(path, checkpoint_fixture())


class LaunchAndMetrics(unittest.TestCase):
    def test_only_epoch_budget_changes_and_nominal_attempts_match(self):
        approved = protocol()
        historical = json.loads((ROOT/'docs/evidence/training_protocol/history_60_epochs.json').read_text())
        previous = historical['configuration']
        self.assertEqual(previous['training']['epochs'], 60)
        changed = deepcopy(approved)
        changed['training']['epochs'] = 60
        self.assertEqual(changed, previous)
        self.assertEqual(approved['training']['epochs'], 80)
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        args = approved['training']
        budget = optimizer_budget(approved['dataset']['training_draws_per_epoch'], args['batch'],
                                  args['nbs'], args['epochs'], args['warmup_epochs'])
        self.assertEqual(budget['updates_per_epoch'], [1139] + [586] * 79)
        self.assertEqual(budget['total_optimizer_updates'], 47433)

    def test_trainer_guard_and_native_cosine_use_frozen_80_epoch_budget(self):
        trainer = OfficialTrainer.__new__(OfficialTrainer)
        trainer.protocol = protocol()
        trainer.args = SimpleNamespace(**trainer.protocol['training'])
        trainer.epochs = trainer.args.epochs
        trainer.model, trainer.optimizer, trainer.scaler, _, trainer.monitor = fixture()
        trainer._setup_scheduler()
        self.assertAlmostEqual(trainer.lf(0), 1.0)
        self.assertAlmostEqual(trainer.lf(40), 0.505)
        self.assertAlmostEqual(trainer.lf(80), 0.01)
        self.assertGreater(trainer.lf(60), 0.01)
        trainer.device = torch.device('cuda:0')
        trainer.batch_size, trainer.accumulate = 6, 4
        trainer.train_loader, trainer.test_loader = MagicMock(), MagicMock()
        trainer.train_loader.dataset, trainer.test_loader.dataset = range(14061), range(3088)
        trainer.train_loader.__len__.return_value = 2344
        trainer.data, trainer.start_epoch = {}, 0
        with patch.object(trainer, '_event'):
            trainer._start_training(trainer)
            self.assertEqual(trainer.warmup_iterations, 2344)
            trainer.epochs = 60
            with self.assertRaisesRegex(ValueError, 'budget'):
                trainer._start_training(trainer)

    def test_modified_lock_config_or_implementation_is_rejected(self):
        import wrist_train.protocol as module
        with tempfile.TemporaryDirectory() as temporary:
            config=Path(temporary)/'official.json'; lock=Path(temporary)/'lock.json'
            config.write_text(json.dumps(protocol()))
            lock.write_text(json.dumps({'protocol_sha256':canonical_hash(protocol()),'implementation_source_sha256':{'code':'good'},'framework_source_sha256':module.framework_hashes(),'training_epochs':80}))
            with patch.object(module,'CONFIG',config),patch.object(module,'LOCK',lock),patch.object(module,'source_hashes',return_value={'code':'good'}):
                load_protocol()
                changed=protocol(); changed['status']='changed'; config.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): load_protocol()
                config.write_text(json.dumps(protocol()))
            with patch.object(module,'CONFIG',config),patch.object(module,'LOCK',lock),patch.object(module,'source_hashes',return_value={'code':'changed'}):
                with self.assertRaises(ValueError): load_protocol()

    def test_insufficient_vram_and_changed_dataset_stop_preflight(self):
        with patch('wrist_train.common.require_cuda',return_value=torch.device('cpu')),patch('torch.cuda.get_device_name',return_value=protocol()['environment']['gpu_name']),patch('torch.cuda.empty_cache'),patch('torch.cuda.mem_get_info',return_value=(2**30,6*2**30)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient free VRAM'):
                environment_check(protocol())
        with patch('wrist_train.protocol.verify_frozen_data',return_value=([],{'split_manifest_sha256':'changed'})):
            with self.assertRaisesRegex(ValueError,'Dataset split'):
                verify_inputs(protocol(),'journal')

    def test_explicit_start_and_resume_are_required(self):
        with patch("wrist_train.launch.train_manually") as train, patch("sys.stderr", new=io.StringIO()):
            for action in ("start", "resume"):
                with self.assertRaises(SystemExit):
                    main([action])
            train.assert_not_called()

    def test_preflight_cannot_call_training(self):
        with patch("wrist_train.launch.preflight", return_value=({}, {}, {"status": "passed"})), patch("wrist_train.launch.train_manually") as train, patch("sys.stdout", new=io.StringIO()):
            main(["preflight"])
            train.assert_not_called()

    def test_freeze_hash_and_approved_values(self):
        approved = protocol()
        validate_protocol(approved)
        self.assertEqual(canonical_hash(approved), canonical_hash(json.loads(json.dumps(approved))))
        changed = deepcopy(approved)
        changed["training"]["epochs"] = 30
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        changed = deepcopy(approved)
        changed["training"]["mosaic"] = 0.1
        with self.assertRaises(ValueError):
            validate_protocol(changed)
        self.assertEqual(json.loads((ROOT/"configs/training_candidate.json").read_text())["training"]["epochs"], 30)

    def test_output_protection_resume_segments_and_active_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "baseline"
            self.assertGreater(output_check(root), 0)
            prepare_directory(root, IDENTITY, False)
            with self.assertRaises(FileExistsError):
                output_check(root)
            segment = prepare_directory(root, IDENTITY, True)
            self.assertNotEqual(segment, root)
            with run_lock(root), patch("wrist_train.launch.process_alive", return_value=True):
                with self.assertRaises(RuntimeError):
                    with run_lock(root):
                        pass

    def test_checkpoint_path_cannot_escape_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                checkpoint_path(root, "../outside.pt")

    def test_epoch_print_uses_actual_metrics_and_flushes(self):
        training = {key: torch.tensor(1.0) for key in ("box_loss", "cls_loss", "dfl_loss")}
        metrics = dict(zip(METRIC_KEYS, (0.1, 0.2, 0.3, 0.4)))
        metrics.update({"val/" + key: 2.0 for key in training})
        record = epoch_record(3, protocol()["training"]["epochs"], training, metrics, 10.0, 2.0, 0.4, {"successful": 7, "skipped": 1})
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.print") as output:
            display_epoch(Path(temporary), record)
            self.assertTrue(output.call_args.kwargs["flush"])
            self.assertIn("Epoch 3/80", output.call_args.args[0])
            saved = json.loads((Path(temporary)/"epochs.jsonl").read_text())
            self.assertEqual(saved["map50_95"], 0.4)
        invalid = dict(metrics)
        invalid[METRIC_KEYS[0]] = float("nan")
        with self.assertRaises(FloatingPointError):
            epoch_record(3, protocol()["training"]["epochs"], training, invalid, 10., 2., 0.4, {})
        metrics.pop("val/box_loss")
        with self.assertRaises(ValueError):
            epoch_record(3, protocol()["training"]["epochs"], training, metrics, 10., 2., 0.4, {})

    def test_per_class_and_csv_history_are_preserved(self):
        metrics = SimpleNamespace(ap_class_index=list(range(8)), class_result=lambda i: (0.1, 0.2, 0.3, 0.4))
        rows = per_class_record(metrics, dict(enumerate(CLASSES)))
        self.assertEqual(rows[0]["class"], "fracture")
        self.assertEqual(rows[0]["ap50_95"], 0.4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"results.csv"
            restore_results_csv(path, {"epoch": [1, 2], "time": [1., 2.]})
            self.assertEqual(path.read_text().splitlines(), ["epoch,time", "1,1.0", "2,2.0"])
            with self.assertRaises(FileExistsError):
                restore_results_csv(path, {"epoch": [1]})


if __name__ == "__main__":
    unittest.main()
