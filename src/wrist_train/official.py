"""Official trainer hooks, full-state checkpoints and per-epoch reporting."""

import csv
import json
import random
import time

from .common import configure_runtime

configure_runtime()
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer

from .amp import UpdateMonitor, controlled_step, guard_training_loss, state_is_finite
from .checkpoints import cpu_state, model_snapshot, publish_checkpoint
from .diagnostics import CheckedTrainer, memory_peaks
from .progress import append_json, display_epoch, epoch_record, per_class_record


class OfficialTrainer(CheckedTrainer):
    def __init__(self, protocol, identity, directory, overrides, checkpoint=None):
        self.protocol = protocol
        self.identity = identity
        self.directory = directory
        self._resume_payload = checkpoint
        self.monitor = UpdateMonitor(protocol["runtime"])
        self._best_model = None
        self._best_seen_score = float("-inf")
        self.best_epoch = None
        self.pending_microbatches = 0
        super().__init__(overrides=overrides)
        if self.save_dir.resolve() != directory.resolve():
            raise RuntimeError("Framework chose a different official output directory")
        self.add_callback("on_train_epoch_start", self._begin_epoch)
        self.add_callback("on_fit_epoch_end", self._report_epoch)

    def check_resume(self, overrides):
        # The launcher already validates the checkpoint and rebuilds all args from the frozen protocol.
        self.resume = self._resume_payload is not None
        if bool(self.args.resume) != self.resume:
            raise ValueError("Resume intent and supplied full-state checkpoint disagree")

    def setup_model(self):
        if not self.resume:
            return super().setup_model()
        # Native setup chooses EMA weights. Recovery must continue the saved live model instead.
        self.model = model_snapshot(self._resume_payload["model"])
        return self._resume_payload

    def _setup_train(self):
        # Bypass only CheckedTrainer's diagnostic scaler reset, keeping native setup/resume intact.
        DetectionTrainer._setup_train(self)
        policy = self.protocol["runtime"]
        if not self.amp:
            raise RuntimeError("Official AMP was unexpectedly disabled")
        if not self.resume:
            self.scaler = torch.amp.GradScaler("cuda", init_scale=policy["amp_initial_scale"],
                growth_factor=policy["amp_growth_factor"], backoff_factor=policy["amp_backoff_factor"],
                growth_interval=policy["amp_growth_interval"])
        elif self.scaler.get_scale() != self._resume_payload["scaler"]["scale"]:
            raise RuntimeError("Restored AMP scale was changed during setup")
        self.model.register_forward_hook(guard_training_loss)
        if self.resume:
            self._restore_rng(self._resume_payload.get("rng"))
        self._resume_payload = None

    def _load_checkpoint_state(self, checkpoint):
        super()._load_checkpoint_state(checkpoint)
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.monitor = UpdateMonitor(self.protocol["runtime"], checkpoint["accounting"])
        self._best_model = checkpoint["best_ema"]
        self._best_seen_score = checkpoint["best_fitness"]
        self.best_epoch = checkpoint["official"]["best_epoch"]
        if self.ema.updates != self.monitor.successful:
            raise ValueError("Restored EMA count does not match optimizer accounting")
        discarded = checkpoint["official"]["pending_microbatches"]
        self._event({"event": "resume", "completed_epoch": checkpoint["epoch"] + 1,
                     "restored_scale": self.scaler.get_scale(), "discarded_pending_microbatches": discarded,
                     "continuation": "epoch boundary; worker RNG and accumulation cadence are not bitwise restored"})

    def _start_training(self, trainer):
        if self.device.type != "cuda" or self.epochs != self.protocol["training"]["epochs"] or self.batch_size != 6 or self.accumulate != 4:
            raise ValueError("Official device/budget/batch/accumulation changed")
        if len(self.train_loader.dataset) != 14061 or len(self.test_loader.dataset) != 3088 or "test" in self.data:
            raise ValueError("Incorrect official train/validation inputs")
        if type(self.optimizer).__name__ != "AdamW" or not state_is_finite(self.model.state_dict()):
            raise ValueError("Incorrect optimizer or non-finite initialization")
        self.train_started = time.perf_counter()
        self.warmup_iterations = self._get_warmup_iterations(len(self.train_loader))
        self._event({"event": "training_started", "start_epoch": self.start_epoch + 1, "epochs": self.epochs,
                     "scale": self.scaler.get_scale(), "accounting": self.monitor.state_dict()})

    def _begin_epoch(self, trainer):
        self._epoch_started = time.perf_counter()
        self._images_at_epoch_start = self.training_images_seen
        torch.cuda.reset_peak_memory_stats()

    def preprocess_batch(self, batch):
        result = super().preprocess_batch(batch)
        self.pending_microbatches += 1
        return result

    def _end_batch(self, trainer):
        if not state_is_finite({"loss": self.loss, **self.loss_items}):
            raise FloatingPointError("Non-finite training loss")
        self.train_batches += 1

    def optimizer_step(self):
        controlled_step(self.model, self.optimizer, self.scaler, self.ema, self.monitor,
                        lambda event: self._event({"event": "amp", **event}))
        self.pending_microbatches = 0
        self.optimizer_attempts = self.monitor.attempts
        self.optimizer_updates = self.monitor.successful

    def _end_epoch(self, trainer):
        torch.cuda.synchronize()
        self.training_seconds = time.perf_counter() - self._epoch_started
        self.training_peak = memory_peaks()
        if self.training_images_seen - self._images_at_epoch_start != 14061:
            raise ValueError("Official epoch did not consume exactly 14,061 training draws")
        if not state_is_finite(self.model.state_dict()) or not state_is_finite(self.ema.ema.state_dict()):
            raise FloatingPointError("Non-finite live/EMA state at epoch end")

    def _handle_nan_recovery(self, epoch):
        if not self.validation_runs or self.validation_runs[-1]["images"] != 3088:
            raise ValueError("The full validation pass is missing")
        self.current_summary = epoch_record(epoch + 1, self.epochs, self.tloss, self.metrics,
            self.training_seconds, self.validation_runs[-1]["seconds"], self.best_fitness, self.monitor.state_dict())
        self.current_per_class = per_class_record(self.validator.metrics, self.data["names"])
        # Genuine corruption is never repaired with nan_to_num or an unrecorded epoch retry.
        return False

    def _event(self, value):
        value = {"epoch": getattr(self, "epoch", -1) + 1, **value}
        append_json(self.directory / "events.jsonl", value)
        if value["event"] == "amp":
            print("AMP " + json.dumps(value, allow_nan=False), flush=True)

    def save_model(self):
        if not state_is_finite(self.ema.ema.state_dict()) or not state_is_finite(self.model.state_dict()):
            raise FloatingPointError("Refusing to save numerical corruption")
        if self.scheduler.last_epoch != self.epoch or self.ema.updates != self.monitor.successful:
            raise ValueError("Checkpoint schedule/EMA position is inconsistent")
        ema = model_snapshot(self.ema.ema)
        new_best = self.fitness >= self._best_seen_score
        if new_best:
            self._best_model, self._best_seen_score, self.best_epoch = ema, self.fitness, self.epoch
        payload = {"epoch": self.epoch, "model": model_snapshot(self.model), "ema": ema,
            "best_ema": self._best_model, "best_fitness": self.best_fitness, "updates": self.ema.updates,
            "stopper": {"best_fitness": self.stopper.best_fitness, "best_epoch": self.stopper.best_epoch},
            "optimizer": cpu_state(self.optimizer.state_dict()), "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(), "accounting": self.monitor.state_dict(),
            "train_args": vars(self.args).copy(), "train_metrics": {**self.metrics, "fitness": self.fitness},
            "train_results": self.read_results_csv(), "rng": self._rng_state(),
            "per_class": self.current_per_class,
            "official": {**self.identity, "format_version": 1, "resumable": True,
                         "best_epoch": self.best_epoch, "pending_microbatches": self.pending_microbatches}}
        publish_checkpoint(self.last, payload, backup=True)
        if new_best:
            publish_checkpoint(self.best, payload, backup=True)
        if (self.epoch + 1) % 10 == 0:
            publish_checkpoint(self.wdir / f"epoch{self.epoch + 1:03}.pt", payload)
        return True

    def _report_epoch(self, trainer):
        record = dict(self.current_summary)
        record["epoch_seconds"] = self.epoch_time
        record["training_memory"] = self.training_peak
        display_epoch(self.directory, record)
        append_json(self.directory / "per_class.jsonl", {"epoch": self.epoch + 1, "classes": self.current_per_class})

    def final_eval(self):
        # Retain resumable FP32 last/best files and the metrics already measured each epoch.
        self._event({"event": "training_loop_completed", "epochs": self.epochs,
                     "successful_updates": self.monitor.successful, "skipped_updates": self.monitor.skipped})

    @staticmethod
    def _rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

    @staticmethod
    def _restore_rng(state):
        if state:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if state["cuda"] is not None:
                torch.cuda.set_rng_state_all(state["cuda"])


def restore_results_csv(path, results):
    # A new resume segment receives only the selected checkpoint's completed history.
    if not results or len({len(values) for values in results.values()}) != 1:
        raise ValueError("Missing or inconsistent checkpoint results history")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(results)
        writer.writerows(zip(*results.values()))
