"""Instrument the real trainer without changing data, losses or optimization rules."""

from copy import copy
import json
import math
import subprocess
import threading
import time

from .common import OUTPUT, configure_runtime

configure_runtime()
import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import SETTINGS

# Optional integrations must not upload this project's local medical artifacts.
SETTINGS.update({key: False for key in ("sync", "hub", "wandb", "comet", "clearml", "dvc", "mlflow", "neptune", "raytune", "tensorboard") if key in SETTINGS})
SETTINGS.update({"weights_dir": str(OUTPUT / "weights")})


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


class Telemetry:
    """Sample GPU utilization independently of the timed batch loop."""

    def __init__(self):
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop_event.is_set():
            try:
                raw = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
                values = [float(v.strip()) for v in raw.strip().split(",")]
                self.samples.append(dict(zip(("utilization_percent", "device_memory_mib", "power_w", "temperature_c", "clock_mhz"), values)))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=6)

    def summary(self):
        if not self.samples:
            return {"samples": 0}
        return {"samples": len(self.samples), **{
            key: {"mean": sum(r[key] for r in self.samples) / len(self.samples), "max": max(r[key] for r in self.samples)}
            for key in self.samples[0]}}


class CheckedValidator(DetectionValidator):
    def postprocess(self, predictions):
        # Check raw predictions before NMS can hide invalid values.
        if not finite_tensors(predictions):
            raise FloatingPointError("Non-finite raw validation prediction")
        return super().postprocess(predictions)


class CheckedTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        self.physical_batch = None
        self.optimizer_updates = 0
        self.optimizer_attempts = 0
        self.gradient_overflows = []
        self.scaler_events = []
        self.parameter_changes = 0
        self.train_batches = 0
        self.training_images_seen = 0
        self.validation_runs = []
        self.train_batch_seconds = []
        self.initial_snapshot = None
        super().__init__(*args, **kwargs)
        self.physical_batch = self.batch_size
        self.add_callback("on_train_start", self._start_training)
        self.add_callback("on_train_batch_start", self._start_batch)
        self.add_callback("on_train_batch_end", self._end_batch)
        self.add_callback("on_train_epoch_end", self._end_epoch)
        self.add_callback("on_val_start", self._start_validation)
        self.add_callback("on_val_batch_end", self._check_validation)
        self.add_callback("on_val_end", self._end_validation)

    def _build_train_pipeline(self):
        # Reject the framework's automatic batch halving; candidates must remain explicit.
        if self.physical_batch is not None and self.batch_size != self.physical_batch:
            raise RuntimeError("Automatic OOM batch reduction blocked; preserve this attempt and use the next explicit candidate")
        super()._build_train_pipeline()

    def get_validator(self):
        return CheckedValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _start_training(self, trainer):
        assert self.device.type == "cuda" and self.data["nc"] == 8
        assert len(self.train_loader.dataset) == 14061
        self.train_started = time.perf_counter()
        self.warmup_iterations = self._get_warmup_iterations(len(self.train_loader))
        torch.cuda.reset_peak_memory_stats()
        self.initial_snapshot = next(self.model.parameters()).detach().clone()

    def _start_batch(self, trainer):
        self.batch_started = time.perf_counter()

    def preprocess_batch(self, batch):
        result = super().preprocess_batch(batch)
        self.training_images_seen += len(result["img"])
        return result

    def _end_batch(self, trainer):
        if not finite_tensors(self.loss_items) or not finite_tensors(self.loss):
            raise FloatingPointError("Non-finite training loss")
        self.train_batches += 1
        self.train_batch_seconds.append(time.perf_counter() - self.batch_started)
        if self.train_batches % 100 == 0:
            print(json.dumps({"training_batches": self.train_batches, "images_seen": self.training_images_seen,
                              "optimizer_updates": self.optimizer_updates, "elapsed_train_s": round(time.perf_counter() - self.train_started, 1)}), flush=True)

    def _end_epoch(self, trainer):
        torch.cuda.synchronize()
        self.training_seconds = time.perf_counter() - self.train_started
        self.training_peak = memory_peaks()
        if not finite_tensors(list(self.model.parameters())):
            raise FloatingPointError("Non-finite trained parameter")
        assert not torch.equal(self.initial_snapshot, next(self.model.parameters()))
        del self.initial_snapshot

    def optimizer_step(self):
        # GradScaler may calibrate its scale initially; keep every skipped update visible.
        self.optimizer_attempts += 1
        previous_scale = self.scaler.get_scale()
        parameter = next(p for p in self.model.parameters() if p.requires_grad)
        before = parameter.detach().clone()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        finite = bool(torch.isfinite(norm))
        if not finite:
            self.gradient_overflows.append(self.optimizer_attempts)
            if not self.amp:
                raise FloatingPointError("Non-finite FP32 gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale = self.scaler.get_scale()
        updated = finite and scale >= previous_scale
        if updated:
            self.optimizer_updates += 1
            self.parameter_changes += int(not torch.equal(before, parameter))
        if scale != previous_scale or not finite:
            self.scaler_events.append({"attempt": self.optimizer_attempts, "before": previous_scale,
                                       "after": scale, "finite_gradients": finite, "updated": updated})
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        # Persistent overflow is a failed diagnostic, rather than a reason to keep tuning.
        if not finite and (self.optimizer_attempts > 5 or len(self.gradient_overflows) > 4):
            raise FloatingPointError("AMP gradient overflow beyond initial scale calibration")

    def _start_validation(self, validator):
        torch.cuda.synchronize()
        self.val_started = time.perf_counter()
        self.val_batches = 0
        torch.cuda.reset_peak_memory_stats()

    def _check_validation(self, validator):
        if validator.training and not finite_tensors(validator.loss):
            raise FloatingPointError("Non-finite trainer-integrated validation loss")
        self.val_batches += 1

    def _end_validation(self, validator):
        torch.cuda.synchronize()
        result = {"trainer_integrated": validator.training, "seconds": time.perf_counter() - self.val_started,
                  "images": validator.seen, "batches": self.val_batches,
                  "batch": validator.dataloader.batch_size, "workers": validator.dataloader.num_workers,
                  "rect": validator.dataloader.dataset.rect, "augmentation": validator.dataloader.dataset.augment,
                  **memory_peaks()}
        self.validation_runs.append(result)

    def _handle_nan_recovery(self, epoch):
        # Preserve the first failure instead of silently rerunning an epoch.
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in self.metrics.values()):
            raise FloatingPointError("Non-finite validation metric")
        return super()._handle_nan_recovery(epoch)


def memory_peaks():
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
