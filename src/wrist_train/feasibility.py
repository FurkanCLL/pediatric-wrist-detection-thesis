"""Run bounded training-step measurements or exactly one diagnostic epoch."""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from .common import OUTPUT, candidate, configure_runtime, require_cuda, save_json, write_data_yaml

configure_runtime()
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import RandomHSV, RandomPerspective
from ultralytics.utils.torch_utils import init_seeds

from .diagnostics import CheckedTrainer, Telemetry, finite_tensors, memory_peaks
from wrist_prepare.common import CLASSES, digest


def augmentation_probe():
    # Exercise intensity augmentation on equal grayscale channels, rather than trusting its name.
    plane = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    original = np.repeat(plane[..., None], 3, axis=2)
    np.random.seed(42)
    transform = RandomHSV(hgain=0.0, sgain=0.0, vgain=0.1)
    outputs = [transform({"img": original.copy()})["img"] for _ in range(16)]
    assert all(np.array_equal(im[..., 0], im[..., 1]) and np.array_equal(im[..., 0], im[..., 2]) for im in outputs)
    changed = sum(not np.array_equal(im, original) for im in outputs)
    assert changed > 0
    # Scale=0.1 is sampled over 0.9..1.1; zero shear/perspective keep the matrix affine.
    import random
    random.seed(42)
    affine = RandomPerspective(degrees=10, translate=0.05, scale=0.1, shear=0, perspective=0, size=(768, 768))
    matrices = [affine._compute_affine_matrix(np.zeros((768, 768, 3), dtype=np.uint8), (768, 768)) for _ in range(32)]
    assert all(0.9 <= scale <= 1.1 and np.array_equal(matrix[2], [0, 0, 1]) for matrix, scale in matrices)
    return {"grayscale_intensity_trials": 16, "changed_trials": changed, "equal_channels_preserved": True,
            "hsv_h": 0, "hsv_s": 0, "hsv_v": 0.1, "affine_trials": 32,
            "scale_bounds": [0.9, 1.1], "shear": 0, "perspective": 0,
            "optional_albumentations": "explicit empty list"}


def make_trainer(args, directory):
    settings = candidate()["training"].copy()
    data_path, frozen = write_data_yaml(directory, subset=args.action == "benchmark")
    weights = OUTPUT / "weights/yolov8s.pt"
    assets = json.loads((OUTPUT / "environment/pretrained_assets.json").read_text())
    expected = next(asset["sha256"] for asset in assets if asset["asset"] == "yolov8s.pt")
    assert digest(weights.read_bytes()) == expected, "Pretrained weights changed"
    # Smoke epochs are a hard scope limit; benchmark setup also cannot start official training.
    settings.update(epochs=1, batch=args.batch, amp=not args.no_amp, workers=args.workers,
                    model=str(weights), project=str(directory), name="run", data=str(data_path))
    save_json(directory / "resolved_config.json", settings)
    save_json(directory / "provenance.json", {
        "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_config_sha256": digest((Path(__file__).resolve().parents[2] / "configs/training_candidate.json").read_bytes()),
        "pretrained_sha256": expected, "split_manifest_sha256": frozen["split_manifest_sha256"]})
    init_seeds(42, deterministic=True)
    return CheckedTrainer(overrides=settings)


def describe_optimizer(trainer):
    return {"class": type(trainer.optimizer).__name__, "accumulation": trainer.accumulate,
            "effective_batch": trainer.batch_size * trainer.accumulate,
            "groups": [{key: group[key] for key in ("param_group", "lr", "weight_decay", "betas") if key in group}
                       for group in trainer.optimizer.param_groups]}


def benchmark(args, directory):
    require_cuda()
    trainer = make_trainer(args, directory)
    trainer._setup_train()
    assert trainer.amp == (not args.no_amp), "Framework changed the requested AMP status"
    assert trainer.batch_size == args.batch and len(trainer.train_loader.dataset) == 180
    trainer._model_train()
    trainer.optimizer.zero_grad()
    plan = candidate()["benchmark"]
    iterator = iter(trainer.train_loader)
    measurements = []
    torch.cuda.reset_peak_memory_stats()
    with Telemetry() as telemetry:
        for step in range(plan["warmup_steps"] + plan["measured_steps"]):
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = next(iterator)
            loaded = time.perf_counter()
            with torch.autocast("cuda", enabled=trainer.amp):
                batch = trainer.preprocess_batch(batch)
                loss, items = trainer.model(batch)
                loss = loss.sum()
            if not finite_tensors(loss) or not finite_tensors(items):
                raise FloatingPointError("Non-finite benchmark loss")
            trainer.scaler.scale(loss).backward()
            # Every measured iteration includes an optimizer/EMA step: conservative timing and state memory.
            trainer.optimizer_step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            measurements.append({"step": step, "warmup": step < plan["warmup_steps"],
                                 "seconds": elapsed, "loader_wait_s": loaded - started,
                                 "images": len(batch["img"]),
                                 "loss": {key: float(value) for key, value in items.items()}})
            print(json.dumps({"step": step, "seconds": round(elapsed, 3), "scale": trainer.scaler.get_scale()}), flush=True)
    steady = measurements[plan["warmup_steps"]:]
    seconds = sum(row["seconds"] for row in steady)
    throughput = sum(row["images"] for row in steady) / seconds
    assert trainer.optimizer_updates > 0 and trainer.parameter_changes > 0
    assert finite_tensors(list(trainer.model.parameters()))
    result = {"status": "passed", "purpose": "diagnostic benchmark; not an experiment result",
              "imgsz": 768, "batch": args.batch, "amp_requested": not args.no_amp, "amp_actual": trainer.amp,
              "workers": args.workers, "train_subset_images": len(trainer.train_loader.dataset),
              "warmup_steps": plan["warmup_steps"], "measured_steps": len(steady),
              "mean_iteration_s": seconds / len(steady), "median_iteration_s": statistics.median(r["seconds"] for r in steady),
              "images_per_second": throughput, "mean_loader_wait_s": statistics.mean(r["loader_wait_s"] for r in steady),
              "estimated_training_epoch_s": 14061 / throughput,
              "optimizer_attempts": trainer.optimizer_attempts, "optimizer_updates": trainer.optimizer_updates,
              "parameter_changes": trainer.parameter_changes, "gradient_overflows": trainer.gradient_overflows,
              "scaler_events": trainer.scaler_events, "optimizer": describe_optimizer(trainer),
              "measured_step_accumulation": 1, "augmentation": augmentation_probe(),
              "gpu_telemetry": telemetry.summary(), **memory_peaks()}
    save_json(directory / "iterations.json", measurements)
    save_json(directory / "result.json", result)
    print(json.dumps(result, indent=2))
    close_loaders(trainer)


def close_loaders(trainer):
    for loader in (trainer.train_loader, trainer.test_loader):
        if hasattr(loader, "close"):
            loader.close()


def smoke(args, directory):
    require_cuda()
    trainer = make_trainer(args, directory)
    started = time.perf_counter()
    with Telemetry() as telemetry:
        trainer.train()
    elapsed = time.perf_counter() - started
    assert trainer.epochs == 1 and trainer.training_images_seen == 14061 and trainer.epoch == 0
    assert trainer.amp == (not args.no_amp) and trainer.batch_size == args.batch
    assert trainer.train_batches == math.ceil(14061 / args.batch)
    assert trainer.optimizer_updates > 0 and trainer.parameter_changes > 0
    assert len(trainer.validation_runs) >= 1 and trainer.validation_runs[0]["trainer_integrated"]
    assert trainer.validation_runs[0]["images"] == 3088
    with trainer.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and all(math.isfinite(float(v)) for v in rows[0].values())
    checkpoints = []
    for path in (trainer.last, trainer.best):
        assert path.is_file()
        loaded = YOLO(str(path)).model
        assert [loaded.names[i] for i in range(8)] == list(CLASSES)
        assert finite_tensors(list(loaded.parameters()))
        checkpoints.append({"role": path.stem, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes()),
                            "reloaded_finite": True, "classes": 8})
        del loaded
    result = {"status": "passed", "purpose": "one-epoch diagnostic only; not the official baseline",
              "epochs_completed": 1, "training_images": trainer.training_images_seen,
              "train_batches": trainer.train_batches, "imgsz": 768, "batch": args.batch, "amp": trainer.amp,
              "workers": args.workers, "optimizer": describe_optimizer(trainer),
              "optimizer_attempts": trainer.optimizer_attempts, "optimizer_updates": trainer.optimizer_updates,
              "parameter_changes": trainer.parameter_changes, "gradient_overflows": trainer.gradient_overflows,
              "scaler_events": trainer.scaler_events, "actual_warmup_iterations": trainer.warmup_iterations,
              "training_s": trainer.training_seconds, "training_images_per_s": 14061 / trainer.training_seconds,
              "total_wall_s": elapsed, "training_memory": trainer.training_peak,
              "validation_runs": trainer.validation_runs, "checkpoint_verification": checkpoints,
              "diagnostic_metrics_only": {k.strip(): float(v) for k, v in rows[0].items()},
              "gpu_telemetry": telemetry.summary(), "augmentation": augmentation_probe(),
              "test_evaluated": False, "official_training_started": False}
    save_json(directory / "result.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["benchmark", "smoke"])
    parser.add_argument("--batch", type=int, choices=[6, 5, 4], default=6)
    parser.add_argument("--workers", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    if not args.attempt.replace("_", "").replace("-", "").isalnum():
        parser.error("Use a simple new attempt name")
    directory = OUTPUT / args.attempt
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "request.json", vars(args))
    try:
        (benchmark if args.action == "benchmark" else smoke)(args, directory)
    except Exception as error:
        # Never erase a failed attempt or publish non-standard JSON NaN values.
        save_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc(), "request": vars(args)})
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
