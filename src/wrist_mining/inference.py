"""Protected, deterministic full-image inference on the frozen training split."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import (ROOT, LOCAL, PUBLIC, config, training_records, training_path,
                     training_hashes, stability_subset, identity, sha, save)
from wrist_eval.adaptive_study import load_verified_checkpoint, ViewRunner
from wrist_eval.metrics import distribution


def source_identity():
    files = [Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("difficulty.py"),
             ROOT/"src/wrist_eval/sliced.py"]
    return {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_text(encoding="utf-8").encode()).hexdigest() for f in files}


def run(checkpoint, attempt):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    from wrist_train.protocol import load_protocol

    cfg = config()
    wanted = cfg["checkpoints"][checkpoint]
    if not attempt or Path(attempt).name != attempt:
        raise ValueError("Invalid attempt name")
    directory = LOCAL / "inference" / attempt
    if (directory / "summary.json").exists():
        summary = json.loads((directory / "summary.json").read_text())
        if (summary["checkpoint_sha256"] != wanted["sha256"] or summary["config_sha256"] != identity(cfg)
                or summary["normalized_source_sha256"] != source_identity()
                or sha(directory/"predictions.jsonl") != summary["predictions_sha256"]):
            raise ValueError("Existing completed inference differs")
        print("Reusing verified complete training inference; no GPU work", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=False)
    gate = json.loads((PUBLIC/"automated_tests.json").read_text())
    if not gate["passed"] or gate["source_sha256"] != source_identity() or gate["config_sha256"] != identity(cfg):
        raise ValueError("Training-only diagnostic tests have not passed")
    rows = training_records()
    if checkpoint != "best.pt":
        rows = stability_subset(rows)
    hashes = training_hashes()
    for row in rows:
        if sha(training_path(row, "labels")) != hashes[row["filestem"]][1]:
            raise ValueError("Frozen training label changed")
    save(directory/"selection.json", [r["filestem"] for r in rows])
    save(directory/"manifest.json", {"checkpoint": checkpoint, "checkpoint_sha256": wanted["sha256"],
         "config_sha256": identity(cfg), "split": "train", "images": len(rows)})
    require_cuda()
    init_seeds(cfg["seed"], deterministic=True)
    _, lock = load_protocol()
    started = time.perf_counter()
    model = load_verified_checkpoint(ROOT/"outputs/training/standard_baseline/weights"/checkpoint, wanted["sha256"], lock)
    runner = ViewRunner(model)
    load_seconds = time.perf_counter()-started
    for _ in range(3):
        runner(np.full((1024, 576, 3), 114, dtype=np.uint8), None)
    records, replay = [], []
    loop_start = time.perf_counter()
    with (directory/"predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            data = training_path(row).read_bytes()
            if hashlib.sha256(data).hexdigest() != hashes[row["filestem"]][0]:
                raise ValueError("Frozen training image changed")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Frozen training geometry differs")
            view = runner(image, None)
            seconds = time.perf_counter()-started
            item = {"stem": row["filestem"], "pred": view["predictions"].tolist(),
                "seconds": seconds, "forward_passes": view["counters"]["forward_passes"],
                "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                "reserved_peak_bytes": torch.cuda.max_memory_reserved()}
            stream.write(json.dumps(item, allow_nan=False)+"\n")
            stream.flush()
            records.append(item)
            if index < 3:
                repeated = runner(image, None)["predictions"]
                if not np.array_equal(view["predictions"], repeated):
                    raise ValueError("Bounded training prediction replay differs")
                replay.append(True)
            if index % 500 == 0:
                print(f"{checkpoint}: {index+1}/{len(rows)} training images", flush=True)
    sources = source_identity()
    result = {"checkpoint": checkpoint, "completed_epoch": wanted["epoch"], "checkpoint_sha256": wanted["sha256"],
        "split": "train", "images": len(rows), "patients": len({r["patient_id"] for r in rows}),
        "config_sha256": identity(cfg), "selection_sha256": sha(directory/"selection.json"),
        "predictions_sha256": sha(directory/"predictions.jsonl"), "model_load_seconds": load_seconds,
        "loop_seconds_including_export_hashing_and_replay": time.perf_counter()-loop_start,
        "timing_decode_to_original_outputs_excluding_file_read_hash": distribution([r["seconds"] for r in records]),
        "peak_allocated_bytes": max(r["allocated_peak_bytes"] for r in records),
        "peak_reserved_bytes": max(r["reserved_peak_bytes"] for r in records),
        "physical_image_calls": len(records), "warmup_calls": 3, "exact_replay_calls": len(replay),
        "all_training_images_and_labels_sha_verified": True, "all_original_coordinate_outputs_valid": True,
        "numerical_convention": "Pinned Phase 3B adapter, unfused FP32 EMA, FP16 input/CUDA autocast; 768, .001 pool, NMS .7, max_det 300",
        "global_shape_policy": "Existing per-view half-stride shape from training image dimensions only; no validation shape profile",
        "no_training_or_holdout_evaluation": True, "normalized_source_sha256": sources,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    save(directory/"summary.json", result)
    save(PUBLIC/(checkpoint.replace(".pt", "")+"_inference.json"), result)
    pointer = LOCAL/"prediction_active.json"
    active = json.loads(pointer.read_text()) if pointer.exists() else {}
    active[checkpoint] = attempt
    save(pointer, active)
    del runner, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=("best.pt", "epoch010.pt", "epoch040.pt"), required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.attempt)
