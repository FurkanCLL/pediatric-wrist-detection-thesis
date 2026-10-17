"""One technical pilot and one full standard-image pass with the verified D EMA."""

import argparse
from datetime import datetime, timezone
import json
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

from wrist_eval.common import image_path, verify_validation_files, config, identity, sha
from wrist_eval.feasibility import pilot_rows, summarize
from wrist_eval.metrics import native_stats, ap_summary, validate_predictions
from wrist_eval.analyze import truth
from wrist_eval.sliced import make_adapter, predict
from .common import (ROOT, OLD_PUBLIC, write_new, read, require, settings, source_identity,
    checked_checkpoint, validation_records, references, preservation, study_path, frozen_study, install_access_guard)


def assert_metrics(actual, expected):
    for key in ("map50", "map75", "map50_95"):
        require(abs(actual[key]-expected[key]) < 1e-12, "Preserved A AP no longer reproduces")
    for name in actual["classes"]:
        for key in ("ap50", "ap75", "ap50_95"):
            require(abs(actual["classes"][name][key]-expected["classes"][name][key]) < 1e-12,
                    "Preserved A per-class AP no longer reproduces")


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    cfg = settings()
    rows = validation_records()
    before = preservation(directory, "before")
    files = verify_validation_files(rows)
    a, a_receipt = checked_checkpoint("A")
    d, d_receipt = checked_checkpoint("D")
    architecture = lambda model: {k: list(v.shape) for k, v in model.state_dict().items()}
    require(architecture(a["ema"]) == architecture(d["ema"]) and a["ema"].yaml == d["ema"].yaml and
            a_receipt["pretrained_sha256"] == d_receipt["pretrained_sha256"], "A/D architectures or initialization differ")
    import torch
    require(sum(isinstance(m, torch.nn.BatchNorm2d) for m in d["ema"].modules()) == 57, "Expected unfused YOLOv8s EMA")
    del a, d
    baseline, shapes, summary = references(rows)
    baseline_metrics = ap_summary([native_stats(truth(r), baseline[r["filestem"]]) for r in rows])
    assert_metrics(baseline_metrics, summary["metrics"])
    from wrist_train.protocol import environment_check, load_protocol
    environment = environment_check(load_protocol()[0])
    selection = pilot_rows(rows)
    require(len(selection) == cfg["pilot_images"] == 24, "Technical pilot selection differs")
    write_new(directory/"validation_linkage.json", [{"stem": r["filestem"], "patient_id": r["patient_id"],
        "height": r["height"], "width": r["width"], "net_shape": shapes[r["filestem"]]} for r in rows])
    manifest = {"configuration": cfg, "common_inference_configuration": config(), "source_identity": source_identity(),
        "implementation_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoints": {"A": a_receipt, "D": d_receipt}, "validation_membership_sha256": identity([r["filestem"] for r in rows]),
        "pilot_selection_sha256": identity([r["filestem"] for r in selection]), "environment": environment,
        "prepared_validation_bytes": files, "baseline_ap_reproduced": baseline_metrics, "preservation_before": before,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    write_new(directory/"manifest.json", manifest)
    print("Preflight passes: A predictions reproduce; both EMA checkpoints and validation provenance verified", flush=True)


def parameter_digest(model):
    import hashlib
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def verify_item(row, prediction, timing):
    require(np.array_equal(prediction, validate_predictions(prediction, row["width"], row["height"])) and
            len(prediction) <= 300, "Malformed prediction or confidence ordering")
    require(timing["forward_passes"] == 1 and timing["slices"] == 0 and
            timing["post_merge_predictions"] == len(prediction), "Unexpected inference calls/slicing")
    require(all(np.isfinite(timing[k]) and timing[k] >= 0 for k in ("seconds", "e2e_seconds", "decode_seconds",
        "gpu_forward_ms", "preprocess_seconds", "model_wall_seconds", "nms_restore_seconds")) and
        timing["gpu_forward_ms"] > 0 and timing["seconds"] <= timing["e2e_seconds"], "Invalid timing fields")


def run(directory, stage):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    manifest = frozen_study(directory)
    rows = validation_records()
    _, shapes, _ = references(rows)
    if stage == "pilot":
        rows = pilot_rows(rows)
    else:
        gate = read(directory/"pilot/summary.json")
        require(gate["passed"] and gate["source_identity"] == source_identity() and
                gate["checkpoint_sha256"] == manifest["checkpoints"]["D"]["sha256"], "Technical pilot gate has not passed")
    destination = directory/stage
    destination.mkdir(exist_ok=False)
    write_new(destination/"selection.json", [r["filestem"] for r in rows])
    require_cuda()
    init_seeds(42, deterministic=True)
    started = time.perf_counter()
    payload, receipt = checked_checkpoint("D")
    model = payload["ema"].float().eval().to("cuda:0").requires_grad_(False)
    before_weights = parameter_digest(model)
    del payload
    # Redirect only the adapter's settings directory; numerical adapter code remains unchanged.
    with patch("wrist_eval.common.LOCAL", destination):
        adapter = make_adapter(model)
    require(adapter.model is model, "Adapter replaced the verified EMA model")
    load_seconds = time.perf_counter()-started
    warmup_start = time.perf_counter()
    for _ in range(3):
        adapter.perform_batch_inference([np.full((1024, 576, 3), 114, dtype=np.uint8)])
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-warmup_start
    timings, saved = [], {}
    loop_started = time.perf_counter()
    with (destination/"predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            start = time.perf_counter()
            image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
            require(image is not None and image.shape[:2] == (row["height"], row["width"]), "Validation image geometry differs")
            decoded = time.perf_counter()
            pred, timing = predict(image, adapter, "standard_serial", shapes[row["filestem"]])
            timing.update(decode_seconds=decoded-start, e2e_seconds=time.perf_counter()-start)
            verify_item(row, pred, timing)
            timings.append(timing)
            saved[row["filestem"]] = pred
            stream.write(json.dumps({"stem": row["filestem"], "patient_id": row["patient_id"],
                "pred": pred.tolist(), "timing": timing}, allow_nan=False)+"\n")
            stream.flush()
            if index % 100 == 0:
                print(f"D {stage}: {index+1}/{len(rows)}", flush=True)
    loop_seconds = time.perf_counter()-loop_started
    replay_images = 0
    if stage == "pilot":
        # A bounded repeat of the first three fixed pilot images checks deterministic numerical behavior.
        for row in rows[:3]:
            image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
            actual, _ = predict(image, adapter, "standard_serial", shapes[row["filestem"]])
            require(actual.shape == saved[row["filestem"]].shape and
                    np.allclose(actual, saved[row["filestem"]], atol=1e-6, rtol=0), "Pilot replay differs")
            replay_images += 1
    else:
        pilot_items = [json.loads(s) for s in (directory/"pilot/predictions.jsonl").read_text().splitlines()]
        for item in pilot_items:
            expected = np.asarray(item["pred"], dtype=np.float32).reshape(-1, 6)
            require(saved[item["stem"]].shape == expected.shape and
                    np.allclose(saved[item["stem"]], expected, atol=1e-6, rtol=0), "Full output differs from technical pilot")
    require(before_weights == parameter_digest(model), "In-memory model state changed during inference")
    summary = {"passed": True, "condition": "D", "stage": stage, "images": len(rows),
        "patients": len({r["patient_id"] for r in rows}), "checkpoint_sha256": receipt["sha256"],
        "source_identity": source_identity(), "common_inference_config_sha256": identity(config()),
        "selection_sha256": identity([r["filestem"] for r in rows]), "prediction_file_sha256": sha(destination/"predictions.jsonl"),
        "timing": summarize(timings), "model_load_and_verification_seconds": load_seconds,
        "synthetic_warmup_seconds": warmup_seconds, "loop_wall_seconds_including_export": loop_seconds,
        "pilot_replay_images": replay_images, "in_memory_model_state_unchanged": True, "pilot_ap_not_calculated": stage == "pilot",
        "new_A_inference": False, "sliced_or_adaptive_inference": False, "test_evaluation": False}
    limits = config()["full_run_limits"]
    require(summary["timing"]["e2e_seconds"]["mean"]*3088 < limits["max_predicted_seconds_per_variant"] and
            summary["timing"]["allocated_peak_bytes"]["max"]/2**30 < limits["pilot_peak_allocated_gib"], "Runtime feasibility gate failed")
    write_new(destination/"summary.json", summary)
    print(f"D {stage} verified: {len(rows)} images; no pilot AP or parameter tuning", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "pilot", "full"))
    parser.add_argument("--study", default="study_01")
    args = parser.parse_args()
    directory = study_path(args.study)
    install_access_guard()
    destination = directory if args.stage == "init" else directory/args.stage
    if destination.exists():
        raise FileExistsError("Existing attempt is preserved; choose a new study name")
    try:
        if args.stage == "init":
            initialize(directory)
        else:
            run(directory, args.stage)
    except BaseException as error:
        if destination.exists() and not (destination/"failure.json").exists():
            write_new(destination/"failure.json", {"stage": args.stage, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
