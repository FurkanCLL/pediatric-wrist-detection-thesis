"""Bounded E/F pilots and full validation with unchanged numerical methods."""

import argparse
from collections import Counter
import json
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

from wrist_eval.common import config, image_path, verify_validation_files, sha, identity
from wrist_eval.feasibility import pilot_rows, summarize as sahi_summary
from wrist_eval.metrics import ap_summary, native_stats, distribution
from wrist_eval.analyze import truth
from wrist_eval.sliced import make_adapter, predict, tile_dimensions
from wrist_eval.adaptive import predict_adaptive
from wrist_eval.adaptive_study import ViewRunner, private_record
from wrist_compare.infer import parameter_digest, assert_metrics
from .common import (ROOT, PUBLIC, read, write_new, require, manifests, source_identity, study_path,
    frozen_study, historical_inputs, preservation, validation_records, checked_checkpoint, install_access_guard)
from .checks import equal_global, verify_f, valid_timing


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    tests = read(PUBLIC/"automated_tests.json")
    require(tests["passed"] and tests["source_identity"] == source_identity(), "Current implementation has not passed its test gate")
    executions, _ = manifests()
    before = preservation(directory, "before")
    rows = validation_records()
    require(sum(len(r["boxes"]) for r in rows) == 7204, "Validation annotation population differs")
    files = verify_validation_files(rows)
    payload, checkpoint = checked_checkpoint("D")
    import torch
    require(sum(isinstance(m, torch.nn.BatchNorm2d) for m in payload["ema"].modules()) == 57, "Expected unfused YOLOv8s")
    del payload
    predictions, shapes, metrics, sources = historical_inputs(rows)
    for condition in "ABCD":
        actual = ap_summary([native_stats(truth(r), predictions[condition][r["filestem"]]) for r in rows])
        assert_metrics(actual, metrics[condition])
    selection = pilot_rows(rows)
    require(len(selection) == 24, "Pilot size differs")
    historical_selection = read(ROOT/"outputs/error_analysis_sahi/pilot/standard_serial/selection.json")
    require([r["filestem"] for r in selection] == historical_selection, "Historical pilot selection differs")
    from wrist_train.protocol import environment_check, load_protocol
    environment = environment_check(load_protocol()[0])
    write_new(directory/"shape_profile.json", shapes)
    write_new(directory/"linkage.json", [{"stem": r["filestem"], "patient_id": r["patient_id"]} for r in rows])
    write_new(directory/"manifest.json", {"executions": executions, "source_identity": source_identity(),
        "implementation_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": checkpoint, "environment": environment, "prepared_validation_bytes": files,
        "preservation_before": before, "historical_sources": sources, "historical_metrics_reproduced": metrics,
        "membership_sha256": identity([r["filestem"] for r in rows]), "pilot_selection_sha256": identity(historical_selection),
        "shape_profile_sha256": sha(directory/"shape_profile.json")})
    print("E/F preflight passes: A/B/C/D reproduce; historical settings and rarity checkpoint verified", flush=True)


def execute(directory, stage, condition):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    manifest = frozen_study(directory)
    _, cfg = manifests()
    rows = validation_records()
    prior, shapes, _, _ = historical_inputs(rows)
    if stage == "pilot":
        rows = pilot_rows(rows)
    else:
        gate = read(directory/f"pilot_{condition}/summary.json")
        require(gate["passed"] and gate["source_identity"] == source_identity() and
            not (directory/f"pilot_{condition}/failure.json").exists(), "Required technical pilot has not passed")
    destination = directory/f"{stage}_{condition}"
    destination.mkdir(exist_ok=False)
    write_new(destination/"selection.json", [r["filestem"] for r in rows])
    require_cuda()
    init_seeds(42, deterministic=True)
    began = time.perf_counter()
    payload, receipt = checked_checkpoint("D")
    model = payload["ema"].float().eval().to("cuda:0").requires_grad_(False)
    before_weights = parameter_digest(model)
    del payload
    with patch("wrist_eval.common.LOCAL", destination):
        adapter = make_adapter(model) if condition == "E" else ViewRunner(model)
    require((adapter.model if condition == "E" else adapter.adapter.model) is model, "Adapter replaced the rarity EMA")
    load_seconds = time.perf_counter()-began
    warmup_start = time.perf_counter()
    for _ in range(3):
        synthetic = np.full((1024, 576, 3), 114, dtype=np.uint8)
        adapter.perform_batch_inference([synthetic]) if condition == "E" else adapter(synthetic, None)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-warmup_start

    def view(image, shape, reference):
        # The same object supplies every global/local call; the core sees no condition or ground truth.
        started = time.perf_counter()
        value = adapter(image, shape)
        value["view_wall_seconds"] = time.perf_counter()-started
        if shape is not None:
            # Stop before selecting or running local views if the required D reproduction fails.
            stem, expected = reference
            try:
                equal_global(value["predictions"], expected)
            except ValueError:
                write_new(destination/"global_discrepancy.json", {"stem": stem,
                    "actual": value["predictions"].tolist(), "expected_D": expected.tolist()})
                raise
        return value

    def infer(row):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
        require(image is not None and image.shape[:2] == (row["height"], row["width"]), "Prepared geometry changed")
        decoded = time.perf_counter()
        if condition == "E":
            pred, timing = predict(image, adapter, "sahi_context", shapes[row["filestem"]])
            timing.update(decode_seconds=decoded-started, e2e_seconds=time.perf_counter()-started)
            from sahi.slicing import get_slice_bboxes
            sh, sw = tile_dimensions(row["height"], row["width"], .667)
            tiles = get_slice_bboxes(row["height"], row["width"], slice_height=sh, slice_width=sw,
                overlap_height_ratio=.2, overlap_width_ratio=.2, auto_slice_resolution=False)
            require(len(tiles) == timing["slices"] == 4, "E must use exactly four context slices")
            valid_timing(timing, 5)
            return pred, timing, None
        reference = (row["filestem"], prior["D"][row["filestem"]])
        result = predict_adaptive(image, lambda crop, shape: view(crop, shape, reference),
                                  shapes[row["filestem"]], cfg, ["V2"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        record = private_record(row["filestem"], result, decoded-started, elapsed,
            {"allocated_peak_bytes": torch.cuda.max_memory_allocated(), "reserved_peak_bytes": torch.cuda.max_memory_reserved()})
        try:
            counts = verify_f(record, reference[1], cfg)
        except Exception:
            write_new(destination/"failed_image_bundle.json", record)
            raise
        counters = [result["global_view"]["counters"], *[v["counters"] for v in result["local_views"]]]
        timing = {"e2e_seconds": elapsed, "decode_seconds": decoded-started,
            "paired_global_e2e_seconds": decoded-started+result["global_view"]["view_wall_seconds"],
            "gpu_forward_ms": sum(v["gpu_forward_ms"] for v in counters), "forward_passes": counts["calls"],
            "global_gpu_forward_ms": counters[0]["gpu_forward_ms"], "planning_seconds": result["planning_seconds"],
            "fusion_seconds": result["fusion_seconds"]["V2"], **{k: record["timing"][k] for k in ("allocated_peak_bytes", "reserved_peak_bytes")}}
        valid_timing(timing, 1+counts["rois"])
        record["verification_counts"] = counts
        return result["variants"]["V2"]["predictions"], timing, record

    timings, saved, diagnostics = [], {}, Counter()
    roi_counts = Counter()
    bundle_stream = (destination/"bundles.jsonl").open("x", encoding="utf-8") if condition == "F" else None
    loop_started = time.perf_counter()
    try:
        with (destination/"predictions.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                pred, timing, bundle = infer(row)
                saved[row["filestem"]] = pred
                timings.append(timing)
                stream.write(json.dumps({"stem": row["filestem"], "patient_id": row["patient_id"],
                    "pred": pred.tolist(), "timing": timing}, allow_nan=False)+"\n")
                stream.flush()
                if bundle is not None:
                    bundle_stream.write(json.dumps(bundle, allow_nan=False)+"\n")
                    bundle_stream.flush()
                    diagnostics.update(bundle["verification_counts"])
                    roi_counts[len(bundle["plan"]["rois"])] += 1
                if index % 100 == 0:
                    print(f"{condition}/{stage}: {index+1}/{len(rows)}", flush=True)
    finally:
        if bundle_stream is not None:
            bundle_stream.close()
    loop_seconds = time.perf_counter()-loop_started
    if stage == "pilot":
        for row in rows[:3]:
            pred, _, _ = infer(row)
            require(np.array_equal(pred, saved[row["filestem"]]), "Technical pilot replay is not bitwise identical")
    else:
        pilot = [json.loads(s) for s in (directory/f"pilot_{condition}/predictions.jsonl").read_text().splitlines()]
        require(all(np.array_equal(saved[v["stem"]], np.asarray(v["pred"], np.float32).reshape(-1, 6)) for v in pilot),
                "Full predictions differ from pilot")
    require(before_weights == parameter_digest(model), "In-memory model state changed")
    timing_summary = sahi_summary(timings) if condition == "E" else {
        **{key: distribution([t[key] for t in timings]) for key in timings[0]},
        "throughput_images_per_second": len(rows)/sum(t["e2e_seconds"] for t in timings)}
    limits = cfg["phase_3c2_plan"]["proposed_success"]
    require(timing_summary["e2e_seconds"]["mean"]*3088 < 10800 and
            timing_summary["allocated_peak_bytes"]["max"]/2**30 < limits["pilot_peak_allocated_gib_max"], "Feasibility limit failed")
    summary = {"passed": True, "condition": condition, "stage": stage, "images": len(rows),
        "patients": len({r["patient_id"] for r in rows}), "checkpoint_sha256": receipt["sha256"],
        "source_identity": source_identity(), "execution_manifest_sha256": identity(manifest["executions"][condition]),
        "prediction_file_sha256": sha(destination/"predictions.jsonl"), "timing": timing_summary,
        "loop_wall_seconds_including_export_checks": loop_seconds, "model_load_verification_seconds": load_seconds,
        "warmup_seconds": warmup_seconds, "pilot_ap_not_calculated": stage == "pilot", "pilot_replay_images": 3 if stage == "pilot" else 0,
        "in_memory_model_state_unchanged": True, "no_training_or_test": True,
        "physical_model_calls": sum(t["forward_passes"] for t in timings)}
    if condition == "F":
        summary.update(global_exact_to_D=True, roi_count_images={str(i): roi_counts[i] for i in range(3)},
            diagnostics=dict(diagnostics), bundles_sha256=sha(destination/"bundles.jsonl"))
    write_new(destination/"summary.json", summary)
    print(f"{condition}/{stage} passes: {len(rows)} images, {summary['physical_model_calls']} calls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "pilot", "full"))
    parser.add_argument("condition", nargs="?", choices=("E", "F"))
    parser.add_argument("--study", default="study_01")
    args = parser.parse_args()
    if args.stage != "init" and args.condition is None:
        parser.error("Select E or F")
    directory = study_path(args.study)
    destination = directory if args.stage == "init" else directory/f"{args.stage}_{args.condition}"
    if destination.exists():
        raise FileExistsError("Existing evidence is preserved; select a new attempt")
    install_access_guard()
    try:
        initialize(directory) if args.stage == "init" else execute(directory, args.stage, args.condition)
    except BaseException as error:
        if destination.exists():
            write_new(destination/"failure.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
