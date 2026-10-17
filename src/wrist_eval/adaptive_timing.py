"""Predeclared standalone-timing fallback; require identical sealed predictions."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .common import ROOT, save, identity, image_path, validation_records, sha
from .adaptive_config import approved_settings
from .adaptive_study import LOCAL, PUBLIC, load_verified_checkpoint, ViewRunner, source_identity
from .adaptive_evidence import active_records
from .adaptive import predict_adaptive
from .metrics import distribution


def run(attempt):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.protocol import load_protocol
    from wrist_train.common import require_cuda
    cfg, manifest = approved_settings()
    if Path(attempt).name != attempt:
        raise ValueError("Invalid timing attempt name")
    base_directory, sealed = active_records("full")
    summary = json.loads((PUBLIC/"full_summary.json").read_text())
    if summary["bundles_sha256"] != sha(base_directory/"bundles.jsonl") or summary["implementation_source_sha256"] != source_identity():
        raise ValueError("Sealed accuracy evidence changed")
    shapes = json.loads((base_directory/"shape_profile.json").read_text())
    rows = validation_records()
    expected = {r["stem"]: r for r in sealed}
    directory = LOCAL/"standalone_timing"/attempt
    directory.mkdir(parents=True, exist_ok=False)
    require_cuda()
    _, lock = load_protocol()
    result = {"reason": "Approved design section 16 fallback: six-image pilot composed/standalone means differ substantially; measure complete standalone paths",
              "no_algorithm_changes": True, "no_new_accuracy_selection": True,
              "proposal_sha256": identity(cfg), "execution_manifest_sha256": identity(manifest),
              "checkpoint_sha256": manifest["checkpoint"]["sha256"], "variants": {},
              "accuracy_bundle_sha256": summary["bundles_sha256"], "implementation_source_sha256": source_identity()}
    for variant in ("V1", "V2"):
        init_seeds(42, deterministic=True)
        began = time.perf_counter()
        model = load_verified_checkpoint(ROOT/manifest["checkpoint"]["path"], manifest["checkpoint"]["sha256"], lock)
        runner = ViewRunner(model)
        load_seconds = time.perf_counter()-began
        began = time.perf_counter()
        for _ in range(3):
            runner(np.full((1024, 576, 3), 114, dtype=np.uint8), None)
        warmup_seconds = time.perf_counter()-began
        # The already-required global call also supplies a contemporaneous one-pass wall-time control.
        def timed_view(image, shape):
            started = time.perf_counter()
            value = runner(image, shape)
            value["view_wall_seconds"] = time.perf_counter()-started
            return value
        timings = []
        loop_start = time.perf_counter()
        with (directory/f"{variant}_timings.jsonl").open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (row["height"], row["width"]):
                    raise ValueError("Prepared timing image geometry differs")
                decoded = time.perf_counter()
                prediction = predict_adaptive(image, timed_view, shapes[row["filestem"]], cfg, [variant])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                prior = expected[row["filestem"]]
                if not np.array_equal(prediction["global"], np.asarray(prior["global"], dtype=np.float32).reshape(-1, 6)):
                    raise ValueError("Paired global timing control changed original predictions")
                values = np.asarray(prior["variants"][variant]["pred"], dtype=np.float32).reshape(-1, 6)
                if not np.array_equal(prediction["variants"][variant]["predictions"], values):
                    raise ValueError("Standalone timing predictions differ from sealed accuracy outputs")
                if prediction["passes"] != prior["timing"]["passes"] or prediction["plan"] != prior["plan"]:
                    raise ValueError("Timing mode changed the ROI plan/pass count")
                counters = [prediction["global_view"]["counters"], *[v["counters"] for v in prediction["local_views"]]]
                timing = {"stem": row["filestem"], "e2e_seconds": elapsed,
                    "paired_global_e2e_seconds": decoded-started+prediction["global_view"]["view_wall_seconds"],
                    "decode_seconds": decoded-started, "planning_seconds": prediction["planning_seconds"],
                    "common_seconds": prediction["common_seconds"], "fusion_seconds": prediction["fusion_seconds"][variant],
                    "global_gpu_forward_ms": counters[0]["gpu_forward_ms"],
                    "gpu_forward_ms": sum(c["gpu_forward_ms"] for c in counters),
                    "preprocess_seconds": sum(c["preprocess_seconds"] for c in counters),
                    "nms_restore_seconds": sum(c["nms_restore_seconds"] for c in counters),
                    "passes": prediction["passes"], "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved()}
                timings.append(timing)
                if timing["allocated_peak_bytes"] > cfg["phase_3c2_plan"]["proposed_success"]["pilot_peak_allocated_gib_max"]*2**30:
                    raise ValueError("Approved tensor-memory bound exceeded")
                stream.write(json.dumps(timing, allow_nan=False)+"\n")
                stream.flush()
                if index % 250 == 0:
                    print(f"Standalone {variant}: {index+1}/{len(rows)}", flush=True)
        result["variants"][variant] = {"images": len(timings), "predictions_bitwise_identical_to_sealed": True,
            "paired_global_outputs_bitwise_identical": True,
            "physical_model_calls": sum(t["passes"] for t in timings), "model_load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds, "loop_wall_seconds_including_export": time.perf_counter()-loop_start,
            "timing": {key: distribution([t[key] for t in timings]) for key in timings[0] if key != "stem"},
            "private_timing_sha256": sha(directory/f"{variant}_timings.jsonl")}
        save(directory/f"{variant}_summary.json", result["variants"][variant])
        del runner, model
        torch.cuda.empty_cache()
    result["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["timing_source_sha256"] = sha(Path(__file__))
    save(directory/"summary.json", result)
    save(PUBLIC/"standalone_timing.json", result)
    save(LOCAL/"standalone_timing_active.json", {"attempt": attempt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    run(parser.parse_args().attempt)
