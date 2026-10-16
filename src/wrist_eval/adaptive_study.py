"""Validation-only CG-ALR runner with preserved references and bounded GPU work."""

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT, LOCAL as PREVIOUS, PUBLIC as PREVIOUS_PUBLIC, identity, save, sha, validation_records, image_path
from .adaptive_config import approved_settings
from .adaptive import predict_adaptive
from .metrics import validate_predictions, distribution
from .sliced import make_adapter

LOCAL = ROOT/"outputs/adaptive_refinement"
PUBLIC = ROOT/"docs/evidence/adaptive_refinement_validation"


def source_identity():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in [ROOT/"src/wrist_eval"/name for name in ("adaptive.py", "adaptive_config.py", "adaptive_study.py")]}


# The checkpoint contract is injected; planning/fusion never sees a training-condition flag.
def load_verified_checkpoint(path, expected_sha, expected_lock):
    import torch
    from wrist_train.checkpoints import checked_bytes, check_payload
    if sha(path) != expected_sha:
        raise ValueError("Checkpoint content differs from approved run manifest")
    payload = torch.load(io.BytesIO(checked_bytes(path)), map_location="cpu", weights_only=False)
    check_payload(payload, expected_lock, require_resume=False)
    model = payload["ema"].float().eval().to("cuda:0")
    model.requires_grad_(False)
    from .common import CLASSES
    if model.names != dict(enumerate(CLASSES)):
        raise ValueError("Checkpoint class mapping changed")
    return model


class ViewRunner:
    def __init__(self, model):
        self.adapter = make_adapter(model)
        self.model = model
        if self.adapter.model is not model:
            raise ValueError("Adapter replaced the loaded checkpoint object")

    def __call__(self, image, net_shape):
        import torch
        from sahi.predict import get_prediction
        self.adapter.global_hw = image.shape[:2]
        self.adapter.global_net_shape = net_shape
        self.adapter.reset_counters()
        result = get_prediction(image, self.adapter, verbose=0)
        prediction = np.array([p.bbox.to_xyxy()+[p.score.value, p.category.id]
                               for p in result.object_prediction_list], dtype=np.float32).reshape(-1, 6)
        prediction = validate_predictions(prediction, image.shape[1], image.shape[0])[:300]
        if self.adapter.counters["forward_passes"] != 1:
            raise ValueError("Single-view wrapper made an unexpected model-call count")
        h, w = image.shape[:2]
        gain = 768/max(h, w)
        gains = (min(math.ceil(h*gain), 768)/h, min(math.ceil(w*gain), 768)/w)
        torch.cuda.synchronize()
        return {"predictions": prediction, "gains": gains, "counters": dict(self.adapter.counters)}


def references():
    active = json.loads((PREVIOUS/"reference_active.json").read_text())["directory"]
    native = json.loads((PREVIOUS/active/"predictions.json").read_text())
    serial_path = PREVIOUS/"full/standard_serial/predictions.jsonl"
    context_path = PREVIOUS/"full/sahi_context/predictions.jsonl"
    for variant, path in (("standard_serial", serial_path), ("sahi_context", context_path)):
        summary = json.loads((PREVIOUS_PUBLIC/f"full_{variant}.json").read_text())
        if sha(path) != summary["prediction_file_sha256"]:
            raise ValueError("Preserved comparator predictions changed")
    serial = {item["stem"]: np.asarray(item["pred"], dtype=np.float32).reshape(-1, 6)
              for item in (json.loads(line) for line in serial_path.read_text().splitlines())}
    shapes = {stem: value["net_shape"] for stem, value in native.items()}
    return serial, shapes


def private_record(stem, result, decode_seconds, wall_seconds, memory):
    def view_record(view):
        return {"predictions": view["predictions"].tolist(), "gains": list(view["gains"]),
                "counters": view["counters"], **({"roi_id": view["roi_id"]} if "roi_id" in view else {})}
    return {"stem": stem, "global": result["global"].tolist(), "plan": result["plan"],
            "global_view": view_record(result["global_view"]),
            "local_views": [view_record(v) for v in result["local_views"]],
            "variants": {v: {"pred": r["predictions"].tolist(), "diagnostics": r["diagnostics"]}
                         for v, r in result["variants"].items()},
            "timing": {"passes": result["passes"], "decode_seconds": decode_seconds,
                "common_seconds": result["common_seconds"], "planning_seconds": result["planning_seconds"],
                "fusion_seconds": result["fusion_seconds"], "joint_wall_seconds": wall_seconds,
                "e2e_composed_seconds": {v: decode_seconds+result["common_seconds"]+seconds
                                         for v, seconds in result["fusion_seconds"].items()}, **memory}}


def summarize(records):
    roi_counts = Counter(len(r["plan"]["rois"]) for r in records)
    return {"images": len(records), "roi_count_images": {str(i): roi_counts[i] for i in range(3)},
            "mean_passes": float(np.mean([r["timing"]["passes"] for r in records])),
            "total_passes": sum(r["timing"]["passes"] for r in records),
            "timing": {v: distribution([r["timing"]["e2e_composed_seconds"][v] for r in records]) for v in ("V1", "V2")},
            "joint_wall_seconds": distribution([r["timing"]["joint_wall_seconds"] for r in records]),
            "allocated_peak_bytes": max(r["timing"]["allocated_peak_bytes"] for r in records),
            "reserved_peak_bytes": max(r["timing"]["reserved_peak_bytes"] for r in records)}


def run(stage, attempt):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    from wrist_train.protocol import load_protocol
    cfg, manifest = approved_settings()
    tests = json.loads((PUBLIC/"automated_tests.json").read_text())
    if not tests["passed"] or tests["implementation_source_sha256"] != source_identity():
        raise ValueError("Current implementation has not passed the full automated gate")
    if Path(attempt).name != attempt or not attempt:
        raise ValueError("Invalid attempt name")
    if stage == "full":
        gate = json.loads((PUBLIC/"pilot_gate.json").read_text())
        if not gate["passed"] or gate["implementation_source_sha256"] != source_identity():
            raise ValueError("Current pilot/visual gate has not passed")
    directory = LOCAL/stage/attempt
    directory.mkdir(parents=True, exist_ok=False)
    require_cuda()
    init_seeds(cfg["global"]["seed"], deterministic=True)
    _, lock = load_protocol()
    serial, shapes = references()
    rows = validation_records()
    if stage == "pilot":
        selection = json.loads((PREVIOUS/"pilot/standard_serial/selection.json").read_text())
        by_stem = {r["filestem"]: r for r in rows}
        rows = [by_stem[stem] for stem in selection]
        if len(rows) != 24:
            raise ValueError("Frozen pilot does not contain 24 images")
    if set(shapes) != set(serial):
        raise ValueError("Global shape profile and comparator membership differ")
    save(directory/"shape_profile.json", shapes)
    save(directory/"selection.json", [row["filestem"] for row in rows])
    began = time.perf_counter()
    model = load_verified_checkpoint(ROOT/manifest["checkpoint"]["path"], manifest["checkpoint"]["sha256"], lock)
    runner = ViewRunner(model)
    load_seconds = time.perf_counter()-began
    began = time.perf_counter()
    for _ in range(3):
        runner(np.full((1024, 576, 3), 114, dtype=np.uint8), None)
    warmup_seconds = time.perf_counter()-began
    records = []
    loop_start = time.perf_counter()
    # Save every completed image immediately; failures never overwrite successful evidence.
    with (directory/"bundles.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Prepared image geometry changed")
            decoded = time.perf_counter()
            result = predict_adaptive(image, runner, shapes[row["filestem"]], cfg)
            torch.cuda.synchronize()
            wall = time.perf_counter()-started
            expected = serial[row["filestem"]]
            if not np.array_equal(result["global"], expected):
                delta = float(np.max(np.abs(result["global"]-expected))) if result["global"].shape == expected.shape else None
                raise ValueError(f"Global reference differs; max delta={delta}")
            counters = [result["global_view"]["counters"], *[v["counters"] for v in result["local_views"]]]
            if sum(c["forward_passes"] for c in counters) != result["passes"]:
                raise ValueError("Physical model-call accounting failed")
            record = private_record(row["filestem"], result, decoded-started, wall,
                {"allocated_peak_bytes": torch.cuda.max_memory_allocated(), "reserved_peak_bytes": torch.cuda.max_memory_reserved()})
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            if index % 100 == 0:
                print(f"{stage}: {index+1}/{len(rows)}; passes={result['passes']}", flush=True)
    loop_wall = time.perf_counter()-loop_start
    summary = {**summarize(records), "stage": stage, "attempt": attempt, "global_reference_exact": True,
        "model_load_seconds": load_seconds, "synthetic_warmup_seconds": warmup_seconds,
        "loop_wall_seconds_including_export": loop_wall, "shape_profile_sha256": sha(directory/"shape_profile.json"),
        "bundles_sha256": sha(directory/"bundles.jsonl"), "proposal_sha256": identity(cfg),
        "execution_manifest_sha256": identity(manifest), "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "implementation_source_sha256": source_identity(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "timing_definition": "Per-variant composed decode + all shared global/planning/local work + own fusion; no division of GPU cost",
        "no_test_evaluation": True, "no_training": True}
    if stage == "pilot":
        calibration, replay = [], []
        for row in rows[:6]:
            for variant in ("V1", "V2"):
                torch.cuda.synchronize()
                started = time.perf_counter()
                image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
                result = predict_adaptive(image, runner, shapes[row["filestem"]], cfg, [variant])
                torch.cuda.synchronize()
                calibration.append({"variant": variant, "seconds": time.perf_counter()-started,
                                    "passes": result["passes"]})
        for row, prior in zip(rows[:3], records[:3]):
            image = cv2.imdecode(np.fromfile(image_path(row), dtype=np.uint8), cv2.IMREAD_COLOR)
            result = predict_adaptive(image, runner, shapes[row["filestem"]], cfg)
            for variant in ("V1", "V2"):
                old = np.asarray(prior["variants"][variant]["pred"], dtype=np.float32).reshape(-1, 6)
                if not np.array_equal(result["variants"][variant]["predictions"], old):
                    raise ValueError("Pilot deterministic replay differs")
            replay.append({"image_ordinal": len(replay)+1, "exact": True, "passes": result["passes"]})
        save(directory/"standalone_calibration.json", calibration)
        summary["standalone_calibration"] = {v: distribution([r["seconds"] for r in calibration if r["variant"] == v]) for v in ("V1", "V2")}
        summary["standalone_calibration_images_per_variant"] = 6
        summary["deterministic_replay"] = replay
        summary["extra_calibration_passes"] = sum(r["passes"] for r in calibration)
        summary["extra_replay_passes"] = sum(r["passes"] for r in replay)
    save(directory/"summary.json", summary)
    save(PUBLIC/f"{stage}_summary.json", summary)
    save(LOCAL/f"{stage}_active.json", {"attempt": attempt})
    print({"stage": stage, "images": len(records), "roi_counts": summary["roi_count_images"],
           "mean_passes": summary["mean_passes"], "global_exact": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "full"])
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    run(args.stage, args.attempt)
