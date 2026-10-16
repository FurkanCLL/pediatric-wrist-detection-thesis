"""Reproduce the trainer's validation path without invoking a trainer or writing its run."""

import json
from pathlib import Path
import time

from .common import (CLASSES, LOCAL, PREPARED, PUBLIC, config, identity, load_model,
                     save, source_snapshot, validation_records, verify_baseline, verify_validation_files)
from .metrics import ap_summary, native_stats, validate_predictions


def run():
    import numpy as np
    import torch
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda

    directory = LOCAL / "reference"
    directory.mkdir(parents=True, exist_ok=False)
    require_cuda()
    init_seeds(42, deterministic=True)
    cfg = config()
    receipt = verify_baseline()
    rows = validation_records()
    before = source_snapshot()
    save(LOCAL / "source_snapshot_before.json", before)
    files = verify_validation_files(rows)
    data = {"val": str(PREPARED / "images/val"), "nc": 8, "channels": 3, "names": dict(enumerate(CLASSES))}
    validator = DetectionValidator(save_dir=directory, args={"imgsz": 768, "conf": .001, "iou": .7,
        "max_det": 300, "batch": 12, "workers": 4, "rect": True, "plots": False, "save_json": False,
        "save_txt": False, "augment": False, "device": "0", "quantize": 16, "compile": False})
    validator.data, validator.device, validator.stride = data, torch.device("cuda:0"), 32
    validator.training = True
    # A stale label cache must fail rather than write into the frozen prepared dataset.
    class ReadOnlyDataset(YOLODataset):
        def cache_labels(self, path):
            raise RuntimeError("Frozen validation cache missing or stale; no dataset writes allowed")
    dataset = ReadOnlyDataset(img_path=data["val"], data=data, imgsz=768, batch_size=12,
        augment=False, hyp=validator.args, rect=True, cache=False, stride=32, pad=.5)
    if {Path(p).stem for p in dataset.im_files} != {r["filestem"] for r in rows}:
        raise ValueError("Native loader changed validation membership")
    # This loader instance stays in-process on Windows; its dataset class forbids cache writes.
    loader = build_dataloader(dataset, 12, 0, shuffle=False, rank=-1, pin_memory=True, device=validator.device)
    model = load_model()
    validator.init_metrics(model)
    native, original, outputs = [], [], {}
    shape_counts = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            batch = validator.preprocess(batch)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = model(batch["img"], augment=False)
            if not all(torch.isfinite(t).all() for t in raw if isinstance(t, torch.Tensor)):
                raise FloatingPointError("Non-finite reference model outputs")
            predictions = validator.postprocess(raw)
            for si, pred in enumerate(predictions):
                pbatch = validator._prepare_batch(si, batch)
                stem = Path(pbatch["im_file"]).stem
                gt_net = torch.cat((pbatch["bboxes"], pbatch["cls"][:, None]), 1).cpu().numpy()
                p_net = torch.cat((pred["bboxes"], pred["conf"][:, None], pred["cls"][:, None]), 1).float().cpu().numpy()
                native.append(native_stats(gt_net, p_net))
                pred_orig = ops.scale_boxes(pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"])
                gt_orig = ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"])
                gt = torch.cat((gt_orig, pbatch["cls"][:, None]), 1).float().cpu().numpy()
                arr = torch.cat((pred_orig, pred["conf"][:, None], pred["cls"][:, None]), 1).float().cpu().numpy()
                arr = validate_predictions(arr, pbatch["ori_shape"][1], pbatch["ori_shape"][0])
                original.append(native_stats(gt, arr))
                outputs[stem] = {"pred": arr.tolist(), "gt_native_original": gt.tolist(), "net_shape": list(pbatch["imgsz"]),
                                 "ratio_pad": pbatch["ratio_pad"]}
                key = "x".join(map(str, pbatch["imgsz"]))
                shape_counts[key] = shape_counts.get(key, 0) + 1
            if bi % 50 == 0:
                print(f"Reference batch {bi+1}/{len(loader)}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    native_result, original_result = ap_summary(native), ap_summary(original)
    # Native CSV is rounded to five decimals; allow only half a final decimal plus numerical noise.
    differences = {k: native_result[k] - receipt["metrics"][m] for k, m in
                   (("map50", "metrics/mAP50(B)"), ("map50_95", "metrics/mAP50-95(B)"))}
    save(directory / "predictions.json", outputs)
    save(directory / "native_stats.json", [{k: v.tolist() for k, v in s.items()} for s in native])
    result = {"native": native_result, "original_coordinates": original_result, "difference_from_recorded": differences,
        "tolerance": 0.0000051, "reproduced": all(abs(v) <= .0000051 for v in differences.values()),
        "seconds": seconds, "images": len(outputs), "objects": sum(len(o["gt_native_original"]) for o in outputs.values()),
        "config_sha256": identity(cfg), "checkpoint_sha256": receipt["checkpoint_sha256"], "prepared_file_checks": files,
        "weights": "FP32 EMA, unfused", "inputs": "FP16 / 255", "autocast": "CUDA float16",
        "batch": 12, "workers_actual": 0, "workers_official": 4, "rect": True, "pad": .5,
        "workers_reason": "Read-only dataset subclass remains in-process; ordering/pixels/batches unchanged",
        "batch_shapes": shape_counts, "effective_validator_args": vars(validator.args),
        "dataset_unchanged": source_snapshot() == before}
    # Absolute framework defaults are retained only in the ignored run receipt.
    save(directory / "receipt.json", result)
    public = {k: v for k, v in result.items() if k != "effective_validator_args"}
    save(PUBLIC / "reference_reproduction.json", public)
    if not result["reproduced"] or not result["dataset_unchanged"]:
        raise RuntimeError("Reference reproduction/integrity gate failed; inspect preserved evidence")
    print({"native": native_result, "differences": differences, "seconds": seconds}, flush=True)


if __name__ == "__main__":
    run()
