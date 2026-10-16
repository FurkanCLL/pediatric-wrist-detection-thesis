"""Outcome-independent pilot selection and bounded validation inference."""

import argparse
import json
import time
import numpy as np

from .common import (CLASSES, LOCAL, PUBLIC, config, identity, image_path, load_model, save, sha,
                     validation_records, baseline_inventory, source_snapshot)
from .metrics import distribution, native_stats, ap_summary
from .sliced import make_adapter, predict, tile_dimensions


def pilot_rows(rows):
    # Choose geometry extremes and class/size examples without looking at predictions.
    ranked = sorted(rows, key=lambda r: identity({"seed":42,"stem":r["filestem"]}))
    chosen = []
    def add(row):
        if row not in chosen:
            chosen.append(row)
    for key in (lambda r:r["width"]*r["height"],lambda r:r["width"]/r["height"]):
        add(min(rows,key=key)); add(max(rows,key=key))
    for c in range(8):
        eligible = [r for r in ranked if any(b["class_id"]==c for b in r["boxes"])]
        add(eligible[0])
        add(min(eligible,key=lambda r:min(min(b["xyxy"][2]-b["xyxy"][0],b["xyxy"][3]-b["xyxy"][1])*768/max(r["height"],r["width"]) for b in r["boxes"] if b["class_id"]==c)))
    add(max(rows,key=lambda r:len(r["boxes"])))
    for row in ranked:
        if len(chosen) >= 24:
            break
        if row["patient_id"] not in {r["patient_id"] for r in chosen}:
            add(row)
    return chosen[:24]


def summarize(timings):
    return {key:distribution([t[key] for t in timings]) for key in
            ("seconds","e2e_seconds","decode_seconds","gpu_forward_ms","preprocess_seconds","model_wall_seconds",
             "nms_restore_seconds","slices","forward_passes","pre_merge_predictions","post_merge_predictions",
             "allocated_peak_bytes","reserved_peak_bytes","padding_only_predictions")} | {
            "throughput_images_per_second":len(timings)/sum(t["e2e_seconds"] for t in timings),
            "merge_seconds":distribution([t["sahi_durations"].get("postprocess",0) for t in timings]),
            "slice_seconds":distribution([t["sahi_durations"].get("slice",0) for t in timings])}


def run(stage, variant):
    import cv2
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    require_cuda()
    init_seeds(42,deterministic=True)
    cfg=config()
    rows=validation_records()
    active=json.loads((LOCAL/"reference_active.json").read_text())["directory"]
    reference=json.loads((LOCAL/active/"predictions.json").read_text())
    if not json.loads((PUBLIC/"reference_reproduction.json").read_text())["reproduced"]:
        raise RuntimeError("Native reference has not passed")
    if stage=="pilot":
        rows=pilot_rows(rows)
    elif not (PUBLIC/"pilot_gate.json").exists() or not json.loads((PUBLIC/"pilot_gate.json").read_text())["passed"]:
        raise RuntimeError("Pilot gate must pass before full validation")
    directory=LOCAL/stage/variant
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/"selection.json",[r["filestem"] for r in rows])
    started=time.perf_counter()
    model=load_model()
    adapter=make_adapter(model)
    load_seconds=time.perf_counter()-started
    # A fixed synthetic warmup keeps initialization out of steady-state patient timings.
    warmup_start=time.perf_counter()
    for _ in range(3):
        adapter.perform_batch_inference([np.full((1024,576,3),114,dtype=np.uint8)])
    torch.cuda.synchronize()
    warmup_seconds=time.perf_counter()-warmup_start
    stats,timings=[],[]
    loop_started=time.perf_counter()
    with (directory/"predictions.jsonl").open("x",encoding="utf-8") as stream:
        for i,row in enumerate(rows):
            start=time.perf_counter()
            image=cv2.imdecode(np.fromfile(image_path(row),dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[:2]!=(row["height"],row["width"]):
                raise ValueError("Prepared image geometry changed")
            decoded=time.perf_counter()
            prediction,timing=predict(image,adapter,variant,reference[row["filestem"]]["net_shape"])
            timing["decode_seconds"]=decoded-start
            timing["e2e_seconds"]=time.perf_counter()-start
            gt=np.array([b["xyxy"]+[b["class_id"]] for b in row["boxes"]],dtype=np.float32).reshape(-1,5)
            stats.append(native_stats(gt,prediction)); timings.append(timing)
            stream.write(json.dumps({"stem":row["filestem"],"pred":prediction.tolist(),"timing":timing},allow_nan=False)+"\n")
            stream.flush()
            if i%100==0:
                print(f"{stage}/{variant}: {i+1}/{len(rows)}",flush=True)
    loop_wall_seconds=time.perf_counter()-loop_started
    result={"stage":stage,"variant":variant,"images":len(rows),"patients":len({r["patient_id"] for r in rows}),
            "config_sha256":identity(cfg),"checkpoint_sha256":cfg["checkpoint_sha256"],"metrics":ap_summary(stats),
            "timing":summarize(timings),"model_load_seconds":load_seconds,"synthetic_warmup_seconds":warmup_seconds,
            "loop_wall_seconds_including_metric_export":loop_wall_seconds,
            "prediction_file_sha256":sha(directory/"predictions.jsonl"),"no_test_evaluation":True}
    save(directory/"summary.json",result)
    save(PUBLIC/f"{stage}_{variant}.json",result)
    print({"stage":stage,"variant":variant,"map":result["metrics"]["map50_95"],"seconds_mean":result["timing"]["seconds"]["mean"]},flush=True)


def gate():
    cfg=config()
    results={v:json.loads((PUBLIC/f"pilot_{v}.json").read_text()) for v in cfg["variants"]}
    checks={v:{"predicted_full_seconds":r["timing"]["e2e_seconds"]["mean"]*3088,
               "peak_allocated_gib":r["timing"]["allocated_peak_bytes"]["max"]/2**30,
               "max_slices":r["timing"]["slices"]["max"]} for v,r in results.items()}
    accounting={}
    for variant in cfg["variants"]:
        items=[json.loads(line) for line in (LOCAL/"pilot"/variant/"predictions.jsonl").read_text().splitlines()]
        expected=1 if variant=="standard_serial" else 5 if variant=="sahi_context" else 10
        accounting[variant]=all(item["timing"]["forward_passes"]==expected
            and item["timing"]["gpu_forward_ms"]>0
            and all(np.isfinite(item["timing"][key]) and item["timing"][key]>=0 for key in
                    ("seconds","e2e_seconds","decode_seconds","gpu_forward_ms","preprocess_seconds","nms_restore_seconds"))
            and item["timing"]["post_merge_predictions"]<=cfg["max_det"] for item in items)
    passed=all(accounting.values()) and all(c["predicted_full_seconds"]<cfg["full_run_limits"]["max_predicted_seconds_per_variant"]
               and c["peak_allocated_gib"]<cfg["full_run_limits"]["pilot_peak_allocated_gib"] for c in checks.values())
    result={"passed":passed,"checks":checks,"forward_and_timing_accounting":accounting,"config_sha256":identity(cfg),"selection_rule":"Geometry extremes, class-first hash ranks, smallest class boxes, most annotations, then new patients; no model outcomes",
            "selection_images":24,"technical_outputs_validated":True,"scope":"validation development only"}
    save(PUBLIC/"pilot_gate.json",result)
    print(result)
    if not passed:
        raise RuntimeError("Full-run budget gate failed; define and document patient-aware fallback before continuing")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["pilot","full","gate"])
    parser.add_argument("variant",nargs="?",choices=list(config()["variants"]))
    args=parser.parse_args()
    if args.stage=="gate": gate()
    elif args.variant: run(args.stage,args.variant)
    else: parser.error("variant required")
