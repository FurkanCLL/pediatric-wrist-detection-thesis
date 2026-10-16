"""Independent result accounting, bounded determinism replay and final preservation."""

import json
import hashlib
import subprocess
import numpy as np

from .common import (CLASSES, ROOT, LOCAL, PUBLIC, config, identity, image_path, load_model,
                     save, sha, validation_records)
from .analyze import load_predictions, truth
from .metrics import validate_predictions, match
from .provenance import environment_receipt, final_checks


def accounting():
    cfg=config();rows=validation_records()
    analysis=json.loads((PUBLIC/"error_analysis.json").read_text())
    checks={}
    for variant in cfg["variants"]:
        source=LOCAL/"full"/variant/"predictions.jsonl"
        summary=json.loads((PUBLIC/f"full_{variant}.json").read_text())
        items=[json.loads(line) for line in source.read_text().splitlines()]
        predictions=load_predictions(variant)
        if summary["prediction_file_sha256"]!=sha(source) or summary["config_sha256"]!=identity(cfg):
            raise ValueError("Prediction file or configuration changed")
        if len(items)!=len(rows) or set(predictions)!={r["filestem"] for r in rows}:
            raise ValueError("Incomplete validation predictions")
        tp=fp=fn=0
        for row,item in zip(rows,items):
            if row["filestem"]!=item["stem"]:raise ValueError("Image ordering changed")
            pred=validate_predictions(predictions[item["stem"]],row["width"],row["height"])
            gt=truth(row);g,p,_=match(gt,pred)
            tp+=int(sum(g>=0));fn+=int(sum(g<0));fp+=int(sum((p<0)&(pred[:,4]>=.25)))
            timing=item["timing"]
            expected=1 if variant=="standard_serial" else 5 if variant=="sahi_context" else 10
            if timing["forward_passes"]!=expected or timing["gpu_forward_ms"]<=0 or timing["seconds"]>timing["e2e_seconds"]:
                raise ValueError("Invalid timing/forward accounting")
            if len(pred)>300 or not all(np.isfinite(timing[k]) and timing[k]>=0 for k in ("seconds","decode_seconds","preprocess_seconds","nms_restore_seconds")):
                raise ValueError("Invalid predictions/timings")
        recorded=analysis["variants"][variant]["operating_points"]["conf0.25_iou0.5"]
        if (tp,fp,fn)!=(recorded["total_tp"],recorded["total_fp"],recorded["total_fn"]):
            raise ValueError("Descriptive counts do not reconcile")
        checks[variant]={"images":len(items),"objects":tp+fn,"tp":tp,"fp":fp,"fn":fn,
                         "forward_passes":sum(i["timing"]["forward_passes"] for i in items),"valid_finite_outputs":True}
    for variant,pair in analysis["comparisons"].items():
        for threshold,details in pair.items():
            for c,counts in details["classes"].items():
                support=analysis["variants"]["standard_serial"]["operating_points"]["conf0.25_iou0.5"]["classes"][c]["objects"]
                if sum(counts.values())!=support:raise ValueError("Recovery groups do not partition support")
    save(PUBLIC/"accounting_verification.json",{"variants":checks,"recovery_partitions_valid":True,"test_evaluation":False})


def replay():
    import cv2
    from ultralytics.utils.torch_utils import init_seeds
    from wrist_train.common import require_cuda
    from .feasibility import pilot_rows
    from .sliced import make_adapter,predict
    require_cuda();init_seeds(42,deterministic=True)
    rows=pilot_rows(validation_records())[:3]
    active=json.loads((LOCAL/"reference_active.json").read_text())["directory"]
    reference=json.loads((LOCAL/active/"predictions.json").read_text())
    adapter=make_adapter(load_model());result={}
    for variant in config()["variants"]:
        expected=load_predictions(variant);differences=[]
        for row in rows:
            image=cv2.imdecode(np.fromfile(image_path(row),dtype=np.uint8),cv2.IMREAD_COLOR)
            actual,_=predict(image,adapter,variant,reference[row["filestem"]]["net_shape"])
            prior=expected[row["filestem"]]
            if actual.shape!=prior.shape:raise ValueError("Replay prediction count changed")
            delta=float(np.max(np.abs(actual-prior),initial=0))
            if delta>1e-6:raise ValueError("Deterministic prediction replay changed")
            differences.append(delta)
        result[variant]={"images":len(rows),"max_absolute_array_difference":max(differences),"passed":True}
    save(PUBLIC/"determinism_replay.json",{"selection":"first three outcome-independent pilot selections","variants":result,
                                         "purpose":"prediction identity check, not a repeated performance benchmark"})


def run():
    accounting()
    replay()
    environment_receipt()
    final_checks()
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest() for p in sorted((ROOT/"src/wrist_eval").glob("*.py"))}
    save(PUBLIC/"implementation_provenance.json",{"normalized_source_sha256":sources,
        "config_sha256":identity(config()),"checkpoint_sha256":config()["checkpoint_sha256"],
        "implementation_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "private_prediction_hashes":{v:sha(LOCAL/"full"/v/"predictions.jsonl") for v in config()["variants"]}})


if __name__=="__main__":run()
