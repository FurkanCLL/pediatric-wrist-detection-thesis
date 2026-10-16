"""Reconcile native network-space validation with common original-coordinate evaluation."""

import json
import numpy as np
from .common import CLASSES, LOCAL, PUBLIC, save, validation_records
from .analyze import truth
from .metrics import ap_summary, native_stats


def run():
    rows=validation_records()
    active=json.loads((LOCAL/"reference_active.json").read_text())["directory"]
    exports=json.loads((LOCAL/active/"predictions.json").read_text())
    stats=json.loads((LOCAL/active/"native_stats.json").read_text())
    # Reconstruct the native rectangular-loader order from frozen dimensions, without image access.
    order=np.argsort(np.array([r["height"]/r["width"] for r in rows]))
    ordered=[rows[i] for i in order]
    source_stats=[];export_gt_stats=[];differences=[]
    boundary_predictions=0
    native_counts=np.zeros((8,10),int);clipped_counts=np.zeros((8,10),int)
    for row,stat in zip(ordered,stats):
        gt=truth(row);value=exports[row["filestem"]]
        native_gt=np.asarray(value["gt_native_original"],dtype=np.float32).reshape(-1,5)
        if not np.array_equal(np.asarray(stat["target_cls"]),gt[:,4]):
            raise ValueError("Native image order reconstruction failed")
        differences.extend(np.abs(gt[:,:4]-native_gt[:,:4]).reshape(-1).tolist())
        pred=np.asarray(value["pred"],dtype=np.float32).reshape(-1,6)
        source=native_stats(gt,pred);export=native_stats(native_gt,pred)
        source_stats.append(source);export_gt_stats.append(export)
        boundary_predictions+=int(np.sum(np.any(pred[:,:2]==0,axis=1)|np.any(pred[:,[2,3]]==[row["width"],row["height"]],axis=1)))
        for c in range(8):
            native_counts[c]+=np.asarray(stat["tp"],dtype=bool).reshape(-1,10)[np.asarray(stat["pred_cls"])==c].sum(0)
            clipped_counts[c]+=source["tp"][source["pred_cls"]==c].sum(0)
    reference=json.loads((PUBLIC/"reference_reproduction.json").read_text())
    serial=json.loads((PUBLIC/"full_standard_serial.json").read_text())["metrics"]
    source_ap=ap_summary(source_stats);export_ap=ap_summary(export_gt_stats)
    result={"native_network_space":reference["native"],"same_native_predictions_clipped_original_source_gt":source_ap,
        "same_native_predictions_clipped_original_native_roundtrip_gt":export_ap,"serial_original_source_gt":serial,
        "max_ground_truth_roundtrip_coordinate_error_pixels":max(differences),
        "predictions_touching_original_image_boundary":boundary_predictions,
        "padding_only_removed":reference["padding_only_predictions_omitted_from_original_export"],
        "per_class_tp_by_iou_native":{c:native_counts[i].tolist() for i,c in enumerate(CLASSES)},
        "per_class_tp_by_iou_clipped_original":{c:clipped_counts[i].tolist() for i,c in enumerate(CLASSES)},
        "interpretation":"Original-coordinate evaluation clips image-boundary predictions; native trainer evaluates network-space boxes including padding. The same predictions already show this difference, before batch-one inference. Source versus roundtrip target differences and the remaining serial batch difference are quantified separately. No weights or thresholds changed.",
        "all_methods_common_evaluation":"Original frozen annotations, clipped valid original-coordinate outputs, same pinned AP evaluator",
        "test_image_access":False,"new_gpu_inference":False}
    save(PUBLIC/"evaluation_bridge.json",result)


if __name__=="__main__":run()
