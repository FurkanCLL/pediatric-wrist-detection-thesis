"""Descriptive errors, geometry strata and paired object recovery on validation."""

from collections import Counter, defaultdict
import csv
import json
import numpy as np

from .common import CLASSES, LOCAL, PUBLIC, config, save, validation_records
from .metrics import (ap_summary, distribution, ious, match, native_stats, paired_recovery,
                      size_bin, validate_predictions)


def load_predictions(variant):
    path=LOCAL/"full"/variant/"predictions.jsonl"
    items=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result={r["stem"]:np.asarray(r["pred"],dtype=np.float32).reshape(-1,6) for r in items}
    if len(items)!=len(result):raise ValueError("Duplicate prediction image")
    return result


def truth(row):
    return np.array([b["xyxy"]+[b["class_id"]] for b in row["boxes"]],dtype=np.float32).reshape(-1,5)


def error_types(gt,pred,confidence=.25,threshold=.5):
    g,p,matrix=match(gt,pred,threshold,confidence)
    gt_reason=np.where(g>=0,"TP","unassigned").astype(object)
    # Residual stages consume their own unmatched predictions/targets once; these are descriptive categories.
    used_p=p>=0
    for reason,min_conf,max_conf,min_iou,same_class in (
        ("classification",confidence,1.01,threshold,False),
        ("low_confidence",.001,confidence,threshold,True),
        ("localization",confidence,1.01,.1,True)):
        for pi in np.argsort(-pred[:,4],kind="stable"):
            if used_p[pi] or not min_conf<=pred[pi,4]<max_conf:continue
            eligible=gt_reason=="unassigned"
            eligible &= (gt[:,4]==pred[pi,5]) if same_class else (gt[:,4]!=pred[pi,5])
            scores=np.where(eligible,matrix[:,pi],-1)
            if len(scores) and scores.max()>=min_iou:
                gi=int(scores.argmax());gt_reason[gi]=reason;used_p[pi]=True
    gt_reason[gt_reason=="unassigned"]="no_residual_candidate"
    fp_reason=[]
    for pi in range(len(pred)):
        if pred[pi,4]<confidence:fp_reason.append("below_operating_confidence");continue
        if p[pi]>=0:fp_reason.append("TP");continue
        same=gt[:,4]==pred[pi,5]
        best_same=matrix[same,pi].max(initial=0)
        best_other=matrix[~same,pi].max(initial=0)
        fp_reason.append("duplicate" if best_same>=threshold else "classification" if best_other>=threshold else
                         "localization" if best_same>=.1 else "background_or_unannotated")
    return g,p,matrix,gt_reason,np.array(fp_reason,dtype=object)


def geometry(row,gt,index):
    b=gt[index,:4];w,h=row["width"],row["height"]
    bw,bh=b[2:]-b[:2];short=min(bw,bh)*768/max(w,h)
    overlap=ious(b[None],gt[:,:4])[0]
    overlap[index]=0
    # Non-text box overlap is only a geometry proxy, not an adjudication of anatomical difficulty.
    overlap[gt[:,4]==2]=0
    border=min(b[0],b[1],w-b[2],h-b[3])/min(w,h)
    relative_area=bw*bh/(w*h)
    return {"short_768":float(short),"short_original":float(min(bw,bh)),"area_original":float(bw*bh),
        "relative_area":float(relative_area),"border_fraction":float(border),
        "size":str(size_bin(short)),"area":"<.005" if relative_area<.005 else ".005-.02" if relative_area<.02 else ">=.02",
        "border":"<.02" if border<.02 else ".02-.1" if border<.1 else ">=.1",
        "aspect":"<.4" if w/h<.4 else ".4-.6" if w/h<.6 else ".6-.8" if w/h<.8 else ">=.8",
        "long_side":"<1024" if max(w,h)<1024 else "1024-1535" if max(w,h)<1536 else ">=1536",
        "overlap_nontext":str(bool(np.any(overlap>.1))),"multiple_same_class":str(sum(gt[:,4]==gt[index,4])>1),
        "objects_per_image":str(min(4,len(gt)))}


def describe(rows,predictions,confidence=.25,threshold=.5):
    per_class={c:{"objects":0,"predictions":0,"tp":0,"fp":0,"fn":0,"gt_errors":Counter(),"fp_errors":Counter(),
                   "tp_conf":[],"fp_conf":[],"all_conf":[],"matched_iou":[],"patients":set(),"images":set(),
                   "fn_candidate_presence":Counter()} for c in CLASSES}
    confusion=np.zeros((9,9),int)
    strata=defaultdict(lambda:{"objects":0,"tp":0,"patients":set(),"iou":[]})
    details={}
    for row in rows:
        gt=truth(row);pred=validate_predictions(predictions[row["filestem"]],row["width"],row["height"])
        g,p,matrix,reasons,fp_reasons=error_types(gt,pred,confidence,threshold)
        ag,ap,_=match(gt,pred,threshold,confidence,class_aware=False)
        for gi,pi in enumerate(ag):confusion[int(gt[gi,4]),int(pred[pi,5]) if pi>=0 else 8]+=1
        for pi,gi in enumerate(ap):
            if pred[pi,4]>=confidence and gi<0:confusion[8,int(pred[pi,5])]+=1
        geometries=[]
        for gi,box in enumerate(gt):
            c=CLASSES[int(box[4])];v=per_class[c]
            v["objects"]+=1;v["patients"].add(row["patient_id"]);v["images"].add(row["filestem"])
            v["gt_errors"][reasons[gi]]+=1
            if g[gi]>=0:v["tp"]+=1;v["matched_iou"].append(float(matrix[gi,g[gi]]))
            else:
                v["fn"]+=1
                # These are candidate-presence diagnostics, not extra matches or recovered detections.
                same=pred[:,5]==box[4]
                for floor in (.001,.05,.25):
                    available=pred[:,4]>=floor
                    v["fn_candidate_presence"][f"no_any_class_iou_ge_0.1_conf_{floor}"]+=int(not np.any((matrix[gi]>=.1)&available))
                    v["fn_candidate_presence"][f"no_same_class_iou_ge_0.1_conf_{floor}"]+=int(not np.any((matrix[gi]>=.1)&available&same))
            geo=geometry(row,gt,gi);geometries.append(geo)
            for dimension in ("size","area","border","aspect","long_side","overlap_nontext","multiple_same_class","objects_per_image"):
                key=(c,dimension,geo[dimension]);s=strata[key]
                s["objects"]+=1;s["tp"]+=int(g[gi]>=0);s["patients"].add(row["patient_id"])
                if g[gi]>=0:s["iou"].append(float(matrix[gi,g[gi]]))
        for pi,box in enumerate(pred):
            v=per_class[CLASSES[int(box[5])]];v["all_conf"].append(float(box[4]))
            if box[4]<confidence:continue
            v["predictions"]+=1
            if p[pi]>=0:v["tp_conf"].append(float(box[4]))
            else:v["fp"]+=1;v["fp_conf"].append(float(box[4]));v["fp_errors"][fp_reasons[pi]]+=1
        details[row["filestem"]]={"g_to_p":g.tolist(),"p_to_g":p.tolist(),"gt_reasons":reasons.tolist(),"fp_reasons":fp_reasons.tolist(),
                                 "geometry":geometries,"matched_iou":[float(matrix[gi,pi]) if pi>=0 else None for gi,pi in enumerate(g)]}
    for c,v in per_class.items():
        v["precision"]=v["tp"]/(v["tp"]+v["fp"]) if v["tp"]+v["fp"] else 0.
        v["recall"]=v["tp"]/v["objects"] if v["objects"] else 0.
        v["fp_per_image"]=v["fp"]/len(rows)
        for key in ("tp_conf","fp_conf","all_conf","matched_iou"):v[key]=distribution(v[key])
        for key in ("patients","images"):v[key]=len(v[key])
        assert v["tp"]+v["fn"]==v["objects"] and v["tp"]+v["fp"]==v["predictions"]
    stratified=[]
    for (c,dimension,label),v in sorted(strata.items()):
        stratified.append({"class":c,"dimension":dimension,"stratum":label,"objects":v["objects"],"patients":len(v["patients"]),
                           "tp":v["tp"],"fn":v["objects"]-v["tp"],"recall":v["tp"]/v["objects"],"matched_iou":distribution(v["iou"])})
    return {"confidence":confidence,"iou":threshold,"classes":per_class,"class_agnostic_confusion_gt_rows_pred_columns_background_last":confusion.tolist(),
            "total_tp":sum(v["tp"] for v in per_class.values()),"total_fp":sum(v["fp"] for v in per_class.values()),
            "total_fn":sum(v["fn"] for v in per_class.values()),"strata":stratified},details


def compare(rows,a,b):
    result={}
    for threshold in (.5,.75):
        classes={c:Counter() for c in CLASSES};sizes={c:defaultdict(Counter) for c in CLASSES}
        deltas={c:[] for c in CLASSES};signals={c:Counter() for c in CLASSES}
        for row in rows:
            gt=truth(row);pa=a[row["filestem"]];pb=b[row["filestem"]]
            pair=paired_recovery(gt,pa,pb,threshold=threshold)
            baseline_reasons=error_types(gt,pa,threshold=threshold)[3]
            for gi,g in enumerate(gt):
                c=CLASSES[int(g[4])];size=str(size_bin(geometry(row,gt,gi)["short_768"]))
                for key in ("both","recovered","lost","neither"):
                    classes[c][key]+=int(pair[key][gi]);sizes[c][size][key]+=int(pair[key][gi])
                if pair["both"][gi]:deltas[c].append(float(pair["iou_b"][gi]-pair["iou_a"][gi]))
                if pair["recovered"][gi]:
                    same=pa[:,5]==g[4]
                    overlap=ious(g[None,:4],pa[:,:4])[0]
                    signals[c]["recovered"]+=1
                    signals[c]["baseline_reason_"+baseline_reasons[gi]]+=1
                    signals[c]["any_same_class_iou_ge_0.1_at_0.001"]+=int(np.any(overlap[same]>=.1))
                    signals[c]["any_same_class_iou_ge_0.1_at_0.05"]+=int(np.any((overlap>=.1)&same&(pa[:,4]>=.05)))
                    signals[c]["intermediate_confidence_same_class_nearby"]+=int(np.any((overlap>=.1)&same&(pa[:,4]>=.05)&(pa[:,4]<.5)))
                    shorts=np.minimum(pa[:,2]-pa[:,0],pa[:,3]-pa[:,1])*768/max(row["height"],row["width"])
                    signals[c]["small_same_class_prediction_nearby_at_0.05"]+=int(np.any((overlap>=.1)&same&(pa[:,4]>=.05)&(shorts<64)))
                    signals[c]["no_prediction_any_class_iou_ge_0.1_at_0.05"]+=int(not np.any((overlap>=.1)&(pa[:,4]>=.05)))
        result[str(threshold)]={"classes":classes,"size_changes":{c:dict(v) for c,v in sizes.items()},
                               "both_detected_iou_delta":{c:distribution(v) for c,v in deltas.items()},"recovered_first_pass_signals":signals}
    return result


def compare_false_positives(rows,a,b):
    # Spatial FP correspondence is separate from target recovery; it cannot claim extra ground-truth detections.
    classes={c:Counter() for c in CLASSES}
    for row in rows:
        gt=truth(row);pa=a[row["filestem"]];pb=b[row["filestem"]]
        _,ma,_=match(gt,pa);_,mb,_=match(gt,pb)
        fa=pa[(ma<0)&(pa[:,4]>=.25)];fb=pb[(mb<0)&(pb[:,4]>=.25)]
        targets=np.column_stack((fa[:,:4],fa[:,5]))
        ga,pb_matches,_=match(targets,fb)
        for c,name in enumerate(CLASSES):
            sa=fa[:,5]==c;sb=fb[:,5]==c
            classes[name]["standard_fp"]+=int(sa.sum());classes[name]["sahi_fp"]+=int(sb.sum())
            classes[name]["shared_fp"]+=int(np.sum(sa&(ga>=0)))
            classes[name]["removed_fp"]+=int(np.sum(sa&(ga<0)))
            classes[name]["introduced_fp"]+=int(np.sum(sb&(pb_matches<0)))
    return {"confidence":.25,"correspondence_iou":.5,"same_class":True,"classes":classes,
            "interpretation":"One-to-one spatial correspondence between each method's unmatched predictions; not a clinical adjudication or an identity match for arbitrarily shifted boxes"}


def run():
    rows=validation_records();cfg=config()
    predictions={v:load_predictions(v) for v in cfg["variants"]}
    if any(set(p)!={r["filestem"] for r in rows} for p in predictions.values()):raise ValueError("Prediction membership differs")
    result={"scope":"Full validation development evidence; validation-selected baseline; no test evaluation","images":len(rows),"patients":914,
            "definitions":{"size_bins_short_768":["<16","16-<32","32-<64","64-<128",">=128"],
              "gt_error_priority":["TP","residual wrong-class IoU>=threshold","residual same-class low-confidence IoU>=threshold","residual same-class IoU>=.1","no residual candidate"],
              "descriptive_matching":"Confidence-ordered best available target, one-to-one, stable source-order ties",
            "ap":"Pinned Ultralytics BaseValidator.match_predictions + ap_per_class, float32 IoU, 101-point trapezoidal envelope"},"variants":{},"comparisons":{},"fp_comparisons":{}}
    for variant,pred in predictions.items():
        aps=ap_summary([native_stats(truth(r),pred[r["filestem"]]) for r in rows])
        points={}
        for conf in cfg["operating_confidences"]:
            for threshold in cfg["descriptive_iou"]:
                counts,details=describe(rows,pred,conf,threshold)
                key=f"conf{conf}_iou{threshold}";points[key]=counts
                if conf==.25 and threshold==.5:save(LOCAL/"analysis"/f"{variant}_details.json",details)
        result["variants"][variant]={"ap":aps,"operating_points":points}
    for variant in ("sahi_context","sahi_detail"):
        result["comparisons"][variant]=compare(rows,predictions["standard_serial"],predictions[variant])
        result["fp_comparisons"][variant]=compare_false_positives(rows,predictions["standard_serial"],predictions[variant])
    save(PUBLIC/"error_analysis.json",result)
    with (PUBLIC/"class_comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["variant","class","ap50","ap75","ap50_95","tp","fp","fn","precision","recall","fp_per_image"])
        writer.writeheader()
        for variant,v in result["variants"].items():
            for c in CLASSES:
                counts=v["operating_points"]["conf0.25_iou0.5"]["classes"][c]
                writer.writerow({"variant":variant,"class":c,**{k:v["ap"]["classes"][c][k] for k in ("ap50","ap75","ap50_95")},
                                 **{k:counts[k] for k in ("tp","fp","fn","precision","recall","fp_per_image")}})
    print({v:r["ap"]["map50_95"] for v,r in result["variants"].items()},flush=True)


if __name__=="__main__":run()
