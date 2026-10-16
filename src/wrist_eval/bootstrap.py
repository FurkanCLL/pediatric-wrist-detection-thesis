"""Paired uncertainty estimates using patients as the resampling unit."""

from collections import defaultdict
import numpy as np
from .common import CLASSES, LOCAL, PUBLIC, config, save, validation_records
from .analyze import load_predictions, truth
from .metrics import ap_summary, native_stats, match


def patient_groups(rows):
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row["patient_id"]].append(i)
    return [groups[key] for key in sorted(groups)]


def resample_indices(groups,rng):
    # Every sampled patient contributes all examinations/views, with multiplicity.
    return [i for g in rng.integers(0,len(groups),size=len(groups)) for i in groups[g]]


def interval(values):
    values=np.asarray(values,dtype=float)
    return {"replicates_present":len(values),"median":float(np.median(values)),
            "lower":float(np.quantile(values,.025)),"upper":float(np.quantile(values,.975))} if len(values) else {"replicates_present":0}


def run():
    cfg=config();rows=validation_records();groups=patient_groups(rows)
    variants=list(cfg["variants"])
    predictions={v:load_predictions(v) for v in variants}
    stats={v:[native_stats(truth(r),predictions[v][r["filestem"]]) for r in rows] for v in variants}
    rng=np.random.default_rng(cfg["patient_bootstrap"]["seed"])
    results=[]
    for rep in range(cfg["patient_bootstrap"]["replicates"]):
        indices=resample_indices(groups,rng)
        sample={}
        for variant in variants:
            ap=ap_summary([stats[variant][i] for i in indices])
            # An absent class has undefined AP, not zero. Overall eight-class mAP is then omitted.
            sample[variant]={c:{k:ap["classes"][c][k] for k in ("ap50","ap75","ap50_95")}
                             for c in CLASSES if c in ap["classes"]}
            if len(ap["classes"])==8:
                sample[variant]["overall"]={k:ap[k] for k in ("map50","map75","map50_95")}
        results.append(sample)
        if rep%50==0:print(f"Patient bootstrap {rep+1}/{cfg['patient_bootstrap']['replicates']}",flush=True)
    save(LOCAL/"bootstrap_replicates.json",results)
    aggregates={};differences={}
    for variant in variants:
        aggregates[variant]={}
        for c in [*CLASSES,"overall"]:
            keys=("map50","map75","map50_95") if c=="overall" else ("ap50","ap75","ap50_95")
            aggregates[variant][c]={k:interval([s[variant][c][k] for s in results if c in s[variant]]) for k in keys}
        if variant!="standard_serial":
            differences[variant]={}
            for c in [*CLASSES,"overall"]:
                keys=("map50","map75","map50_95") if c=="overall" else ("ap50","ap75","ap50_95")
                differences[variant][c]={k:interval([s[variant][c][k]-s["standard_serial"][c][k] for s in results if c in s[variant] and c in s["standard_serial"]]) for k in keys}
    save(PUBLIC/"patient_bootstrap.json",{"unit":"patient, all views/examinations together","patients":len(groups),
        **cfg["patient_bootstrap"],"method":"paired nonparametric percentile; same patient draws for all variants; conditional on class presence",
        "limitations":["development data and validation-selected model","not corrected for multiple comparisons or method development",
                       "four bone-lesion patients cannot establish generalization; many resamples repeat the same objects"],
        "metrics":aggregates,"paired_ap_differences":differences})


if __name__=="__main__":run()
