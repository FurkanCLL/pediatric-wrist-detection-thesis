# CG-ALR V3 — experimental results

One frozen V3.0 algorithm, two predeclared checkpoints, 3,088 validation images/914 patients/7,204 objects. These are exploratory validation findings, not independent confirmation. Numerical configuration and both design copies were committed/pushed before V3 performance evaluation; [freeze](freeze.json), [execution manifest](execution_manifest.json) and [verification](verification.json) bind the inputs and code. Historical A–F, Phase 5A and all dataset/checkpoint files remain unchanged.

## Two AP tracks

| Track | Method | mAP50 | mAP75 | mAP50–95 | Fracture AP50 | Fracture AP75 | Fracture AP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical | A | 0.59900535 | 0.30227745 | 0.33386896 | 0.92117499 | 0.53459554 | 0.51973720 |
| historical | C | 0.59861937 | 0.30236096 | 0.33375909 | 0.92132656 | 0.53497564 | 0.51989787 |
| historical | V3_standard | 0.59908506 | 0.30231104 | 0.33391967 | 0.92145901 | 0.53468957 | 0.51975903 |
| historical | D | 0.61184505 | 0.31556154 | 0.33755131 | 0.91619815 | 0.52567633 | 0.51203447 |
| historical | F | 0.61208287 | 0.31575230 | 0.33769342 | 0.91661803 | 0.52591099 | 0.51237914 |
| historical | V3_rarity | 0.61231529 | 0.31562591 | 0.33769194 | 0.91669413 | 0.52586924 | 0.51229119 |
| deterministic | A | 0.59884706 | 0.30230177 | 0.33382021 | 0.92114027 | 0.53481164 | 0.51968397 |
| deterministic | C | 0.59840806 | 0.30232835 | 0.33366909 | 0.92129661 | 0.53504073 | 0.51987490 |
| deterministic | V3_standard | 0.59896629 | 0.30227904 | 0.33382828 | 0.92140456 | 0.53483246 | 0.51974356 |
| deterministic | D | 0.61191522 | 0.31563576 | 0.33758845 | 0.91624741 | 0.52565656 | 0.51201845 |
| deterministic | F | 0.61166985 | 0.31566521 | 0.33757827 | 0.91658717 | 0.52586886 | 0.51230326 |
| deterministic | V3_rarity | 0.61205655 | 0.31562781 | 0.33761271 | 0.91662569 | 0.52571317 | 0.51221780 |

Historical uses the unchanged pinned matcher/evaluator and original final row order. Deterministic sensitivity restores original global-row order before the same native per-image matching and uses descending unchanged score, frozen image index and original global ID for AP ties. It retains the same ten IoUs and integration. This second track does not overwrite historical results. Inference has no GT input; annotation matching is post-inference evaluation only. All eight classes and both tracks are in [AP.csv](AP.csv).

## Paired uncertainty

| Track | Contrast | Metric | Delta, pp | Paired 95%, pp | Valid replicates |
| --- | --- | --- | --- | --- | --- |
| historical | V3_standard-A | overall | +0.00507 | [-0.01448, +0.01543] | 492 |
| historical | V3_standard-A | fracture | +0.00218 | [-0.01082, +0.03064] | 500 |
| historical | V3_standard-C | overall | +0.01606 | [-0.00484, +0.03550] | 492 |
| historical | V3_standard-C | fracture | -0.01388 | [-0.03305, +0.00772] | 500 |
| historical | V3_rarity-D | overall | +0.01406 | [-0.01719, +0.02059] | 492 |
| historical | V3_rarity-D | fracture | +0.02567 | [+0.00307, +0.03693] | 500 |
| historical | V3_rarity-F | overall | -0.00015 | [-0.02543, +0.02151] | 492 |
| historical | V3_rarity-F | fracture | -0.00880 | [-0.04910, +0.01050] | 500 |
| deterministic | V3_standard-A | overall | +0.00081 | [-0.00282, +0.00575] | 492 |
| deterministic | V3_standard-A | fracture | +0.00596 | [-0.00386, +0.02122] | 500 |
| deterministic | V3_standard-C | overall | +0.01592 | [+0.00194, +0.03246] | 492 |
| deterministic | V3_standard-C | fracture | -0.01313 | [-0.02808, +0.00252] | 500 |
| deterministic | V3_rarity-D | overall | +0.00243 | [-0.00339, +0.01192] | 492 |
| deterministic | V3_rarity-D | fracture | +0.01993 | [+0.00674, +0.03310] | 500 |
| deterministic | V3_rarity-F | overall | +0.00344 | [-0.01141, +0.01657] | 492 |
| deterministic | V3_rarity-F | fracture | -0.00855 | [-0.05033, +0.00795] | 500 |

500 identical patient draws, seed 314159, all views/examinations with multiplicity. Historical A/C/D/F replicates are reused after hash/draw and representative AP checks; all new deterministic-track/V3 replicates are calculated. Bone-lesion/overall intervals use 492 draws; other classes 500. Undefined absent-class AP is not zero and overall never averages fewer than eight classes. [All class/metric contrasts](paired_AP_contrasts.csv) include effective support. Nominal percentile intervals do not account for validation reuse, seed variation or multiplicity.

## Operating detections and object recovery

Confidence .25 / IoU .50, unchanged descriptive one-to-one matcher; FP is annotation-relative unmatched prediction, not a clinical adjudication.

| Method | TP | FP | FN | Precision, % | Recall, % | Fracture TP | Fracture FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 6467 | 884 | 737 | 87.974 | 89.770 | 2463 | 346 |
| C | 6505 | 1055 | 699 | 86.045 | 90.297 | 2478 | 403 |
| V3_standard | 6478 | 917 | 726 | 87.600 | 89.922 | 2471 | 377 |
| D | 6518 | 1016 | 686 | 86.514 | 90.478 | 2453 | 365 |
| F | 6552 | 1152 | 652 | 85.047 | 90.949 | 2474 | 404 |
| V3_rarity | 6538 | 1037 | 666 | 86.310 | 90.755 | 2470 | 380 |

| Contrast | Recovered objects | Lost objects | Recovered fracture | Lost fracture |
| --- | --- | --- | --- | --- |
| V3_standard-A | 11 | 0 | 8 | 0 |
| V3_standard-C | 0 | 27 | 0 | 7 |
| V3_rarity-D | 20 | 0 | 17 | 0 |
| V3_rarity-F | 0 | 14 | 0 | 4 |

[operating_points.csv](operating_points.csv) retains confidence .05/.25/.50 at IoU .50/.75 for every class and method. Counts and source correspondences use their original definitions; recovered/lost targets must not be confused with introduced/removed unmatched predictions. Paired operating intervals are retained in [results.json](results.json). Residual GT and unmatched-prediction categories are in [error_categories.json](error_categories.json), with annotated-object/patient support:

| Method | Background/unannotated FP | Localization FP | Duplicate FP | Classification FP |
| --- | --- | --- | --- | --- |
| A | 388 | 322 | 163 | 11 |
| C | 460 | 362 | 221 | 12 |
| V3_standard | 412 | 329 | 164 | 12 |
| D | 569 | 314 | 124 | 9 |
| F | 625 | 344 | 174 | 9 |
| V3_rarity | 586 | 318 | 124 | 9 |

## Confidence decisions and ROI utilization

| Method | Promoted recovered TP | Reassigned TP | Promoted FP | Update below .25 | Inherited TP→FP | Inherited FP→TP |
| --- | --- | --- | --- | --- | --- | --- |
| V3_standard | 11 | 0 | 33 | 76 | 0 | 0 |
| V3_rarity | 20 | 0 | 21 | 96 | 0 | 0 |

| Method | Local calls | ROIs with update | ROIs with recovery | ROIs with promoted FP | ROIs with both | ROIs without update |
| --- | --- | --- | --- | --- | --- | --- |
| V3_standard | 981 | 118 | 11 | 33 | 0 | 863 |
| V3_rarity | 883 | 132 | 20 | 21 | 0 | 751 |

An updated row that becomes TP can reassign an already-detected object, displacing an inherited row. Those events are separate from new recovery. ROI update count is not a utility score: harmful promotions and mixed recovery/FP ROIs are explicitly shown. [All rejection/ownership counts](promotion_and_ROI.json) describe protected conflicts, reserved evidence, ambiguous graph degrees and sequential conflicts.

## Actual inference cost

| Method | Calls | Mean calls | Mean ms | Median ms | p95 ms | Paired global ms | Ratio | Peak alloc/res MiB | 0/1/2 ROI images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V3_standard | 4069 | 1.31768 | 72.235 | 61.023 | 130.227 | 58.140 | 1.2424 | 79.11 / 114.00 | 2267 / 661 / 160 |
| V3_rarity | 3971 | 1.28595 | 70.961 | 60.258 | 124.072 | 58.088 | 1.2216 | 79.11 / 114.00 | 2325 / 643 / 120 |

Each image uses one required global forward and at most two local forwards. A model-level hook independently reconciles per-view call counters, including three synthetic warmups and three first-pilot-image replays recorded separately. Full population counts exclude warmups; no hidden extra pass supplies the paired control. Pilot/full predictions agree exactly, all new globals reproduce saved A/D, model state is unchanged, and no technical rerun is used to improve timing.

E2E begins before decode and ends after fusion/checks/CUDA synchronization. The paired standard prefix ends after required global prediction/restoration, before reference-array equality checking and local planning. Export, offline replay/GT/AP, loading and warmups are excluded. Per-image logs and outer loop times remain private; [runtime.csv](runtime.csv) records mean/median/p95 and allocator memory. PyTorch peaks exclude driver/display allocations. Separate historical A/C/D/F session times are not controlled speed comparisons.

## Frozen scientific gates

| Checkpoint | Track | Frozen criterion | Result |
| --- | --- | --- | --- |
| standard | historical | overall_map50_95_positive | PASS |
| standard | historical | overall_map50_95_interval_positive | FAIL |
| standard | historical | overall_map50_nondecreasing | PASS |
| standard | historical | fracture_ap_points_preserved | PASS |
| standard | historical | fracture_interval_tolerance | PASS |
| standard | historical | all_class_fp_ratio | PASS |
| standard | historical | fracture_fp_ratio | PASS |
| standard | historical | mean_latency_ratio | PASS |
| standard | historical | maximum_physical_calls | PASS |
| standard | deterministic | overall_map50_95_positive | PASS |
| standard | deterministic | overall_map50_95_interval_positive | FAIL |
| standard | deterministic | overall_map50_nondecreasing | PASS |
| standard | deterministic | fracture_ap_points_preserved | PASS |
| standard | deterministic | fracture_interval_tolerance | PASS |
| standard | deterministic | all_class_fp_ratio | PASS |
| standard | deterministic | fracture_fp_ratio | PASS |
| standard | deterministic | mean_latency_ratio | PASS |
| standard | deterministic | maximum_physical_calls | PASS |
| rarity | historical | overall_map50_95_positive | PASS |
| rarity | historical | overall_map50_95_interval_positive | FAIL |
| rarity | historical | overall_map50_nondecreasing | PASS |
| rarity | historical | fracture_ap_points_preserved | PASS |
| rarity | historical | fracture_interval_tolerance | PASS |
| rarity | historical | all_class_fp_ratio | PASS |
| rarity | historical | fracture_fp_ratio | PASS |
| rarity | historical | mean_latency_ratio | PASS |
| rarity | historical | maximum_physical_calls | PASS |
| rarity | deterministic | overall_map50_95_positive | PASS |
| rarity | deterministic | overall_map50_95_interval_positive | FAIL |
| rarity | deterministic | overall_map50_nondecreasing | PASS |
| rarity | deterministic | fracture_ap_points_preserved | PASS |
| rarity | deterministic | fracture_interval_tolerance | PASS |
| rarity | deterministic | all_class_fp_ratio | PASS |
| rarity | deterministic | fracture_fp_ratio | PASS |
| rarity | deterministic | mean_latency_ratio | PASS |
| rarity | deterministic | maximum_physical_calls | PASS |

SUCCESS requires every original primary criterion and stable-track corroboration on both checkpoints. Original secondary trade-off flags are retained in JSON for historical context but do not replace the strict primary criteria. The 10% FP allowance and −.005 fracture interval tolerance are engineering limits, not clinical safety or noninferiority margins.

## Predeclared component attribution

| Checkpoint | Predefined cell | Stable mAP50–95 | Stable fracture AP50–95 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| standard | legacy_V1_on_historical_views | 0.33365761 | 0.51978302 | 6505 | 1055 | 699 |
| standard | legacy_V1_on_V3_views | 0.33372872 | 0.51977985 | 6505 | 1029 | 699 |
| standard | V3_arbitration_with_legacy_cap_on_V3_views | 0.33383619 | 0.51973806 | 6480 | 925 | 724 |
| standard | actual_V3 | 0.33382828 | 0.51974356 | 6478 | 917 | 726 |
| rarity | legacy_V1_on_historical_views | 0.33756167 | 0.51220391 | 6552 | 1152 | 652 |
| rarity | legacy_V1_on_V3_views | 0.33758743 | 0.51224807 | 6552 | 1128 | 652 |
| rarity | V3_arbitration_with_legacy_cap_on_V3_views | 0.33759893 | 0.51221173 | 6540 | 1048 | 664 |
| rarity | actual_V3 | 0.33761271 | 0.51221780 | 6538 | 1037 | 666 |

All cells use original global coordinates and deterministic AP. Historical-view versus V3-view legacy V1 cells isolate available views with the same frozen score/ownership rule. The next step changes arbitration with the legacy cap held fixed; the last step adds midpoint moderation. These are descriptive CPU replays, not candidate V3 configurations or post-hoc selection. They require no additional GPU inference, do not provide independent confirmation, and are not substituted for the official V2 geometry comparator.

**Overall classification: PARTIAL IMPROVEMENT.** See the [scientific assessment](V3_SCIENTIFIC_ASSESSMENT.md) for interpretation and remaining uncertainty.
