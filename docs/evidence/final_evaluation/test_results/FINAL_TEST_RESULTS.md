# Final held-out test results

Six fixed conditions completed on 3178 images, 913 patients and 7,361 annotations. `final_C` is the predefined primary proposed pipeline. Final C/F use V3; historical validation C/F used V2 and remain unchanged. [Execution freeze](execution_freeze.json), [data integrity](data_integrity.json) and [verification](verification.json) record provenance. AP is on a 0–1 scale; contrast intervals below are percentage points (pp).

## Overall and fracture performance

| Condition | mAP50–95 | mAP50 | mAP75 | TP | FP | FN | Precision % | Recall % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_A | 0.31708115 | 0.54602277 | 0.32987946 | 6532 | 983 | 829 | 86.919 | 88.738 |
| final_B | 0.28512014 | 0.48866650 | 0.28932063 | 6702 | 3508 | 659 | 65.642 | 91.047 |
| final_C | 0.31709538 | 0.54604046 | 0.32990034 | 6547 | 1016 | 814 | 86.566 | 88.942 |
| final_D | 0.33025835 | 0.57202511 | 0.31829333 | 6601 | 1109 | 760 | 85.616 | 89.675 |
| final_E | 0.28358515 | 0.49550993 | 0.27010718 | 6760 | 3878 | 601 | 63.546 | 91.835 |
| final_F | 0.33029334 | 0.57212159 | 0.31830693 | 6627 | 1138 | 734 | 85.344 | 90.029 |

| Condition | Fracture AP50 | AP75 | AP50–95 | TP | FP | FN | Precision % | Recall % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_A | 0.91074752 | 0.50366920 | 0.50524317 | 2463 | 412 | 397 | 85.670 | 86.119 |
| final_B | 0.85499948 | 0.47424031 | 0.47774610 | 2556 | 1785 | 304 | 58.880 | 89.371 |
| final_C | 0.91096869 | 0.50371477 | 0.50535441 | 2478 | 440 | 382 | 84.921 | 86.643 |
| final_D | 0.90826463 | 0.48508891 | 0.49970648 | 2446 | 422 | 414 | 85.286 | 85.524 |
| final_E | 0.85053821 | 0.46691906 | 0.47095515 | 2535 | 1834 | 325 | 58.022 | 88.636 |
| final_F | 0.90883884 | 0.48519613 | 0.49989725 | 2468 | 441 | 392 | 84.840 | 86.294 |

Operating counts use confidence .25 / IoU .50. FP means unmatched to annotations, not a clinically confirmed false alarm.

## Class AP50–95 and support

| Class | Objects | Patients | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 2860 | 538 | 0.505243 | 0.477746 | 0.505354 | 0.499706 | 0.470955 | 0.499897 |
| metal | 123 | 24 | 0.560143 | 0.394716 | 0.560143 | 0.571493 | 0.374127 | 0.571493 |
| text | 3723 | 912 | 0.691919 | 0.686462 | 0.691919 | 0.697556 | 0.679919 | 0.697556 |
| periostealreaction | 463 | 163 | 0.188088 | 0.145615 | 0.188091 | 0.217214 | 0.158331 | 0.217303 |
| pronatorsign | 84 | 83 | 0.274339 | 0.256194 | 0.274339 | 0.280095 | 0.231343 | 0.280095 |
| bonelesion | 5 | 4 | 0.157709 | 0.156423 | 0.157709 | 0.190000 | 0.189684 | 0.190000 |
| softtissue | 68 | 61 | 0.126251 | 0.121107 | 0.126251 | 0.116688 | 0.119926 | 0.116688 |
| boneanomaly | 35 | 14 | 0.032957 | 0.042698 | 0.032957 | 0.069315 | 0.044395 | 0.069315 |

[AP.csv](AP.csv) contains AP50/AP75/AP50–95 for every class and both partitions. [support.csv](support.csv) also gives image support. Rare-class estimates require caution, especially the small bone-lesion cohort.

## Paired comparisons

| Frozen contrast | Overall delta pp | 95% interval pp | Fracture delta pp | 95% interval pp |
| --- | --- | --- | --- | --- |
| D−A (training) | +1.3177 | [+0.2208, +3.0145] (n=492) | -0.5537 | [-1.0834, +0.0283] (n=500) |
| E−B (training) | -0.1535 | [-1.2205, +1.7359] (n=492) | -0.6791 | [-1.2977, -0.0448] (n=500) |
| F−C (training) | +1.3198 | [+0.2209, +3.0192] (n=492) | -0.5457 | [-1.0683, +0.0326] (n=500) |
| B−A (SAHI) | -3.1961 | [-4.3818, -2.0918] (n=492) | -2.7497 | [-3.3936, -2.0808] (n=500) |
| E−D (SAHI) | -4.6673 | [-5.6950, -3.3678] (n=492) | -2.8751 | [-3.6140, -2.2469] (n=500) |
| C−A (V3, primary) | +0.0014 | [-0.0046, +0.0076] (n=492) | +0.0111 | [+0.0001, +0.0254] (n=500) |
| F−D (V3) | +0.0035 | [-0.0043, +0.0124] (n=492) | +0.0191 | [+0.0078, +0.0300] (n=500) |
| C−B (V3−SAHI) | +3.1975 | [+2.0923, +4.3825] (n=492) | +2.7608 | [+2.0887, +3.4018] (n=500) |
| F−E (V3−SAHI) | +4.6708 | [+3.3716, +5.6986] (n=492) | +2.8942 | [+2.2659, +3.6318] (n=500) |
| (E−D)−(B−A) | -1.4712 | [-2.3467, -0.3667] (n=492) | -0.1254 | [-0.7506, +0.4991] (n=500) |
| (F−D)−(C−A) | +0.0021 | [-0.0091, +0.0126] (n=492) | +0.0080 | [-0.0101, +0.0244] (n=500) |

All 500 patient resamples use seed 314159 and common draws across methods. Intervals are nominal percentile intervals; absent-class AP and affected eight-class means are undefined, with effective counts shown. The primary contrast is `C−A (V3, primary)` (C−A). Secondary comparisons and interactions are unadjusted for multiplicity. [paired_AP.csv](paired_AP.csv) and [results.json](results.json) retain all class/metric intervals and operating-count uncertainty.

## Object recovery

| Comparison | Recovered objects | Lost objects | Recovered fracture | Lost fracture |
| --- | --- | --- | --- | --- |
| final_B-final_A | 187 | 17 | 103 | 10 |
| final_C-final_A | 15 | 0 | 15 | 0 |
| final_C-final_B | 21 | 176 | 14 | 92 |
| final_E-final_D | 176 | 17 | 98 | 9 |
| final_F-final_D | 26 | 0 | 22 | 0 |
| final_F-final_E | 24 | 157 | 14 | 81 |

Recovery and loss refer to the same annotated objects under canonical operating-point matching. They are not counts of updated scores.

## Inference cost

| Condition | Mean ms | Median ms | p95 ms | Calls | Mean calls | Peak alloc/res MiB | Session ratio vs standard | V3 paired ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_A | 54.21 | 56.12 | 71.03 | 3178 | 1.0000 | 79.70/114.00 | 1.000 | — |
| final_B | 219.68 | 223.66 | 260.42 | 15890 | 5.0000 | 79.70/114.00 | 4.052 | — |
| final_C | 67.70 | 60.04 | 127.46 | 4281 | 1.3471 | 79.70/114.00 | 1.249 | 1.274 |
| final_D | 55.25 | 54.14 | 72.67 | 3178 | 1.0000 | 79.70/114.00 | 1.000 | — |
| final_E | 211.08 | 210.76 | 252.53 | 15890 | 5.0000 | 79.70/114.00 | 3.821 | — |
| final_F | 67.99 | 57.99 | 122.78 | 4127 | 1.2986 | 79.70/114.00 | 1.231 | 1.243 |

Time includes decode through final output and CUDA completion. Loading, three synthetic warmups, export and offline metrics are excluded. Physical forwards are independently counted. Allocator peaks exclude driver/display memory. Separate A–F session ratios are descriptive; the V3 paired prefix uses its required global call without an extra pass. Fixed order and laptop GPU conditions limit timing comparisons. Complete private logs retain ROI/slice calls and model-state digests.

## Frozen engineering checks

| Condition | Criterion | Result |
| --- | --- | --- |
| final_C | overall_map50_95_positive | PASS |
| final_C | overall_lower95_positive | FAIL |
| final_C | map50_nondecreasing | PASS |
| final_C | fracture_AP_points_preserved | PASS |
| final_C | fracture_lower95_tolerance | PASS |
| final_C | FP_limit | PASS |
| final_C | fracture_FP_limit | PASS |
| final_C | paired_latency_limit | PASS |
| final_C | maximum_calls | PASS |
| final_F | overall_map50_95_positive | PASS |
| final_F | overall_lower95_positive | FAIL |
| final_F | map50_nondecreasing | PASS |
| final_F | fracture_AP_points_preserved | PASS |
| final_F | fracture_lower95_tolerance | PASS |
| final_F | FP_limit | PASS |
| final_F | fracture_FP_limit | PASS |
| final_F | paired_latency_limit | PASS |
| final_F | maximum_calls | PASS |

Every criterion is unchanged from Phase 5C. The FP allowances and fracture tolerance are engineering criteria, not clinical safety margins. No test pilot, repeat inference, parameter revision or new training was used. [verification.json](verification.json) covers independent counts/AP, bootstrap checks, all saved V3 replays and historical preservation.

[condition_summary.csv](condition_summary.csv) provides a flat table for later figures, including throughput derived as 1/mean complete-path seconds. [supplementary_checks.json](supplementary_checks.json) records the separate scalar audit of all 1,584 interval channels, non-target AP invariance and recovery accounting. No figures are produced in this phase.
