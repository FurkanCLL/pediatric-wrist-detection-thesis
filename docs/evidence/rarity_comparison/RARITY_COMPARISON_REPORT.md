# Rarity-only training gives mixed gains and lower fracture AP

Phase 4D / Experiment D, 2026-10-17. **Execution and integrity verification: PASS.** The common-validation result is mixed: rarity-only raises overall mAP50–95 by **0.368 percentage points**, with a paired 95% interval spanning zero, while fracture AP50–95 falls by **0.770 points**, with its nominal paired interval below zero. Periosteal-reaction AP improves; greater operating recall comes with more unmatched predictions. This does **not** establish an overall benefit with fracture preservation.

## 1. Objective and fixed conditions

The question is whether patient-aware rarity-only sampling improves eight-class detection relative to standard YOLOv8s training under identical full-image inference and original-coordinate evaluation. A uses the preserved standard epoch-22 checkpoint; D uses the independently verified rarity-only epoch-26 checkpoint. Both were independently trained from the same COCO initialization, with the frozen common training protocol and patient-separated data. No model is initialized from the other trained checkpoint.

| Identity | Value |
| --- | --- |
| Implementation revision, committed/pushed before inference | `0d6ece885aab10779f487637e442b57c194f8cd5` |
| A checkpoint SHA-256 | `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b` |
| D checkpoint SHA-256 | `cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e` |
| Common training protocol SHA-256 | `281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01` |
| Frozen patient split SHA-256 | `ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49` |
| Common inference configuration SHA-256 | `86ed8fd6450b77734e6cdea4eff5dec5eed59d5f0ea9790f0e37c507f9336c13` |
| Reused A predictions SHA-256 | `cb056711668c04ebe8a7736724e35d46c6092f5a6e8ac5a9becba9a53bd1bee6` |
| New D predictions SHA-256 | `b653b3d5e1df5ca227b29588e6085bb115d8530c99e934495d6e1d630a2b661d` |

Validation contains **3,088 radiographs, 914 patients and 7,204 target boxes** in the original fixed eight-class order. Source-coordinate annotations, patient membership and class mapping are unchanged. The eight classes include radiographic findings and artifacts; they are not eight fracture subclasses.

## 2. Common implementation and technical verification

The additive `wrist_compare` package reuses the unchanged standard-serial numerical functions, AP helpers, deterministic error accounting and patient resampling. It checks checkpoint bytes/sidecars/provenance, all model states and mapping, identical architecture and original initialization. D retains the FP32 unfused EMA, including 57 batch-normalization modules, with normalized FP16 inputs and CUDA FP16 autocast. In-memory state hashes agree before/after each inference stage.

Image size 768, confidence 0.001, multi-label class-aware NMS at IoU 0.7, max 300, original resize/padding/restoration/clipping, and confidence ordering match A. Rectangular padding shapes are reused from A and independently reconstructed from validation dimensions and native batch-size 12 grouping. Inference itself is serial batch one. No ground-truth box selects a view or changes inference.

The existing standard path uses its library adapter for a single full image; the sliced branch is never invoked. Adapter settings storage alone is redirected into the new attempt. Each full D record has exactly **one model call and zero slices**. No new A inference is performed. Its saved predictions and all per-class AP values reproduce within 1e-12 before comparison.

The unchanged outcome-independent 24-image pilot passed finite output/class/bounds/geometry/runtime checks. Its first three images replayed within 1e-6; pilot AP was not calculated and no parameter was tuned. Its 248.05s full-run extrapolation and 0.0776 GiB allocated peak passed the existing 10,800 s / 4 GiB limits. The full pass then produced exactly one ordered record per expected image; all 24 overlapping pilot predictions agree within 1e-6. Runtime guards block raw-data, held-out image/label and secret-file reads.

The [pre-inference contract](IMPLEMENTATION_PROTOCOL.md), [preflight](preflight.json), [pilot](pilot.json), [full D receipt](full_D.json), [verification](verification.json) and [preservation](preservation.json) retain exact identities. All 208 tests passed before new inference, including 11 focused comparison tests. Final checks recompute both AP summaries and operating counts, reconcile recovery, verify all 500 bootstrap memberships/deltas/class-presence/intervals, and independently recompute AP replicates 0, 1, 249, 499. Additionally, **all 500 A bootstrap results exactly match the preserved earlier A bootstrap**.

## 3. Overall common-evaluator metrics

AP values below are fractions; differences and interval bounds are **percentage points (pp), D minus A**. Intervals are paired patient-percentile intervals. These are original-coordinate results, not trainer-native network-space scores.

| Metric | A | D | Difference (pp) | Paired95% interval (pp) | Valid replicates |
| --- | ---: | ---: | ---: | --- | ---: |
| mAP50–95 (primary) | 0.33386896 | 0.33755131 | +0.368 | [-3.157, +2.481] | 492 |
| mAP50 | 0.59900535 | 0.61184505 | +1.284 | [-3.234, +4.668] | 492 |
| mAP75 | 0.30227745 | 0.31556154 | +1.328 | [-10.774, +13.500] | 492 |

The primary point estimate increases, but its wide interval includes material decreases and increases. The paired bootstrap does not establish an overall AP gain. D’s historical native selection scores 0.61319/0.33669 are preserved separately and are not substituted into this comparison. The earlier [evaluation bridge](../error_analysis_sahi/evaluation_bridge.json) explains the network-space versus clipped-original-coordinate distinction.

## 4. Per-class results and fracture preservation

| Class | GT / patients | A AP50–95 | D AP50–95 | Difference (pp) | Paired95% interval (pp) | Valid |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fracture | 2808 / 538 | 0.51973720 | 0.51203447 | -0.770 | [-1.402, -0.248] | 500 |
| metal | 96 / 24 | 0.57459777 | 0.58650862 | +1.191 | [-2.261, +3.888] | 500 |
| text | 3611 / 913 | 0.69774791 | 0.70604306 | +0.830 | [+0.525, +1.143] | 500 |
| periostealreaction | 493 / 163 | 0.21165596 | 0.23352884 | +2.187 | [+0.798, +3.606] | 500 |
| pronatorsign | 83 / 83 | 0.23866168 | 0.24076650 | +0.210 | [-3.747, +4.009] | 500 |
| bonelesion | 5 / 4 | 0.24050000 | 0.20200000 | -3.850 | [-36.483, +9.950] | 492 |
| softtissue | 72 / 61 | 0.12450314 | 0.15986579 | +3.536 | [-1.366, +7.413] | 500 |
| boneanomaly | 36 / 14 | 0.06354804 | 0.05966324 | -0.388 | [-6.719, +7.859] | 500 |

| Class | A AP50 | D AP50 | Difference (pp) | A AP75 | D AP75 | Difference (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fracture | 0.92117499 | 0.91619815 | -0.498 | 0.53459554 | 0.52567633 | -0.892 |
| metal | 0.89040901 | 0.90563961 | +1.523 | 0.55916186 | 0.61037449 | +5.121 |
| text | 0.98874444 | 0.98961892 | +0.087 | 0.82386688 | 0.83050847 | +0.664 |
| periostealreaction | 0.56599566 | 0.58615978 | +2.016 | 0.09957755 | 0.13976241 | +4.018 |
| pronatorsign | 0.57107216 | 0.54843688 | -2.264 | 0.13084787 | 0.09705247 | -3.380 |
| bonelesion | 0.41500000 | 0.41318182 | -0.182 | 0.19500000 | 0.19500000 | +0.000 |
| softtissue | 0.29603621 | 0.33779040 | +4.175 | 0.05689592 | 0.10987390 | +5.298 |
| boneanomaly | 0.14361034 | 0.19773486 | +5.412 | 0.01827398 | 0.01624424 | -0.203 |

**Observed:** periosteal-reaction and text AP50–95 intervals lie above zero; fracture AP50–95 lies below zero. These are nominal, unadjusted intervals on the fixed validation set. Metal, pronator-sign and soft-tissue AP50–95 point estimates increase, while bone-lesion and bone-anomaly AP50–95 decrease, with intervals spanning zero. Bone-anomaly AP50 increases while AP75/AP50–95 decline: improved loose-overlap detection is not uniform localization improvement.

**Fracture preservation is not demonstrated.** AP50 is lower by 0.498 pp (paired 95% interval [−0.973, +0.189] pp); AP50–95 is lower by 0.770 pp (paired 95% interval [−1.402, −0.248] pp). AP75 also falls by 0.892 pp. No noninferiority margin was predefined, so an interval spanning zero on another fracture metric cannot establish noninferiority.

Periosteal-reaction improvement is a measured benefit for that annotated finding. Text is an image artifact; its improvement does not establish better fracture detection. Soft-tissue and bone-anomaly loose-overlap gains need their broad uncertainty and operating false-positive costs considered. Bone lesion has only five boxes from four patients, so its changes cannot support a generalization claim.

## 5. Operating detection behavior

At confidence 0.25 / IoU 0.50, the descriptive matcher, separate from AP, assigns predictions in descending confidence to the best available same-class annotation. TP+FN equals GT and TP+FP equals retained operating predictions.

| Class | A TP / FP / FN | D TP / FP / FN | A precision / recall | D precision / recall | Recovered / lost |
| --- | --- | --- | --- | --- | --- |
| fracture | 2463 / 346 / 345 | 2453 / 365 / 355 | 87.68% / 87.71% | 87.05% / 87.36% | 73 / 83 |
| metal | 86 / 25 / 10 | 89 / 25 / 7 | 77.48% / 89.58% | 78.07% / 92.71% | 3 / 0 |
| text | 3552 / 171 / 59 | 3551 / 150 / 60 | 95.41% / 98.37% | 95.95% / 98.34% | 11 / 12 |
| periostealreaction | 298 / 295 / 195 | 333 / 328 / 160 | 50.25% / 60.45% | 50.38% / 67.55% | 55 / 20 |
| pronatorsign | 46 / 35 / 37 | 55 / 75 / 28 | 56.79% / 55.42% | 42.31% / 66.27% | 14 / 5 |
| bonelesion | 1 / 0 / 4 | 2 / 1 / 3 | 100.00% / 20.00% | 66.67% / 40.00% | 1 / 0 |
| softtissue | 17 / 9 / 55 | 27 / 60 / 45 | 65.38% / 23.61% | 31.03% / 37.50% | 11 / 1 |
| boneanomaly | 4 / 3 / 32 | 8 / 12 / 28 | 57.14% / 11.11% | 40.00% / 22.22% | 5 / 1 |
| overall | 6467 / 884 / 737 | 6518 / 1016 / 686 | 87.97% / 89.77% | 86.51% / 90.48% | 173 / 122 |

D recovers 173 annotations missed by A and loses 122 detected by A, net +51 TP. Recall increases from 89.77% to 90.48% (+0.708pp); precision declines from 87.97% to 86.51% (−1.460pp). The paired 95% intervals are [+17,+88]TP, [+65.475,+197.525]FP, [+0.234,+1.232]pp recall and [−2.190,−0.688]pp precision. All use 500 valid replicates.

Fracture loses ten net TP and adds 19 FP:73 recovered versus83 lost. Its operating-recall delta is−0.356pp, with95%[−1.207,+0.552]pp; the AP50–95 regression remains the stronger preservation concern. Periosteal reaction gains35TP with33 extraFP. Pronator sign gains9TP with40 extraFP; soft tissue gains10TP with51 extraFP, reducing its precision from65.38% to31.03%. Bone anomaly gains4TP with9 extraFP. Metal gains3TP with unchanged FP; text loses one TP but removes 21 FP.

| Confidence / IoU | A TP / FP / FN | D TP / FP / FN |
| --- | --- | --- |
| 0.05 / 0.5 | 6777 / 4351 / 427 | 6779 / 3850 / 425 |
| 0.05 / 0.75 | 5200 / 5928 / 2004 | 5219 / 5410 / 1985 |
| 0.25 / 0.5 | 6467 / 884 / 737 | 6518 / 1016 / 686 |
| 0.25 / 0.75 | 5002 / 2349 / 2202 | 5074 / 2460 / 2130 |
| 0.5 / 0.5 | 5942 / 249 / 1262 | 6102 / 301 / 1102 |
| 0.5 / 0.75 | 4773 / 1418 / 2431 | 4892 / 1511 / 2312 |

The complete per-class precision/recall at all six predefined settings is in [operating_points.csv](operating_points.csv). No confidence or IoU was optimized using these outcomes.

## 6. Unmatched predictions and localization patterns

Unmatched-prediction counts rise 884→1,016 (+132). Same-class one-to-one spatial correspondence at IoU 0.50 partitions these into 519 shared, 365 A-only/removed and 497 D-only/introduced predictions. This is correspondence between annotation-unmatched boxes, not proof of clinical false alarms or pathology absence. Source annotations may be incomplete.

At confidence 0.25 / IoU 0.50, fracture residual localization-category misses rise 19→27, while low-confidence misses change 269→264 and no-residual-candidate misses 56→63. For periosteal reaction, low-confidence misses fall 172→134; for soft tissue, 47→35. These ordered residual categories are descriptive and do not identify a unique cause.

Fracture TP at IoU 0.75 is 1,736 for A and 1,737 for D: 189 recovered versus 188 lost. Among the 2,380 fractures detected by both at IoU 0.50, D-minus-A matched IoU has mean −0.000645 and median +0.000238 (5th/95th percentiles −0.08998/+0.08523). This survivor subset does not include misses and cannot establish overall localization benefit. The lower fracture AP alongside nearly unchanged strict-IoU operating TP also cautions against explaining AP changes solely as box-coordinate accuracy; confidence ordering and coverage matter.

## 7. Patient-level uncertainty

Use 500 paired bootstrap replicates, seed 314159, sampling 914 patients with replacement. Every selected patient contributes all examinations/views with multiplicity, with the identical draw used for A and D. Intervals are percentile 2.5%/97.5% bounds. Eight replicates omit all four bone-lesion patients; bone-lesion AP and the fixed eight-class overall mAP are undefined there and are omitted rather than set to zero. Their intervals use 492 replicates; all other class AP intervals use 500. Operating-count/recall/precision intervals retain their own defined-pair counts in [patient_bootstrap.json](patient_bootstrap.json).

These intervals describe patient resampling of this reused development/validation set with two fixed, validation-selected checkpoints. They do not measure training-seed variability, selection uncertainty across new training runs, external-dataset generalization or clinical reliability. Multiple per-class/threshold comparisons are not multiplicity-adjusted. The primary interval is particularly sensitive to the small bone-lesion cohort; its conditional definition is retained explicitly.

## 8. Inference efficiency

Current device: NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 12.6, PyTorch 2.14.1+cu126 and Ultralytics 8.4.177. Both use the same architecture, input precision/geometry and one-pass protocol, with three identical synthetic warmups per stage.

| Measurement | Preserved A session | Current D session |
| --- | ---: | ---: |
| Mean end-to-end / image (ms) | 62.334 | 59.739 |
| Mean GPU forward / image (ms) | 33.733 | 31.047 |
| Median end-to-end / image (ms) | 61.769 | 58.875 |
| 95th percentile end-to-end / image (ms) | 77.481 | 74.934 |
| Throughput from summed image end-to-end time (images/s) | 16.043 | 16.739 |
| Recorded full loop (s) | 196.602 | 185.785 |

Per-image end-to-end boundaries include decode plus the same preprocessing, synchronized model call, NMS/restoration and standard wrapper. They exclude loading, warmup, JSON serialization and later metrics. A’s recorded loop also includes per-image metric/export work; D’s loop includes export and validity checks, with AP evaluated afterward. These loop totals therefore have a boundary difference. Separate sessions also have uncontrolled hardware/background load. The lower measured D latency is **not a controlled model-speed improvement**. All actual timing distributions and memory counters are retained in the aggregate receipts.

## 9. Preservation and reproducibility

Before/after hashes agree for all 674 protected private artifacts, covering both official training runs, previous A/SAHI/CG-ALR evidence, readiness, mining and Phase 4C outputs. All 121,973 raw/prepared stat entries match the historical inventory. All previous tracked source/config/dependency/evidence files remain unchanged from `c5e64e9`; only the additive comparison and current handoff documentation change. No previous result or verification report was replaced. Private predictions, linkage, detailed errors and bootstrap records remain under ignored `outputs/rarity_comparison/study_01/`.

No inference failure, numerical repair, threshold change, fitting, new A inference, E/F execution, adaptive optimization or held-out content access occurred. To reproduce deliberately, use a new study name; successful outputs are never destinations for reruns:

```powershell
.\.venv\Scripts\python.exe -m wrist_compare.infer init --study study_repeat_01
.\.venv\Scripts\python.exe -m wrist_compare.infer pilot --study study_repeat_01
.\.venv\Scripts\python.exe -m wrist_compare.infer full --study study_repeat_01
.\.venv\Scripts\python.exe -m wrist_compare.analysis analyze --study study_repeat_01
.\.venv\Scripts\python.exe -m wrist_compare.analysis bootstrap --study study_repeat_01
.\.venv\Scripts\python.exe -m wrist_compare.analysis verify --study study_repeat_01
```

Exact numerical tables are [overall_metrics.csv](overall_metrics.csv), [per_class_metrics.csv](per_class_metrics.csv), [comparison_summary.json](comparison_summary.json) and [patient_bootstrap.json](patient_bootstrap.json). The verification receipt binds private artifact hashes and states the scope of recomputation; it does not claim to have rerun all 500 AP calculations twice.

## 10. Interpretation and stopping point

**Observed:** a small primary-metric point increase, clear periosteal-reaction and text AP changes within the stated nominal intervals, lower fracture AP50–95, and a recall-versus-unmatched-prediction tradeoff. **Supported interpretation:** rarity-only changes detection behavior and helps some findings, but it does not demonstrate a reliable overall improvement with fracture preservation in this comparison. It should be reported as a mixed result, not an unqualified successful intervention.

**Unverified hypotheses:** whether a different seed, scoring calibration, alternative sampler or external population would change this tradeoff. No such experiment or tuning was conducted. Finite-worker augmentation realizations are not paired with the baseline, and successful-update counts differ by one under the same attempted-update budget, as already disclosed in Phase 4C. These two selected runs cannot isolate training-seed or augmentation variability as independent effects.

**Experiment D is complete and verified. Stop for student review.** Experiments E/F, CG-ALR changes, further training and the held-out test set remain unauthorized and untouched. Formal supervisor alignment of the revised thesis scope is still pending.
