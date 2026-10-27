# Chapter 5 figures and tables

Final A–C use the standard checkpoint; D–F use the rarity-only checkpoint. Inference is Standard/SAHI Context/CG-ALR V3 within each group. C is the predefined primary pipeline. Historical V2 figures remain unchanged.

Quantitative exports are 300 DPI PNG plus vector SVG at 6.6 inches (167.6 mm) width. Text is designed for a normal A4 thesis page. AP figures use percentages, contrast figures use percentage points, losses retain native units, and latency uses milliseconds. Use the following text as technical caption guidance; the student writes the final thesis text.

Native training-time validation curves are separate from the common final original-coordinate evaluator. Training losses reflect different sampled mixtures and are descriptive. No smoothing or missing measurements are added.

## 5.1 — train losses

[PNG](5_1_training/train_losses.png) · [SVG](5_1_training/train_losses.svg)

Training losses for both 80-epoch runs. Raw recorded epochs, without smoothing or interpolation. Losses are native training-framework values.

## 5.1 — val losses

[PNG](5_1_training/val_losses.png) · [SVG](5_1_training/val_losses.svg)

Validation losses for both 80-epoch runs. Raw recorded epochs, without smoothing or interpolation. Losses are native training-framework values.

## 5.1 — native validation metrics

[PNG](5_1_training/native_validation_metrics.png) · [SVG](5_1_training/native_validation_metrics.svg)

Native training-time validation metrics. Markers identify selected epochs 22 and 26 on mAP50–95. Native precision/recall use framework reporting, not the final fixed .25 operating point; these are not the final common original-coordinate metrics.

## 5.2 — test map50 95

[PNG](5_2_overall/test_map50_95.png) · [SVG](5_2_overall/test_map50_95.svg)

Final test eight-class mAP50–95. A–C use standard training; D–F use rarity-only training. In each group the inference order is Standard, SAHI Context, V3. C is the predefined primary pipeline. Bars begin at zero; tiny V3 differences should not be read as established superiority.

## 5.2 — validation test map50 95

[PNG](5_2_overall/validation_test_map50_95.png) · [SVG](5_2_overall/validation_test_map50_95.svg)

Comparable validation (3,088 images/914 patients) and test (3,178 images/913 patients) results under the same final deterministic evaluator. C/F use V3 in both partitions. Separate cohorts; this is not a paired validation-minus-test inference.

## 5.3 — class ap50 95 heatmap

[PNG](5_3_classes/class_ap50_95_heatmap.png) · [SVG](5_3_classes/class_ap50_95_heatmap.svg)

Test AP50–95 for all eight classes; displayed cell labels are rounded to one decimal percent. A–C: standard training; D–F: rarity-only. The full 0–100% scale is shared. Table 3 gives support, including bone lesion (5 objects/4 patients) and bone anomaly (35/14).

## 5.3 — fracture ap

[PNG](5_3_classes/fracture_ap.png) · [SVG](5_3_classes/fracture_ap.svg)

Fracture AP under the six final conditions. Every panel begins at zero. V3 preserves all three fracture point metrics relative to its corresponding standard inference; rarity-only training does not improve every target. Paired uncertainty is reported separately.

## 5.3 — rarity class differences

[PNG](5_3_classes/rarity_class_differences.png) · [SVG](5_3_classes/rarity_class_differences.svg)

Rarity-only minus standard training with standard inference (D−A), by class. Points and nominal patient-paired percentile 95% intervals use 500 draws/seed314159; valid counts vary with class presence. All bounds are retained. The sparse-class intervals limit interpretation; this compares two fixed training runs, not a general sampler effect.

## 5.4 — false positives

[PNG](5_4_errors/false_positives.png) · [SVG](5_4_errors/false_positives.svg)

Annotation-relative unmatched predictions at confidence .25 and IoU .50 on the test set. These are not clinically confirmed false diagnoses. All six final conditions use the same operating-point matcher.

## 5.4 — recovery fp tradeoff

[PNG](5_4_errors/recovery_fp_tradeoff.png) · [SVG](5_4_errors/recovery_fp_tradeoff.svg)

Recovered/lost annotated objects and added unmatched predictions relative to the corresponding standard inference. The panels have separate count scales. Recovery is determined by annotation correspondence, not by the number of changed confidence scores.

## 5.5 — latency and calls

[PNG](5_5_efficiency/latency_and_calls.png) · [SVG](5_5_efficiency/latency_and_calls.svg)

Measured complete-path mean/p95 latency and actual mean physical calls per image. Three synthetic warmups, loading, export and offline metrics are excluded. Separate sessions and fixed A–F order limit speed comparisons. V3 paired global-prefix ratios are 1.274×/1.243×; p95 is a latency percentile, not a confidence interval.

## 5.6 — main paired differences

[PNG](5_6_uncertainty/main_paired_differences.png) · [SVG](5_6_uncertainty/main_paired_differences.svg)

Main predefined test contrasts: nominal patient-paired percentile 95% intervals from 500 draws/seed314159 (492 valid eight-class replicates). The common scale and zero line retain SAHI/training effects without magnifying tiny V3 changes. C−A is primary; its interval includes zero. Secondary intervals are unadjusted and do not establish clinical validation. Table 4 contains all eleven contrasts.

## Scientific tables

CSV files use UTF-8 with BOM for Word/Excel import. Markdown has simple headers and no merged cells. AP point estimates are percentages; differences and intervals use percentage points. Rounding is for display only.

- Section 5.1: [Markdown](tables/table_1_training_checkpoints.md) · [CSV](tables/table_1_training_checkpoints.csv) — Training and selected checkpoints. Native metrics are the logged values at epochs selected by native eight-class validation mAP50–95; they are distinct from the final evaluator. Steady accumulation excludes the native warmup ramp. Full checkpoint identities are in the source manifest.
- Section 5.2: [Markdown](tables/table_2_final_test_performance.md) · [CSV](tables/table_2_final_test_performance.csv) — Final test performance at confidence .25 / IoU .50 for operating counts. A/B/C: standard training; D/E/F: rarity-only. Inference order within each group: Standard, SAHI Context, CG-ALR V3. C is primary. AP uses the common final deterministic evaluator. Displayed values are rounded; exact values remain in the verified Phase 5D results.json. The source manifest records displayed cells and source hashes.
- Section 5.3: [Markdown](tables/table_3_class_AP_support.md) · [CSV](tables/table_3_class_AP_support.csv) — Test class-specific AP50–95 and support. AP columns are percentages rounded to two decimals; tiny V3 changes can disappear at this precision. All 7,361 annotations are included. Bone-lesion and bone-anomaly support is sparse. AP50/AP75 remain available in the verified Phase 5D AP.csv.
- Section 5.6: [Markdown](tables/table_4_paired_comparisons.md) · [CSV](tables/table_4_paired_comparisons.csv) — All eleven predefined paired test comparisons. Patient bootstrap: 500 draws, seed 314159. Eight-class AP is undefined in samples missing a class; 492 samples contribute here. Secondary intervals are unadjusted for multiplicity. These are not clinical-validation or general training-seed uncertainty intervals.

## Local qualitative examples

Section 5.4 has two local-only PNG/SVG comparisons: a genuine saved V3 fracture recovery and an annotation-relative SAHI fracture FP. They are under `outputs/chapter5_figures/build_01/qualitative/`, with a private index and patient-linked provenance outside Git. Each uses Ground truth/A/B/C, identical display crops, fracture-only overlays, confidence .25 and matching IoU .50. Annotated text is masked and image metadata is not copied. Original prediction coordinates are preserved. Selection is the first hash-ranked eligible case whose target avoids text redaction; examples are explanatory, not representative.

The cached official release records CC BY 4.0, but complete individual privacy clearance is not established. Therefore no radiographs or qualitative image identifiers are published in this package. The local index records attribution and transformations.

## Sources, interpretation and reproduction

[Source manifest](source_manifest.json) records every plotted value, table value, source hash and rendering identity. [Verification](verification.json) records data, file, layout and visual checks. Run `.venv/Scripts/python.exe -m wrist_final_figures.build` with existing local evidence. This reads saved data only. It does not load weights, train, infer, rerun bootstrap estimates, or alter evaluation rules.

A separate global mAP50/mAP75 figure was omitted because Table 2 reports these metrics. The three-panel cost figure includes p95 latency; a redundant latency-distribution figure was omitted. No learning-curve metric was unavailable: both verified logs contain all 80 epochs and the requested loss/P/R/AP fields.

The figures retain the mixed result: SAHI increases unmatched predictions and lowers AP; rarity-only raises overall test macro AP but not every class; V3 adds modest operating recoveries within budget, with overall AP intervals spanning zero. Bootstrap intervals are nominal and do not establish clinical validation. Sparse findings and one training seed per strategy limit interpretation.
