# CG-ALR V2 — saved-execution diagnostics

24 October 2026. Analyze the unchanged standard C and rarity-only F execution bundles. Public evidence is [v2_mechanism.json](v2_mechanism.json), [error categories](error_categories.json), [strata](error_strata.csv) and [recovery](recovery.csv); exact sources/hashes are in the [manifest](source_manifest.json). Detailed linked events stay private. No local inference or V3 replay is performed.

## 1. Detection errors before refinement

Use the frozen residual priority: operating TP, wrong-class IoU≥.50, same-class low-confidence IoU≥.50, same-class operating-confidence IoU≥.10, then no residual candidate. Residual assignments consume rows/objects once. “No residual candidate” is not complete absence of all spatial predictions.

| Model / class | Objects / patients | TP | FN | Low confidence | Localization | Classification | No residual candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A / fracture | 2808 / 538 | 2463 | 345 | 269 | 19 | 1 | 56 |
| A / periostealreaction | 493 / 163 | 298 | 195 | 172 | 8 | 0 | 15 |
| D / fracture | 2808 / 538 | 2453 | 355 | 264 | 27 | 1 | 63 |
| D / periostealreaction | 493 / 163 | 333 | 160 | 134 | 5 | 0 | 21 |

Low-confidence failures dominate both target classes, but the two models make different errors. D has more fracture localization/no-residual misses and fewer periosteal low-confidence misses than A. Classification is rare in these two classes. Other classes' counts and patient supports are retained in the synthesis/error tables; V2 cannot repair their classification or coverage because only classes 0 and 3 are mutable.

Annotation-relative unmatched prediction categories:

| ID | Background/unannotated | Localization | Duplicate | Classification |
| --- | --- | --- | --- | --- |
| A | 388 | 322 | 163 | 11 |
| B | 926 | 1233 | 1004 | 46 |
| C | 460 | 362 | 221 | 12 |
| D | 569 | 314 | 124 | 9 |
| E | 1289 | 1306 | 998 | 55 |
| F | 625 | 344 | 174 | 9 |

Background/unannotated and localization categories dominate; SAHI adds many duplicates as well. A duplicate here is an extra operating row with same-class annotated IoU≥.50 after one-to-one matching, not necessarily a pair above the inference NMS threshold .70. None is a clinically adjudicated false alarm. Category priority is operational and is not a causal error taxonomy.

At ideal 768 scale, periosteal <16-pixel targets have 61 objects/45 patients: A detects 12, D 24. For 16–<32 pixels, support is 212/110 with 119/137 TPs. Fracture 32–<64-pixel support is 1,633/488 with 1,394 TPs for both; 64–<128 has 1,133/350 with 1,039/1,029. Smaller targets are harder in these data, but density, overlap, aspect and border strata are confounded descriptive associations. [All strata](error_strata.csv) report objects/patients/TP/FN; zero-support strata are absent, not observed zero-recall groups. No subgroup AP with ambiguous FP attribution is invented.

## 2. Frozen pipeline and candidate generation

The algorithm receives only global predictions and image dimensions. It retains global confidence≥.001; candidates are fracture/periosteal, .05≤score<.25, short side≥1 ideal 768 pixel. Priority is .6Q+.4Z with short-side references 64/32; each represented target class gets a first head, then pooled ordering. Crops cover at least .75 of each image dimension with expanded anchor containment, preserving aspect/context. At most two selected ROIs/three calls are permitted. Coalescing requires ROI IoU≥.70 and envelope containment.

C has 827 fracture/2,046 periosteal eligible anchors; F has 740/1,518. Selected/coalesced/budget-exhausted counts are 1,285/1,512/76 and 1,215/1,000/43. There are no observed crop-geometry rejections. Typical ideal zoom is only about 1.333×, so these are broad contextual views, not high-magnification crops. Candidate confidence/size distributions and independent patient supports are in the diagnostic JSON.

| Stage | C (standard) | F (rarity-only) |
| --- | --- | --- |
| Eligible anchors | 2873 | 2258 |
| Anchors below .15 fixed-cap recovery floor | 2177 | 1678 |
| Local calls | 1285 | 1215 |
| Retained local evidence | 2007 | 2004 |
| Accepted protected matches | 1448 | 1454 |
| Accepted mutable matches / score changes | 459 | 429 |
| Unassigned retained evidence | 100 | 121 |
| Rejected matching pairs | 1596 | 1145 |
| Mutable pairs blocked by protected owner | 724 | 548 |
| ROIs without retained evidence | 215 | 201 |
| ROIs without output update | 897 | 843 |
| ROI images without output update | 654 | 622 |
| ROIs contributing a recovery | 37 | 34 |

The +.10 cap means a row originally below .15 cannot reach the .25 operating point in one update: 2,177/2,873 C candidates and 1,678/2,258 F candidates fall below this algebraic floor. This does not make them useless for AP: low-score ranking and geometry can still matter. Removing them is not an approved optimization.

## 3. ROI efficiency and local matching

Zero/one/two ROI image counts are 2,074/743/271 (C) and 2,121/719/248 (F). ROI images total 1,014/967; mean calls/image are 1.4161/1.3935. No-output-update calls are 897/1,285 (69.81%) and 843/1,215 (69.38%). Only 37/34 ROIs contribute to an operating recovery (2.88%/2.80% of local calls). These are observed productivity labels after local inference, not signals available for free beforehand.

Local filtering rejects 27,923/18,235 low-score rows, 5,229/3,453 non-target rows and 85/77 artificial-boundary rows; .70 union NMS suppresses 658/640 of 2,665/2,644 eligible rows, leaving 2,007/2,004. C/F matching consumes 1,907/1,883 retained rows. Protected high-confidence rows account for 75.93%/77.22% of accepted matches and cannot be updated. Pair matching is same-class, global/local IoU≥.50, greedy one-to-one by IoU then scores/IDs; mutable envelopes must fit the local source ROI. Counts of rejected pairs are not unique discarded objects. Protected-owner blocking counts include multiple competing pairs and are not recoverable-target counts.

Of the 100/121 unassigned retained rows, 45/43 overlap some same-class annotation at IoU≥.50, but most overlap already-detected objects. The separate one-to-one local oracle below limits baseline missed-target capacity to 7/10. It is not justified to add all unassigned evidence as new detections.

Higher seed confidence is associated with any update: for fracture-only seeds, productive/nonproductive median confidence is .148/.102 (C), .133/.106 (F); for periosteal-only seeds .177/.126 and .162/.125. Context/size associations differ by seed-class composition. Productivity includes harmful score changes; these associations neither prove causality nor provide a validated ROI-selection rule.

## 4. Confidence fusion and promotion outcomes

Matched mutable rows receive min(global+.10, local, .35), with conservative float32 quantization. The same checkpoint supplies both views, so local confidence is correlated evidence, not independent confirmation. C changes 459 scores: 250 remain below .25, 38 promoted rows recover objects, and 171 promoted rows are unmatched. F changes 429: 259 remain below .25, 34 recover objects, one claims an already-detected target, and 135 are unmatched. That reassignment makes one unchanged inherited row unmatched: **F net TP+34/FP+136 is compatible with 35 promoted TPs/135 promoted FPs.** Protected output fields do not guarantee preservation of one-to-one match ownership.

| Condition | Feature | Recovered TP median [IQR] | Promoted FP median [IQR] | P(TP feature > FP) |
| --- | --- | --- | --- | --- |
| C | global_confidence | 0.221 [0.206, 0.234] | 0.193 [0.173, 0.223] | 0.703 |
| C | local_confidence | 0.364 [0.308, 0.464] | 0.363 [0.300, 0.458] | 0.520 |
| C | global_local_iou | 0.816 [0.778, 0.871] | 0.818 [0.748, 0.886] | 0.521 |
| C | confident_context | 2.000 [1.000, 3.000] | 3.000 [2.000, 4.000] | 0.387 |
| F | global_confidence | 0.202 [0.182, 0.230] | 0.192 [0.169, 0.216] | 0.601 |
| F | local_confidence | 0.441 [0.364, 0.621] | 0.398 [0.312, 0.506] | 0.609 |
| F | global_local_iou | 0.823 [0.752, 0.864] | 0.825 [0.761, 0.861] | 0.518 |
| F | confident_context | 2.000 [1.000, 2.750] | 3.000 [2.000, 5.000] | 0.280 |

TP columns use newly recovered objects only (C n=38, F n=34); FP columns use unmatched promoted rows (171/135), excluding F's reassigned TP and inherited displaced row. The final probability gives half credit for ties; .5 means no univariate stochastic separation. It is not a trained classifier, optimized cutpoint, cross-validated discrimination estimate or significance claim. Event/image/patient support is reported by class/outcome in JSON. Groups overlap by patient and rows are not independent observations.

Local score and global/local IoU distributions overlap substantially. Within fracture, IoU separation is .467/.454 (C/F); within periosteal .537/.430. Higher IoU alone is therefore not supported as a reliable beneficial-promotion discriminator. Global score separates moderately (.703/.601 overall) but IQRs overlap; box size even changes association direction (.369/.542 for short side). Confident surrounding detections are more common for unmatched promotions, especially F, but class/context confounding prevents a causal or ready-to-deploy rule. Selective fusion is an evidence-backed problem to investigate, not an established solution.

## 5. Geometry refinement

V2 proposes a quarter-weight blend only for mutable matches with local confidence≥.50. Invalid/collision proposals retain the original box and V1 score, without rematching. Proposed changes total 82/116; accepted/veto counts and annotation-relative effects are:

| Condition | Accepted / vetoed | Improved / worsened / equal | Changes above .25 | Geometry-only recoveries / losses |
| --- | --- | --- | --- | --- |
| C | 64 / 18 | 29 / 17 / 18 | 29 | 0 / 0 |
| F | 90 / 26 | 43 / 22 / 25 | 40 | 0 / 0 |

Accepted changes occur on 64/89 images (2.07%/2.88%). Best same-class annotated IoU is used here; an unannotated box gets zero and no one-to-one TP is implied. Some changes worsen IoU; an identical maximum can also mean no nearby same-class annotation. The coordinate footprint is small and leaves most outputs untouched. Confidence-only V1 reconstruction from the frozen score updates yields the same .25/.50 object recovery/loss as V2 for both checkpoints, isolating zero operating-point gain from geometry. No alternate AP/V3 metric is calculated. Small AP effects may arise at stricter IoU or from ranking/ties, rather than broad localization recovery.

## 6. Structural recovery limits and retrospective oracles

Baseline missed target support is C fracture 345 objects/206 patients and periosteal 195/105; F fracture 355/204 and periosteal 160/97. Partition these exact objects; entries below show objects / independent patients. Stage groups can share patients, so patient counts are not additive.

| Stage (objects / independent patients) | C fracture | C periosteal | F fracture | F periosteal |
| --- | --- | --- | --- | --- |
| No eligible same-class global IoU≥.50 anchor | 225 / 148 | 84 / 57 | 238 / 160 | 77 / 55 |
| Eligible anchor outside selected ROI | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Covered anchor without mutable match | 77 / 68 | 77 / 62 | 71 / 59 | 56 / 42 |
| Mutable match without operating recovery | 28 / 26 | 11 / 11 | 25 / 24 | 14 / 14 |
| Recovered | 15 / 14 | 23 / 21 | 21 / 18 | 13 / 12 |

“Eligible anchor” here explicitly requires same-class original-box IoU≥.50 to the missed annotation, in addition to actual V2 candidate eligibility. “Covered” requires that candidate's expanded envelope inside a selected ROI. There are zero misses in the outside-selected-ROI stage: within this diagnostic definition, budget/coalescing does not reduce available adequate-anchor coverage. This says nothing about new, unobserved crop shapes or lower-IoU anchors.

Of the no-eligible-anchor fracture misses, 151/150 still have some same-class retained global row at IoU≥.50; 74/88 do not. Periosteal figures are 61/51 with and 23/26 without. The first group can include rows below .05 or otherwise ineligible/consumed rows. All retained rows are post-NMS, confidence≥.001 and max-300; absence here does not prove absence of a latent neural feature. A weak global anchor can exist while local evidence is missing or assigned elsewhere. Among covered-without-mutable-match misses, retained local evidence overlaps the annotation for only 4/8 C fracture/periosteal and 0/8 F. Those signals can fail the global/local matching or ownership rule; they cannot simply be counted as accepted matches.

Oracle capacities use the actual baseline missed targets and maximum one-to-one graph matching at annotated IoU≥.50. Original-box graphs use only unused baseline weak predictions, preserving already-operating matches. The final graph uses unassigned retained local evidence separately. They measure additional baseline objects under stated constraints, not AP, achievable recall, a proposed detector, or additive gains across rows:

| Oracle capacity (objects) | C fracture | C periosteal | F fracture | F periosteal |
| --- | --- | --- | --- | --- |
| All retained weak rows ≥.001 | 269 | 172 | 264 | 134 |
| Eligible .05–<.25 anchors | 120 | 110 | 117 | 82 |
| Eligible anchors inside selected ROIs | 120 | 110 | 117 | 82 |
| Also reachable under fixed +.10 score cap | 43 | 56 | 44 | 39 |
| Unassigned saved local evidence (separate graph) | 6 | 1 | 8 | 2 |

The .05 band has 230/199 one-to-one original-box capacities, reduced to 99/83 after the existing score-cap reachability condition, before requiring actual useful local evidence. Observed recovery is 38/34. Geometry is excluded from these original-box upper bounds. A GT-centered crop would be a different oracle with unseen local outputs; no such inference or claimed performance is produced. Saved local evidence comes only from the already-selected ROIs and cannot estimate an implementable image-only fallback's coverage or FP rate.

## 7. Implications

The three central limits are unreliable promotion evidence, missing/ineligible global anchors, and low useful-update yield from broad local views. Geometry is a small, mixed-quality component, not the main demonstrated recovery lever. More aggressive fusion or unassigned-local insertion could increase unmatched predictions; more calls cannot by itself repair classification, poor training representation or correlated confident errors. These diagnostics justify examining FP control first and ROI utility second, with fallback coverage remaining a separate unresolved research question.
