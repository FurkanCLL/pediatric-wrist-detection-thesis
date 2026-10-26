# Final evaluation protocol

26 October 2026 — Phase 5C. The student selected CG-ALR V3.0 as the final proposed inference method before test evaluation. This protocol fixes the six conditions and their comparisons. It does not claim a test result or authorize Phase 5D.

## Conditions and fixed components

| Internal ID | Display ID | Training | Inference |
| --- | --- | --- | --- |
| final_A | A | Standard YOLOv8s | Standard |
| final_B | B | Standard YOLOv8s | SAHI Context |
| final_C | C | Standard YOLOv8s | CG-ALR V3 |
| final_D | D | Rarity-Only YOLOv8s | Standard |
| final_E | E | Rarity-Only YOLOv8s | SAHI Context |
| final_F | F | Rarity-Only YOLOv8s | CG-ALR V3 |

`final_C` is the primary proposed configuration; `final_F` is the training-strategy comparison. Historical validation A–F remain unchanged, including C/F using V2. Final exports must include the internal ID, split, training condition and inference version. Never overwrite the historical C/F rows or mix historical AP with the new metric policy.

Use the existing standard epoch-22 and rarity-only epoch-26 best EMA checkpoints. Their SHA-256 values are respectively `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b` and `cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e`. [The matrix](final_experiment_matrix.json) binds paths, training provenance, numerical configurations, code and dependencies.

Keep the verified 16-bit-to-8-bit preparation, three equal input channels, original coordinates and eight-class order: fracture, metal, text, periostealreaction, pronatorsign, bonelesion, softtissue, boneanomaly. Retain empty-label images. No enhancement or evaluation augmentation is added. Input size is 768 with the same batch-12 rectangular shape schedule reconstructed from dimensions in lexical image order. Inference remains sequential batch one, FP32 unfused EMA weights, FP16 normalized inputs and CUDA FP16 autocast, seed 42, confidence floor .001, class-aware multi-label NMS at IoU .70, and at most 300 final predictions per image.

SAHI Context retains .667 image-dimension slices, .20 overlap, the full-image view and class-aware NMS/IoU .70 merging. V3 retains the exact frozen proposal, reservation, matching, score and geometry policies on both checkpoints. No training, new proposal mechanism, parameter change or threshold search is allowed.

## Population and metrics

The fixed patient-separated test partition has **3,178 images, 913 patients and 7,361 target annotations** in eight classes. These counts come from the existing Phase 1B aggregate receipt, not new annotation inspection. No exclusions may be chosen from model outputs.

The primary metric is eight-class mAP50–95, averaged across all eight classes and ten IoUs .50:.05:.95. Secondary metrics are mAP50, mAP75, each class's AP50/AP75/AP50–95, fracture AP, and TP/FP/FN with precision/recall. The operating point is confidence **.25** and IoU **.50**, using the existing separate one-to-one descriptive matcher. False positives are unmatched predictions relative to the annotations, not clinically adjudicated errors. Report class image/object/patient support and recovered/lost objects for the planned inference comparisons. The prior optional reporting thresholds are not used to select a final operating point.

### Common deterministic ordering

All six conditions use the same rule, including SAHI rows without global detection IDs. Work from the saved, unchanged float32 prediction rows. Canonicalize each image by **descending confidence, class ID, x1, y1, x2, y2**. Exactly identical rows occupy consecutive duplicate occurrences; their identity is interchangeable because every value is equal. No coordinate rounding, confidence perturbation, deletion or fusion is performed by this step.

Supply canonical rows to the unchanged pinned native AP matcher. Use the existing deterministic AP integration with the lexicographic key **descending confidence, frozen lexical image index, canonical row rank, bootstrap image occurrence**. The final key separates repeated patient/image copies. Ground-truth order remains the frozen prepared annotation order. Operating-point matching receives the same canonical rows. Empty outputs remain valid; absent-class AP is undefined, while a present class with no predictions has zero AP.

This is a new common final-evaluation identity policy, not a correction to published validation results. Phase 5B's sensitivity track used original global IDs; those cannot identify every SAHI row. The new content-based rule is applied uniformly to all six final methods before any test result is available. Historical pinned and Phase 5B metrics stay separate. Do not interpret changed ordering as a detector improvement.

## Comparisons and uncertainty

The primary contrast is `final_C − final_A` for eight-class mAP50–95. Report these predefined secondary contrasts with their direction unchanged:

- Training effects: D−A, E−B, F−C.
- SAHI effects: B−A and E−D.
- V3 effects: C−A and F−D.
- V3 versus SAHI: C−B and F−E.
- Interactions: (E−D)−(B−A) and (F−D)−(C−A).

Here letters refer only to `final_` IDs. [The matrix](final_experiment_matrix.json) gives the exact 11 coefficient sets. Training effects compare the two fixed single-seed checkpoints; they do not establish a general causal effect of the sampler.

Use **500 paired patient-bootstrap samples, seed 314159**. Sort patient IDs, sample 913 patients with replacement, and include all their examinations/views with multiplicity. Use identical draws across all six methods and all contrasts. Recompute AP from concatenated per-image statistics and TP/FP/FN from summed image counts; derive precision/recall from those totals. Calculate contrasts within each replicate before taking the 2.5/97.5 percentiles. Store draw identities and effective replicate counts. Do not substitute zero for absent-class AP, omit absent classes from an eight-class mean, or assume the validation study's valid replicate count will also apply to test. Undefined precision/recall denominators remain undefined.

Report AP and operating-point contrast intervals. Secondary intervals are nominal and unadjusted for multiple comparisons. One training seed, sparse classes, validation-based method development and annotation limitations still restrict interpretation. No final method switch or new optimization may follow a disappointing test outcome.

The historical engineering checks remain visible for C−A and F−D: positive overall mAP50–95 and lower interval bound, nondecreasing mAP50, preserved fracture AP50/AP75/AP50–95 with lower-interval tolerance −.005, total/fracture FP ratios ≤1.10, V3 paired mean latency ≤2× and at most three calls. These are engineering criteria, not clinical safety margins. Report every check; a favorable secondary result cannot replace a failed primary check.

## Execution and cost, after separate authorization

Run exactly one full pass per condition in A/B/C/D/E/F order. Use three fixed synthetic warmups per condition and count them separately. No test pilot, test replay, confidence sweep or repeated timing attempt is planned. A technical failure stops the affected condition and preserves partial output; no repair may use test performance to change scientific rules. Cross-check standard global predictions against the corresponding required V3 global outputs without another forward. Keep raw predictions and complete V3 diagnostics private.

Check hashes, patient membership, class mapping, annotation counts, prepared image/label integrity and shapes after Phase 5D authorization, before inference. This Phase 5C preflight intentionally does not open test files or mixed annotation containers. The array-level predictor and evaluator are prepared; opening the test partition and run orchestration are reserved for Phase 5D.

Record checkpoint/configuration/code identities, software/hardware, condition order, model-state digests before/after, commands, run receipts and failures. Use new private directories under `outputs/final_evaluation/phase5d/`. The model factory rejects calls without explicit Phase 5D authorization; Phase 5C exposes no test loader or inference command.

For every image, synchronize CUDA and reset memory peaks before starting decode. Time from before decode through preprocessing, all forwards, NMS/restoration, planning, merging/fusion, validity checks and final CUDA completion. Pass that original timestamp to the array predictor. Exclude model loading, warmups, JSON export and offline metrics; record outer loop time separately. Report mean/median/p95 latency, throughput, actual model calls, slice/ROI counts, and peak CUDA allocated/reserved memory. Allocator peaks exclude driver/display memory. V3's paired global prefix ends after its required restored global output, before local planning, with no extra call. Global equality checks belong outside that paired prefix. Fixed session order and laptop GPU conditions limit cross-session timing comparisons.

A model-level forward hook must reconcile with per-view counters: standard uses one call, SAHI Context uses four slices plus one global call, and V3 uses one global plus at most two local calls. Verify finite outputs, geometry, caps, unchanged model state and complete population coverage before publishing aggregates.

## Phase 5C verification and stop

Run `.venv/Scripts/python.exe -m wrist_final.verify baseline`, then `preflight`. The latter runs synthetic tests under the dataset-access guard, reproduces all saved validation shape references, checks row-order invariance on the fixed 24 saved validation predictions for each final mapping, verifies historical receipts/checkpoint bytes and writes [preflight_verification.json](preflight_verification.json). No validation images, labels, new inference or performance calculation are needed for these adapter checks.

The receipt binds this document, the matrix, sources and tests. Keep `AGENTS.md` local, ignored and absent from both staging and the commit. **Stop after Phase 5C. Test access, inference and metrics require separate student authorization.**
