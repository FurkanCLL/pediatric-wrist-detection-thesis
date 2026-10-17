# Experiment D execution contract

Phase 4D, 2026-10-17. This record precedes new model inference. The direct student request authorizes **A versus D on frozen validation only**, followed by analysis, verification, documentation and commit/push. It excludes training, tiled inference, adaptive inference, optimization and held-out evaluation. The requested deliverable is a repository Markdown report with aggregate JSON/CSV evidence.

## Fixed inputs and numerical path

- A: preserved standard epoch-22 `best.pt`, SHA-256 `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`.
- D: Phase 4C verified rarity-only epoch-26 `best.pt`, SHA-256 `cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e`.
- Dataset: unchanged validation membership, 3,088 radiographs / 914 patients / eight classes. Use source original-coordinate target geometry; no annotations participate in inference decisions.
- A predictions: reuse the recorded standard serial artifact, SHA-256 `cb056711668c04ebe8a7736724e35d46c6092f5a6e8ac5a9becba9a53bd1bee6`. Verify membership, class/box/score validity, source/config/checkpoint provenance and reproduce common AP from saved outputs. Do not repeat A GPU inference.
- Numerical reuse: unchanged `wrist_eval.sliced.make_adapter` and `predict(..., 'standard_serial', ...)`, existing resize/padding/restoration, native multi-label class-aware NMS, confidence .001, IoU .7, max300, FP32 unfused EMA, CUDA FP16 inputs/autocast, imgsz768, one full-image pass. The existing standard wrapper uses the library adapter; its sliced branch is never invoked. Every image must report **one model call and zero slices**.
- Reuse A's dimension-derived rectangular padding shapes, independently reconstructing them from metadata/native batch-size12 geometry. Redirect only adapter settings storage into the new attempt. Frozen numerical functions remain unchanged.
- AP: unchanged `native_stats` and `ap_summary` on original-coordinate annotations. Native training scores remain historical and are not the A/D endpoint.

## Pilot, execution and rejection rules

The unchanged outcome-independent `pilot_rows` selects 24 validation examples using geometry extremes, hash-ranked class examples, smallest class boxes, annotation count and patient coverage. It does not inspect model outputs. Pilot AP is not calculated. Verify checkpoint provenance, finite valid original-coordinate predictions, class IDs, counts and timing; replay its first three images at absolute tolerance1e-6. Retain the common feasibility limits: predicted full duration<10,800s and allocated peak<4GiB. No parameter adjustment follows the pilot.

Only a passed pilot permits the complete D pass. All 3,088 images must appear exactly once in frozen lexical order; predictions remain privately linked to patients. Full outputs for the 24 pilot images must match within1e-6. Input and checkpoint files are read-only; model state hashes before/after inference must agree. Existing attempt directories/receipts refuse replacement; failed partial artifacts remain available. A runtime file-open guard rejects raw-data reads, held-out image/label reads and secret files.

Private root: `outputs/rarity_comparison/study_01/`. Separate init, pilot, full, analysis, bootstrap and verification outputs prevent a failed stage from replacing successful evidence. Hash protected baseline/rarity training, previous inference, readiness, mining and Phase4C outputs before and after; compare raw/prepared stat inventories and all previous tracked sources/configs/evidence against revision `c5e64e9`.

## Predeclared analysis

Report A, D and D-minus-A for overall mAP50-95 (primary), mAP50, mAP75 and all eight classes' AP50/AP75/AP50-95. Use existing descriptive matching/error categories at confidences .05/.25/.50 and IoU .50/.75; emphasize .25/.50, fracture preservation, recovered/lost annotations, unmatched-prediction changes and paired localization diagnostics. Error categories are annotation-relative, without clinical adjudication.

Reuse `patient_groups`, `resample_indices` and percentile `interval`: **500 replicates, seed314159**, identical patient draws for A and D, all views/examinations together with multiplicity. Retain absent-class AP as undefined; report conditional valid replicate counts and omit overall eight-class mAP in resamples missing any class. Include paired AP intervals and TP/FP/FN/precision/recall intervals at confidence.25/IoU.50. These reflect validation-patient sampling with fixed selected checkpoints, not training-seed or external-population variability.

Final verification recomputes both point AP summaries and operating counts; reconciles target recovery; checks all500 resampling memberships, paired deltas, class-presence and intervals; recomputes AP for replicates0,1,249,499. Timing uses the same three synthetic warmups and per-image boundaries as A. Separate sessions and current hardware load prohibit interpreting raw time differences as controlled model-speed gains.

The small additive package is `src/wrist_compare/`; focused tests use generated data and no fitting. Stop after documented Experiment D. E/F, CG-ALR changes and the test set require later authorization.
