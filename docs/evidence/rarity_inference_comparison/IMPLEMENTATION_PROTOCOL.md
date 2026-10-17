# Phase 4E pre-inference contract

Student authorization: Phase 4E request, SHA-256 `770ebf427b2b61cfcb5f29766f84f472346a950da4b078959f89acc1fbab1e85`.
Starting revision: `323f221ae6978b9bea0e7398c976642e62e48036`.
This contract precedes new GPU inference. It authorizes E/F validation and reporting only; the held-out test, training, optimization and final method selection remain outside scope.

## Frozen methods and inputs

- Use rarity-only epoch-26 `best.pt`, SHA-256 `cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e`; check its sidecar, common/condition provenance, architecture, unfused FP32 EMA, eight classes and finite tensors before use.
- Evaluate the same 3,088 validation images, 914 patients and 7,204 original-coordinate annotations. Reconstruct and check the preserved rectangular global shapes from dimensions. Never open held-out images/labels/predictions.
- E calls unchanged `wrist_eval.sliced.predict(..., 'sahi_context', ...)`: 768, fractions .667/.667, overlap .20/.20, four slices plus global, sequential batch one, FP16 input/autocast over unfused FP32 weights, confidence .001, per-view and class-aware union NMS .70, max300. No NMM or SAHI Detail.
- F injects one verified rarity model through the unchanged `ViewRunner` and `adaptive.predict_adaptive(..., ['V2'])`. The pure algorithm receives no condition or ground truth. Reuse every frozen ROI, evidence, score, geometry, collision and tie rule; maximum2 ROIs/3 physical calls. Additive manifests bind these unchanged settings to rarity. Historical manifests/configs/sources remain immutable.
- Every F global array must have identical float32 shape and bytes to saved D (SHA-256 `b653b3d5e1df5ca227b29588e6085bb115d8530c99e934495d6e1d630a2b661d`). Check immediately before local calls and again from saved output. Stop the affected experiment and retain discrepancy evidence on failure.

## Execution and gates

Complete tests and commit/push implementation before `init`, which includes a small environment CUDA probe. Each condition then receives the identical historical outcome-independent24-image pilot, no pilot AP. Check finite original-coordinate rows, ordering, counts, numerical state, actual calls, memory and projected runtime; repeat only the first3 pilot images to verify exact determinism. Full inference requires its passed current pilot. Full predictions must equal all24 pilot outputs. Peak allocation must remain below4GiB and projected full inference below3h.

Private outputs: `outputs/rarity_inference_comparison/study_01/`. Existing attempts are rejected without writes. Fresh failures retain partial evidence and a failure receipt. Neither checkpoint is trained or modified. A/B/C/D predictions and metrics are reused and independently reproduced, without repeating inference. Constructor settings writes are redirected into the new stage directory.

F additionally verifies all protected rows/classes, exact row membership, confidence caps, unchanged confidence-.50 predictions, unique evidence assignment, original-coordinate restoration, quarter geometry blends and collision bounds. Replay every saved bundle through the unchanged pure planning/fusion functions without GPU calls.

## Analysis fixed before outcomes

Use unchanged original-coordinate AP and deterministic operating matching. Report overall/per-class AP50/AP75/AP50-95; all confidence .05/.25/.50 by IoU .50/.75 points; emphasize .25/.50. Include annotated-target recovery/loss, spatial correspondence of unmatched predictions, fracture preservation and actual latency/calls/usage.

Compute E-D, F-D, E-B, F-C and interactions `(E-D)-(B-A)`, `(F-D)-(C-A)`. Use the same500 patient draws, seed314159, all examinations/views grouped, for all six conditions. Compute interaction values within each replicate. Missing-class AP stays undefined; overall eight-class mAP requires all eight classes. Report conditional valid counts. Reproduce all2,000 old A/B/C/D AP replicates; separately verify indices, all contrasts/intervals and representative AP recomputations. Reapply the original CG-ALR scientific gates to F versus D, including the contemporaneous required global-call timing reference; no final algorithm selection.

Time actual decode-through-final-output paths before export/offline verification. Record GPU time, mean/median/p95, allocated/reserved peaks, physical calls and full loop time including export/checks. Historical A/B/C/D timings come from separate sessions. F's required global call provides its contemporaneous reference without extra inference; disclose its small exact-comparison overhead. Do not claim controlled speedups across sessions.

## Preservation and publication

Hash every prior protected private artifact, including D, and verify the original121,973-entry raw/prepared stat inventory before/after. Check Git changes against the starting revision: only additive E/F package/config/tests/evidence plus README status are allowed. Publish aggregate JSON/CSV and the requested report after independent verification. Keep image identifiers, patients, coordinates, full predictions and model files ignored and private. Preserve negative findings and state single-seed, validation-development, low-support and clinical-generalization limitations.

```powershell
.\.venv\Scripts\python.exe -m wrist_inference_comparison.run init --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.run pilot E --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.run pilot F --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.run full E --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.run full F --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.analysis analyze --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.analysis bootstrap --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.analysis verify --study study_01
.\.venv\Scripts\python.exe -m wrist_inference_comparison.export --study study_01
```

Commands are an audit trail, not instructions to repeat completed evidence. Stop after Phase4E for student review.
