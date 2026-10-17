# Official rarity-only training integrity verification

Phase 4C, 2026-10-17. **PASS — GO for Phase 4D review.** The official rarity-only run completed all 80 epochs without interruption. Saved sampling, optimizer, AMP, metrics and checkpoint evidence reconcile. No training, new validation inference, D/E/F evaluation, CG-ALR change or held-out test evaluation was performed in this audit. Phase 4D still requires the student's separate authorization.

## Identity and scope

The initial working tree was clean on `main` at `bb5a60de9c31e241d29a403348d1654d216162b2`, which is also the actual training launch revision. The isolated supplementary verifier and focused tests are committed at `a7d7c97`. The common trainer's inherited `implementation_revision` remains its earlier frozen revision; its exact source hashes, the additive rarity source hashes and the installed framework hashes all match their locks.

| Identity | SHA-256 |
| --- | --- |
| Common training protocol | 281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01 |
| Rarity condition | 802c3f295182b346eb2a809e262cf7309b4d6dcea40de188c2bc2b8ddc58f2d5 |
| Patient split manifest | ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49 |
| Prepared source records | 0cd3a72c83ad46ca32ebae1512131c0a4da034e455082d81b8c34a12f54c327c |
| Original COCO YOLOv8s initialization | 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a |

Actual run: `outputs/training/rarity_only/`, active directory `.`, seed 42, official/non-diagnostic, original eight-class mapping, original prepared full-image labels. `launch.json`, `run_identity.json`, `condition_protocol.json`, `args.yaml`, checkpoint arguments, preparation journal and frozen dataset provenance agree. COCO initialization bytes retain their approved hash. No failure record, continuation directory, partial checkpoint or active run lock remains; start/end events and `completed.json` agree.

The frozen verifier was inspected first. Its existing `post_training_verification.json` already reported a pass. To honor the explicit prohibition on overwriting reports, Phase 4C reran **the unchanged `verify_run` used by `python -m wrist_rarity.verify run`**, intercepting only its final report write into a new private audit directory. The literal CLI was not rerun because it replaces the existing report. Every verification condition ran unchanged, and the new result equals the previous report. All **277 original training files** retain their initial hashes.

The supplement closes original-verifier gaps: exact official CSV length/order, complete class records and macro reconciliation, rounded-fitness/latest-tie selection, full history prefixes, sidecar semantic fields, all saved parameter/update counters, scaler-event replay and argument/lineage consistency. Private evidence is in `outputs/rarity_training_completion/audit_01/`; [the aggregate summary](verification_summary.json) binds its receipts and the tested source identities without exposing patient IDs or source indices.

## Completion and actual consumption

Measured launcher wall time is **33,399.034 s = 9 h 16 min 39 s**. Per-epoch training totals 30,982.539 s and existing trainer-integrated validation totals 2,210.989 s. CSV elapsed time is 33,375.9 s; its measurement boundaries differ from the launcher wall timer. There are exactly 80 ordered, finite CSV rows, 80 progress records and 80 per-class records containing all eight classes (**640 class entries**).

Both original and supplementary checks reconstruct actual consumed source indices. The original also verifies each label signature, position, epoch, worker ID/seed, batch size and augmented-repeat receipt against the regenerated approved plan. The supplement independently calculates population constraints from actual consumed indices and training metadata.

| Constraint | Verified in every one of 80 epochs |
| --- | --- |
| Draws / physical batches | 14,061 / 2,344: 2,343 batches of six, final batch of three |
| Reallocated repeated draws / unique images | 625 / 13,436 |
| Patient coverage | All 4,264 training patients |
| Fracture-positive / no-fracture-box draws | 9,307 / 4,754 |
| Unique images in those strata | 8,842 / 4,594; both meet the 95% minimum |
| Maximum image repetition | Two draws |
| Patient extra cap | `min(3, ceil(0.5 * original patient views))`; no violation |
| Lesion-patient protection | All 66 views of all 16 patients exactly once |
| Actual repeated-source augmentation | All 625 pairs have different transformed-tensor hashes |
| Workers and boundaries | Both workers 0/1, approved epoch seeds, no stale indices or labels |

Total verified consumption is **1,124,880 draws and 187,520 batches**. No-fracture-box denotes an annotation stratum, not verified clinical normality. Source labels are checked before the unchanged native augmentation; normal geometric target filtering remains part of that common policy.

## Optimizer, AMP and saved states

Actual settings match the frozen protocol: 768 pixels, physical batch six, AdamW, initial LR 0.001, final LR factor 0.01, betas 0.9/0.999, weight decay 0.0005 on ordinary weights and zero on normalization/bias groups, cosine schedule and one warmup epoch. Nominal batch 24 gives accumulation 1–4 during first-epoch warmup and four thereafter. Main augmentation and native validation settings remain unchanged; validation uses 3,088 images, batch 12, four workers, rectangular padding and no augmentation.

| Quantity | Observed |
| --- | --- |
| Optimizer attempts | 47,433: 1,139 in epoch 1, then 586 per epoch |
| Successful optimizer / EMA updates | 47,415 / 47,415 |
| Skipped updates / immediate recoveries | 18 / 18 |
| AMP initial / maximum / final scale | 128 / 8,192 / 2,048 |
| Scale growth events / final growth tracker | 22 / 1,122 |
| Maximum consecutive skips / skips in any 100-attempt window | 1 / 2; below the stop limits of 5 / 10 |
| Peak training allocated / reserved VRAM | 1.9051 / 2.3945 GiB |
| Device telemetry maximum memory / temperature | 3,044 MiB / 78 C |

All 58 sparse AMP events reproduce exactly from the 47,433-attempt history, including growth, backoff and recovery. Epoch deltas, cumulative progress, completion accounting and saved states agree. All **183 trainable parameter states** have AdamW step counters equal to the corresponding successful-update/EMA count. The fixed `model.22.dfl.conv.weight` is correctly present in a parameter group without Adam moments. An initially overstrict supplementary check assumed every grouped parameter acquired state; inspecting this fixed projection justified correcting the audit and adding a focused regression test. No training artifact was repaired.

All 12 checkpoint/sidecar pairs pass hash/length, epoch/protocol binding, CPU reload, FP32 finite live/EMA/best-EMA and optimizer state, scaler, scheduler, history-prefix and class-record checks. These include best/last, their predecessors and every ten-epoch snapshot. The final embedded best EMA equals the selected best checkpoint EMA tensor for tensor. Synthetic CUDA inference was unnecessary and was not run.

## Checkpoint selection and native validation metrics

| Checkpoint | Completed epoch | Native mAP50 | Native mAP50-95 | SHA-256 |
| --- | --- | --- | --- | --- |
| `best.pt` | **26** | **0.61319** | **0.33669** | cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e |
| `last.pt` | 80 | 0.54012 | 0.30274 | e3bc29b265e9eab5c767016e102a14e62031345611b4d22c306635d064d5c115 |

The pinned native validator rounds its macro metrics **and fitness to five decimal places before selection** (`ultralytics/engine/validator.py:298`). Fitness weights are `[0, 0, 0, 1]`; the project selects the latest exact tie. Epoch 26 is the unique maximum of the actual rounded eight-class macro mAP50-95 history. Its unrounded per-class mean is 0.33669001101409646, agreeing with saved fitness/CSV 0.33669 within the documented half-unit rounding bound of 0.000005. All epochs' macro metrics reconcile similarly; checkpoint history prefixes agree exactly with the parsed CSV. The student-reported expected values are therefore confirmed from local evidence.

**These are trainer-native, validation-selected scores. D/E/F common original-coordinate evaluator results remain pending.** They do not establish held-out performance or clinical validity.

## Preservation, limitations and next gate

All **36 standard-baseline artifacts** retain their historical byte hashes. All **121,973 raw/prepared stat entries** match the earlier inventory. Frozen source/configuration/dependency files and earlier public experiment evidence, including negative inference findings and readiness records, are unchanged from the launch revision. Shared frozen provenance metadata was read; no held-out test image, prepared label or prediction file was opened. Private medical/index/checkpoint evidence remains ignored.

The run needed no continuation, retry or numerical repair. Eighteen recovered gradient overflows are disclosed, rather than described as zero-overflow training. Relative to the preserved standard baseline's 47,416 successful updates, this run has one fewer successful update under the same 47,433-attempt budget. Its final checkpoint records three pending microbatches, which do not count as completed optimizer updates; no extra step or epoch was added. Finite-worker randomness follows the documented seed schedule, but augmented views are not paired with the baseline. The single seed and validation-selected checkpoint remain methodological limits.

The final full suite passes **197 tests in 31.239 s**, including ten focused completion tests. An initial direct focused invocation encountered Windows sandbox temporary-directory permissions; the established workspace-temp harness resolved that tooling issue. Tests use generated fixtures and do not fit models or access held-out data.

**Decision: GO for the student's Phase 4D review, with no unresolved training-integrity blocker.** Preserve both official checkpoints and all prior evidence. Stop here: implementation or execution of the D/E/F evaluator, further CG-ALR optimization and held-out evaluation require the next explicit authorization.
