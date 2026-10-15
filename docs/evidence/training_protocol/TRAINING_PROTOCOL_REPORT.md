# Official training protocol and baseline readiness

Updated: 2026-10-16. Phase 2B now freezes the student-approved common protocol at **80 epochs** and prepares a manual standard-baseline launch. No official epoch, class-aware training, SAHI or adaptive inference was started. Formal supervisor acceptance of the revised thesis scope remains pending.

The [official configuration](../../../configs/training_official.json) and [protocol lock](protocol_lock.json) identify the intended run. The 60-to-80 change was approved before the first official training run. The earlier 30-epoch diagnostic candidate and all Phase 2A evidence remain unchanged. The passed 60-epoch configuration, lock, preflight, test and estimate receipts are preserved in [the history record](history_60_epochs.json); the original report is retained in Git at `a2ae3c8`. The [GPU feasibility report](../gpu_feasibility/GPU_FEASIBILITY_REPORT.md) contains the measured training evidence; it is not an official baseline result.

The student selected a longer predefined window to reduce the risk of stopping before a useful validation checkpoint is reached. This is a planning choice, not a convergence guarantee. As literature context, [Ju and Cai (2023), Scientific Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654405/) report optimizer-dependent best epochs, including epoch 70 for YOLOv8m at 1024 with Adam. Our YOLOv8s/768/AdamW protocol, patient splits and preparation differ; their best-epoch behavior does not predict ours. Best-checkpoint selection still uses only our validation mAP50-95.

## Frozen common protocol

| Setting | Value and reason |
|---|---|
| Architecture / initialization | YOLOv8s detection, official generic COCO `yolov8s.pt`; never the smoke checkpoint |
| Inputs | Frozen eight-class prepared dataset and patient-separated seed-42 allocation |
| Resolution | 768; the measured batch-6 configuration fits the RTX 4050 Laptop GPU |
| Physical / nominal batch | 6 / 24; steady accumulation 4 |
| Budget | 80 epochs, 14,061 training draws per epoch; student-approved 60-to-80 extension before official training |
| Optimizer | CUDA fused AdamW, initial LR 0.001, betas (0.9, 0.999), eps 1e-8 |
| Weight decay | 0.0005 for ordinary weights; native zero decay for normalization weights and biases |
| Schedule | Cosine, `lrf=0.01`; one warmup epoch, warmup momentum 0.8, bias LR 0 |
| Precision | AMP enabled, initial scale 128; growth x2 after 2,000 successful updates, backoff x0.5 |
| Gradient clipping | Global norm 10 after unscaling; reject non-finite norms |
| Reproducibility | Seed 42, deterministic mode, pinned environment and source hashes |
| Loading | Two training workers, no full-dataset cache; validation batch 12 and four workers |
| Validation / selection | Every epoch, all 3,088 validation images, no augmentation; best eight-class validation mAP50-95, latest exact tie |
| Stopping / recovery | Early stopping disabled; last every epoch, best on selection, periodic recovery after completed epochs 10/20/30/40/50/60/70/80 |

Training uses rotation ±10 degrees, translation ±5%, scale 0.9–1.1 and grayscale-compatible HSV value magnitude 0.1. Hue/saturation changes, flips, Mosaic, MixUp, CutMix, CopyPaste, shear, perspective and optional Albumentations transforms are disabled. Loss gains remain box 7.5, classification 0.5 and DFL 1.5. Class weighting/remapping, multi-scale input, model compilation and channels-last mode are disabled. Detection confidence 0.001, IoU 0.7 and maximum detections 300 are explicit. The configuration's `augment=false` disables inference-time augmentation; the training loader still applies the specified on-the-fly transforms.

The native warmup also ramps accumulation from 1 to 4. The effective batch is therefore approximately 6–24 during warmup, and the final microbatch has three images. There are 2,344 microbatches per epoch. The uninterrupted native cadence predicts 1,139 optimizer attempts in the first epoch and 586 in each of the remaining 79 epochs: **47,433 nominal attempts** across 80 epochs. Actual successful updates and skipped updates are recorded separately. Resume can change cadence at an epoch boundary; this must be disclosed in later comparisons.

The future class-aware condition must retain the same split, architecture, initialization, common augmentation, optimizer, schedule, validation, selection rule, 80 epochs and 14,061 draws per epoch. Only its intended sampling/cropping strategy may change. That strategy is not implemented here. The two checkpoints will later support the unchanged 2 x 3 inference matrix.

## Identity and environment

- Configuration SHA-256, canonical sorted JSON: `281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01`.
- COCO checkpoint SHA-256: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`.
- Split manifest SHA-256: `ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49`.
- Implementation revision: `382d9c8315d5585607c2943bb0232ee3a469fda1`. The lock also records normalized project-source hashes, eight installed framework-source hashes and the existing preparation-journal hash. The lock explicitly binds `training_epochs=80`; checkpoint arguments must match that budget. Documentation commits may follow this implementation revision.
- Python 3.12.5; torch 2.14.1+cu126; torchvision 0.29.1+cu126; Ultralytics 8.4.177; NumPy 2.3.5; Pillow 12.3.0; OpenCV 5.0.0.93; PyYAML 6.0.3; CUDA build 12.6.
- Hardware established in Phase 2A: RTX 4050 Laptop GPU, about 6 GiB VRAM, driver 560.94, i5-12450H and 16 GB nominal RAM.

Preflight rejects a changed configuration, covered implementation/framework source, dependency version, class mapping, frozen record/manifest/policy hash, preparation journal, prepared train/validation bytes or COCO checkpoint. Configuration and code identity use normalized text/canonical JSON rather than Git checkout line endings. The original 80-class COCO model transfers 349/355 entries into the eight-class YOLOv8s detector, as in Phase 2A.

## AMP and numerical failure policy

The default initial scale 65,536 previously overflowed gradients despite finite losses. Initial scale 128 passed the existing benchmark and 586-update diagnostic epoch. That successful GPU work was reused without another smoke epoch.

The official trainer allows normal GradScaler backoff and growth. Each attempt checks unscaled gradients, clips finite gradients, then checks the AdamW parameter-state step counter to determine whether an update actually happened. This matters because fused AdamW can enter `optimizer.step()` while its kernel skips the update on overflow. An optimizer hook alone would count that event incorrectly. EMA advances only after an actual update; attempts, successful updates, skips and rolling overflow history survive recovery.

Every skip, scaler change and first successful recovery after an overflow is written to `events.jsonl` and flushed to the console. Stop thresholds are **five consecutive skipped attempts, or ten skipped attempts within the most recent 100 attempts**. These are explicit engineering stop rules, not evidence of a scientifically optimal threshold. Isolated recoverable overflows continue within those limits. A non-finite loss stops before backward; non-finite gradient norms, live model/EMA parameters or buffers, validation losses/predictions/metrics also stop the run. There is no silent NaN repair, automatic epoch replay, precision change or batch reduction.

Synthetic tests force backoff, recovery and scale growth with a shortened growth interval, including fused AdamW skip accounting. A separate four-attempt tiny CUDA fixture also verified the actual fused GPU kernel: three successful updates, one forced overflow skip, three EMA updates, backoff 128 to 64, recovery and growth to 128, plus CUDA optimizer step-state restoration. Neither fixture uses patient inputs. These checks verify the control flow; they do not establish 80-epoch stability of YOLOv8s. PyTorch describes this scaler behavior in its [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html). Actual pinned implementation and test evidence take precedence over assumptions about default behavior.

## Checkpoint contents and resume limits

Installed Ultralytics saves EMA and optimizer state in reduced precision, uses direct checkpoint writes, selects EMA when loading a model, and strips training state at normal completion. The diagnostic subclass also resets its scaler during setup. The official subclass overrides these specific behaviors while retaining the working native training loop, loaders, losses and validation.

Each official recovery checkpoint retains FP32 live model, EMA and optimizer state; AMP scaler; LR scheduler; completed zero-based epoch; EMA/update accounting; early-stop state; Python/NumPy/parent Torch CPU/CUDA RNG states; metrics history; the best EMA known at that point; and frozen identity. Saving the live model permits continuation from its actual weights rather than its EMA approximation. Best snapshots embedded in recovery checkpoints preserve the selection history even when deliberately rolling back to an older epoch. Completion preserves these files and uses the final epoch's existing validation; it does not strip them or run an additional validation for display.

Files are written to unique temporary siblings, flushed, then replaced atomically. A SHA-256/length sidecar detects incomplete or changed bytes. Replacing `last.pt` or `best.pt` first retains its previous verified pair as `last.previous.pt` or `best.previous.pt`. File and sidecar publication are two operations: a power loss between them is detected; use a verified predecessor or periodic checkpoint. Periodic files cannot be overwritten. Interrupted `.partial` files remain local evidence and are not recovery inputs. Storage hardware still determines the durability of a power-loss event.

Resume requires an explicit command, an official file inside this run, valid sidecar, matching identity, finite eight-class state, complete optimizer/scaler/scheduler/accounting and consistent epoch/CSV history. Stripped diagnostic checkpoints and completed epoch-80 checkpoints are rejected for resume. Checkpoints after completed epochs 60, 70 and 79 may resume under the matching new 80-epoch identity; a stale 60-epoch identity or saved argument budget is rejected. The launcher takes an exclusive process lock, restores live weights, optimizer, EMA/count, schedule, epoch position and the **saved scaler scale**, then writes into a new `resumes/<timestamp-id>/` segment. It cannot silently fall back to fresh COCO initialization. The existing history remains untouched; the new segment's native CSV begins with the selected checkpoint's completed history.

This is **epoch-boundary recovery, not bitwise continuation**. Native workers/shuffle generators are recreated; pending accumulated gradients are discarded and the native loop's local update-cadence offset is restarted. The checkpoint records how many pending microbatches were discarded. Parent RNG restoration does not restore every worker state or prefetched item. A continuation may therefore have different batches/updates than an uninterrupted run, even with seed 42. Report interruptions and actual update counts when comparing conditions. The generic [Ultralytics resume documentation](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings) is supplemented here by inspection and synthetic restoration tests against this installed version.

## Live progress and local outputs

After each completed training epoch and its existing validation pass, two flushed console lines show epoch/80, training and validation box/cls/DFL losses, precision, recall, mAP50, mAP50-95, current best, training/validation/whole-epoch seconds and cumulative successful/skipped updates. Whole-epoch timing includes checkpoint saving; separate training/validation timers measure those passes. Missing/non-finite values cause an explicit failure before a misleading summary/checkpoint is accepted.

`results.csv` remains native. `progress.log` contains the readable summaries; `epochs.jsonl` contains numeric summaries and training memory; `per_class.jsonl` retains all eight classes including fracture AP50/AP50-95 and recall. Per-class values come from the same validator, without another model pass. `events.jsonl`, frozen configuration/identity, launch metadata, checkpoint sidecars and completion/failure receipts stay local. These files and the medical artifacts remain Git-ignored.

## Verification performed

**74 automated tests pass**, covering 49 other tests and 25 protocol tests. The update adds checks for the exact configuration-only epoch change, 47,433 nominal attempts, native cosine schedule through 80, matching resume at completed epochs 60/70/79, rejection at 80 and stale identities, periodic save names through `epoch080.pt`, and `X/80` progress. New checks include frozen configuration/source tampering, insufficient VRAM, changed data identity, AMP overflow/backoff/growth/recovery and persistent limits, fused update accounting, non-finite losses/state, actual save hooks, FP32 checkpoint round trips, scaler/optimizer/EMA/scheduler restoration, interrupted publication and verified predecessor, completed-budget rejection, progress/CSV/per-class logging, path containment, run locking, overwrite protection and manual-start enforcement. `pip check` reports no broken requirements; the editable package and `wrist-train` entry point are installed.

The original bounded native GPU setup-only check is preserved unchanged in `setup_only.json`. It used **60 configured epochs**, with all other settings identical to the current configuration: real GPU YOLOv8s, frozen loaders, fused AdamW/betas/decay groups, accumulation 4 and initial scale 128. Both `train()` and `_do_train()` were blocked. It performed zero backward passes, optimizer updates, training epochs and checkpoint writes. The receipt records 32 MiB allocated at its final in-process check, followed by process exit. It is historical setup evidence, not a newly performed 80-epoch setup. The current update tests the native 80-epoch cosine function and trainer guard with synthetic CPU fixtures, then runs actual CUDA/data preflight. No additional native model setup, GPU benchmark or smoke epoch was needed.

The updated **80-epoch preflight passed** in 158.03 seconds: **34,298 prepared files** (14,061 training and 3,088 validation image/label pairs), zero patient overlap, unchanged frozen dataset/class/COCO identities, exact pinned environment, **4.944 GiB free VRAM** and **105.280 GiB free disk**. Configuration, source and framework hash checks passed against the regenerated 80-epoch lock. The CUDA probe left zero allocated tensor bytes; the output permissions probe passed without creating the official run root. Warnings identify epoch-boundary resume limits and unmeasured 80-epoch stability. No held-out test image or label was opened for this check. Start/resume repeat preflight before entering training; allow several minutes for prepared-file hashes.

Receipts: [historical setup-only](setup_only.json), [automated verification](verification.json), [synthetic CUDA](synthetic_cuda.json), [preflight](preflight.json), [timing/update estimates](duration_estimates.json). The official run directory was still absent at completion of these checks. Prepared image/label hashes were checked, without re-decoding raw images, changing splits, converting files or evaluating test images.

## Timing and remaining limits

The measured diagnostic training pass took **295.93 s**, and trainer-integrated validation took **38.22 s**, or **334.15 s per epoch** before additional official checkpoint/monitoring overhead. Linear extrapolation gives **7.43 hours (about 7 h 26 min) per 80-epoch model**, or **8.54 hours (8 h 32 min) with 15% allowance**. Both models total **14.85 hours (14 h 51 min)**, or **17.08 hours (17 h 5 min)** with allowance. The existing budget calculator verifies 47,433 nominal attempts per model; AMP skips can reduce successful updates. This calculation uses the measured pass times rather than multiplying the one-off smoke startup/reloaded-validation overhead by 80. There is no extra final validation in the official trainer.

Phase 2A measured training peak 1.905 GiB allocated / 2.025 GiB reserved, validation reserve peak 2.486 GiB, and sampled total-device peak 2,985 MiB. These measurements support batch 6; they do not guarantee identical future peaks. Preflight requires at least 3.5 GiB free VRAM, warns below 4 GiB, and requires 5 GiB free disk. Full-state checkpoints are substantially larger than the approximately 22.5 MB stripped smoke checkpoints; resume segments retain additional files and need continuing disk space.

The 15% allowance is a planning estimate, not a measured upper bound. The first official epoch has more optimizer attempts during warmup; finite checks, FP32 checkpoint publication, long-run thermal/power behavior and future class-aware sampling/cropping may add overhead. Scaler growth beyond the diagnostic's 586 updates was tested synthetically, not through another YOLO epoch. Keep the laptop powered and prevent sleep during the manual run. Failed runs and any AMP skips must remain available for later analysis. No test outcomes were used to set this protocol, and no clinical reliability claim follows from these checks.

## Exact manual commands

Run from this repository root in PowerShell. Passing preflight does not start training:

```powershell
.\.venv\Scripts\wrist-train.exe preflight
```

After reviewing this report, launch the official **standard** baseline deliberately:

```powershell
.\.venv\Scripts\wrist-train.exe start --confirm-standard-baseline
```

Its stable local root is `outputs/training/standard_baseline/`. A new start refuses an existing root. In a second terminal:

```powershell
.\.venv\Scripts\wrist-train.exe status
.\.venv\Scripts\wrist-train.exe monitor
```

The monitor prints completed-epoch summaries and follows new resume segments. The training terminal also shows native batch progress and AMP events. Ctrl+C stops the monitor only; use the training terminal to interrupt training. After an interruption, first confirm the prior process has exited, then:

```powershell
.\.venv\Scripts\wrist-train.exe preflight --resume --checkpoint last
.\.venv\Scripts\wrist-train.exe resume --confirm-resume-standard-baseline --checkpoint last
```

`last` resolves within the active segment named by `active_run.json`. If that pair is incomplete/corrupt, explicitly inspect and select `last.previous` or a retained `epoch010`, `epoch020`, ..., `epoch070`. A periodic file from an earlier segment can be supplied relative to the stable root, for example `--checkpoint weights/epoch010.pt` for the original segment. A recovery from an earlier epoch intentionally rolls back later work. Automatic fallback is disabled. If the newest segment has not completed an epoch, use the earlier relative checkpoint path recorded in `active_run.json`.

The process lock refuses concurrent writers and archives a stale lock only after checking that its process is no longer active. If the very first run fails before any checkpoint exists, there is nothing to resume: preserve and manually archive/rename that failed run directory inside ignored outputs before a deliberate fresh start. Do not delete evidence or substitute smoke weights. Resume preserves earlier segments; never merge their CSV histories by blindly concatenating them. Completed epoch 80 is the end of this budget; `epoch080.pt` is retained for final-state inspection and cannot resume further.

**Neither confirmed launch command was invoked during Phase 2B.** The student will launch the first official run after reviewing this work.
