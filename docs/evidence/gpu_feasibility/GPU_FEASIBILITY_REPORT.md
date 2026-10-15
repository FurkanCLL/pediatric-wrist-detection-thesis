# GPU feasibility and training protocol validation

**Phase 2A, 2026-10-15.** CUDA setup, a bounded 768-resolution benchmark and one complete diagnostic training epoch passed. Both saved checkpoints reload correctly. No official standard/class-aware experiment has started, and the common protocol remains a proposal for student review. The frozen dataset and approved two-training by three-inference design are unchanged.

## Hardware and environment

| Item | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6 GiB VRAM |
| Driver / CUDA wheel runtime | 560.94 / 12.6 |
| CPU | Intel Core i5-12450H, 8 physical cores / 12 logical processors |
| RAM / OS | 16 GB nominal; 15.73 GiB reported physical memory / Windows 11 Home |
| Python | 3.12.5 |
| PyTorch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 |
| Ultralytics / cuDNN | 8.4.177 / 9.10.2 |

Only the PyTorch and torchvision build variants changed. Their base versions were preserved, with no unrelated package upgrades. PyYAML 6.0.3 was already installed and is now explicitly declared because the new code imports it. `requirements.txt` and `pyproject.toml` remain consistent; `requirements-cuda.txt` selects the official CUDA wheels. Before/after package lists and installation reports remain local.

The first sequential wheel transfer was very slow and was stopped before package replacement. The same official wheels were obtained through parallel HTTP ranges and verified against the complete SHA-256 values published in the [official PyTorch CUDA 12.6 index](https://download.pytorch.org/whl/cu126/torch/). The torchvision index and both hashes are in `cuda_wheels.json`. No driver or system CUDA toolkit installation was needed; CUDA compatibility context is available in [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

`torch.cuda.is_available()` is true. GPU convolution forward/backward, a real AdamW update and torchvision CUDA NMS passed. The verification returned allocated tensor memory to zero after cleanup. `pip check` reports no broken requirements. The completed training process exited successfully and released its GPU resources.

Initialization uses the official `ultralytics/assets` v8.4.0 `yolov8s.pt`, verified as the 80-class COCO detection checkpoint with 11,166,560 parameters. Its SHA-256 is `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`. The eight-class model has 11,138,696 parameters; 349/355 pretrained state entries transfer, with class remapping disabled. The small `yolo26n.pt` asset is used only by the framework's native AMP compatibility check. The [official YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) describes the COCO-pretrained detector family. No GRAZPEDWRI-trained initialization was used.

## Frozen inputs and benchmark

The existing preparation receipts, source-record hashes, split manifest and eight-class YAML were checked without repeating conversion or creating new splits. The split-manifest SHA-256 remains `ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49`. Diagnostic YAML files contain train and validation paths only. Test images/labels were not loaded for model evaluation or configuration decisions.

The benchmark selected 180 training images by a fixed seed-42 hash order, without selecting for predictions or scores. It used YOLOv8s at 768, physical batch 6, two training workers, uncached images, conservative augmentation, forward/loss/backward, AdamW and EMA updates. Five warmup steps preceded twenty measured steps. Every benchmark iteration updated the optimizer to exercise its state and cost. This differs deliberately from four-batch accumulation in the smoke test.

| Corrected benchmark | Measurement |
|---|---:|
| AMP / initial scaler value | enabled / 128 |
| Successful optimizer updates | 25 / 25 |
| Steady iteration mean | 0.1689 s |
| Steady image throughput | 35.53 images/s |
| Mean direct loader wait | 0.356 ms |
| Peak allocated / reserved | 1.829 / 1.885 GiB |
| Full training-epoch extrapolation | 6.60 min |
| Gradient overflows / skipped updates | 0 / 0 |

Batch 6 passed with substantial memory headroom, so batches 5 and 4 were not needed. No OOM occurred. The optional 1024 probe was omitted to preserve time and the student's explicit 768 preference.

## AMP failure, diagnosis and correction

The first batch-6 benchmark used the native initial GradScaler value, 65,536. Forward losses were finite, but scaled backward gradients overflowed. Five successive backoffs reached the diagnostic guard before any successful optimizer update; the failed attempt and traceback remain local as `benchmark_b6_amp_01`. The diagnostic guard was stricter than waiting indefinitely for native scale calibration. This failure alone does not establish persistent AMP instability.

A bounded comparison restored the same model weights, running statistics and augmented training batch before every precision trial. FP32 gradients were finite. AMP gradients were non-finite at scales 65,536, 2,048 and 512, but finite at 128 and 1. The AMP forward loss was identical across those scales. No optimizer updates occurred in this comparison. This isolates backward loss scaling as the reproduced cause.

The targeted correction sets the initial scale to **128** in the project trainer. Resolution, physical batch, weights, optimizer, learning rate, labels and augmentation were unchanged. The corrected benchmark `benchmark_b6_amp_02` passed all 25 updates. The subsequent full smoke test passed all 586 updates with **no overflow, scale backoff or skipped update**. AMP is therefore recommended as a candidate with this explicit initial scale. `runtime.amp_initial_scale` is consumed by `CheckedTrainer`; passing only the native training arguments to plain `YOLO.train()` would omit this correction.

Native scaler growth/backoff behavior otherwise remains unchanged: growth factor 2, backoff factor 0.5 and growth interval 2,000 successful updates. Growth was not reached during this one-epoch test. Longer-run stability remains unproven; the strict diagnostic overflow guard is separate from the official run/recovery policy that still needs review. No installed Ultralytics source files were patched.

## Completed one-epoch smoke test

The run `smoke_b6_amp_01` started from fresh COCO weights and the committed scale correction (`ef79a55`), with no uncommitted source changes at launch. It processed every training image exactly once and produced one finite metrics row in `results.csv`. Its checkpoint and detailed metric values remain ignored local diagnostics, not an official baseline or evidence of convergence.

| Stage or check | Measured outcome |
|---|---|
| Training coverage | 14,061 images / 2,344 batches / exactly one epoch |
| AdamW updates / parameter changes | 586 / 586 |
| Training duration / throughput | 295.93 s (4.93 min) / 47.51 images/s |
| Trainer-integrated validation | 3,088 images / 258 batches / 38.22 s |
| Reloaded-checkpoint validation | 3,088 images / 23.41 s |
| Complete run wall time | 398.60 s (6.64 min), including setup and both validations |
| Training allocated / reserved peaks | 1.905 / 2.025 GiB |
| Validation allocated / reserved peaks | 0.682 / 2.486 GiB |

Training losses, gradient norms, parameters, raw validation predictions and trainer-integrated validation losses passed finite checks. Both approximately 22.5 MB `last.pt` and `best.pt` checkpoints were saved and reloaded. Independent completion verification checked all **355 parameter/buffer tensors per checkpoint**, the eight-class names, three input channels and unchanged checkpoint hashes. Native final validation also exercised the reloaded checkpoint. The interruption arrived after successful completion; no benchmark, epoch or validation pass was repeated afterward.

Validation used batch 12, four workers, rectangular letterboxing and no augmentation. Its higher reserved-memory peak includes allocator caching; it is distinct from allocated live tensors. GPU telemetry over the full smoke lifecycle sampled a maximum **2,985 MiB total device memory**, leaving roughly **3.1 GiB** on this GPU. Average utilization was 68.0% including setup/validation, with peak 94%; sampled maximum temperature was 73 C. These are sampled device observations, not an isolated steady-state utilization or exact device-memory maximum.

## Proposed common protocol, subject to review

| Setting | Candidate for both future training conditions |
|---|---|
| Model / initialization / dataset | YOLOv8s / same official COCO checkpoint / same prepared base and frozen patients |
| Input / physical batch / accumulation | 768 / 6 / 4; nominal and steady effective batch 24 |
| Precision | AMP enabled, initial scale 128; monitored gradients and validation |
| Optimizer | explicit AdamW; LR 0.001, betas (0.9, 0.999) |
| Weight decay | 0.0005 on ordinary weights; zero on normalization weights and biases |
| Schedule | cosine schedule with `lrf=0.01`, one requested warmup epoch |
| Budget | **30 epochs proposed**, 14,061 draws per epoch, approximately 18,133 optimizer updates |
| Early stopping | disabled (`patience=0`) to preserve a matched fixed budget |
| Workers / cache | 2 training workers, 4 validation workers / `cache=False` |
| Validation / checkpoints | every epoch; EMA validation; native best eight-class mAP50-95, plus last checkpoint; latest exact tie |
| Seed / reproducibility | 42; deterministic mode, CUBLAS workspace `:4096:8`, recorded framework/source versions |

The actual one-epoch smoke had **zero warmup iterations** because this framework caps warmup at `epochs - 1`. Its constant first-epoch LR was 0.001. The longer candidate's warmup and cosine schedule have not been exercised end to end. Source-based cadence calculations predict 1,139 updates in the warmup epoch and 586 per subsequent epoch, producing 18,133 updates over 30 epochs. These are calculations, not additional training runs.

For AdamW, the configured `warmup_momentum=0.8` does not alter beta1: this version only ramps groups containing an SGD-style `momentum` field. Beta1 remains 0.9.

Use explicit AdamW rather than `optimizer=auto`: the installed automatic policy can change optimizer and LR with the total iteration estimate. `nbs=24` gives exactly four-step accumulation at batch 6 and leaves the requested weight decay unchanged. Framework defaults such as `nbs=64` would change those details. Deterministic mode uses warning-only handling for unsupported deterministic operations, so cross-driver bitwise identity is not guaranteed.

Common on-the-fly augmentation is rotation +/-10 degrees, translation +/-5%, scale 0.9-1.1 and HSV-value gain +/-10%. Hue and saturation are zero; the value transform was exercised on equal grayscale channels and changed all 16 synthetic trials while retaining equal channels. Horizontal/vertical flips, Mosaic, MixUp, CutMix, CopyPaste, shear, perspective, BGR swapping and optional Albumentations transforms are explicitly disabled. The empty optional-transform list prevents hidden blur/CLAHE defaults. The normal box clipping/filtering of the training affine transform remains common to both conditions. Prepared images/labels are unchanged.

The future class-aware strategy is still unimplemented. Prefer the same 14,061 draws per epoch and the same accumulation schedule for it, changing only the intended sampling/crop policy. If its draw count differs, equal epoch counts will not ensure equal updates; agree an explicit update budget and aligned schedule before launch. Sampling weights and crop parameters must use training statistics only. The checkpoint rule, official evaluator and failure/recovery policy also require final review.

## Cost estimates and remaining decisions

A measured training epoch plus full trainer-integrated validation is **5.57 minutes**. The following are linear estimates from this one diagnostic epoch, assuming comparable draw counts and loading cost in both conditions:

| Epochs per model | One model | Both models | Both, with 15% planning margin | Updates per model with proposed warmup |
|---|---:|---:|---:|---:|
| 10 | 0.93 h | 1.86 h | 2.13 h | 6,413 |
| 20 | 1.86 h | 3.71 h | 4.27 h | 12,273 |
| 30 | 2.78 h | 5.57 h | 6.40 h | 18,133 |

Allow one final checkpoint validation and startup per model in addition to the simple extrapolation. The 15% column is a planning allowance, not a confidence interval. Additional optimizer steps during warmup, thermal changes, CPU work and future class-aware crop/sampling overhead can change wall time. The short benchmark's 6.60-minute training estimate was conservative because it updated AdamW every iteration. The full smoke's measured 4.93-minute training time is the better current basis for accumulated training.

A **30-epoch matched budget** appears practical at roughly **6.4 hours for both models with the margin**. A 20-epoch alternative is about **4.3 hours with the margin** if the available window is tighter. No convergence claim or accuracy-based choice follows from one epoch. GPU forward/backward accounts for most measured step time; direct dataloader wait was small in the short subset probe. The full lifecycle utilization also includes setup and validation, so it should not be interpreted as pure training occupancy.

The student must approve the resolution/batch/AMP choice, optimizer/schedule, final epoch/update budget, checkpoint/evaluation rule and official failure policy before any official launch. Class-aware design remains a separate training-only task. The one-epoch evidence does not validate clinical reliability, sustained multi-epoch AMP behavior, a 1024 configuration or a new class-aware sampler. The earlier dataset limitations remain in the preparation/audit reports. Formal supervisor approval of the revised thesis scope remains pending.

## Evidence and verification

The accompanying compact JSON files record hardware/software, official wheel and checkpoint provenance, benchmark timing/memory, same-batch precision checks, one-epoch and reload checks, duration/update calculations and the preserved failed attempt. Public records contain aggregate diagnostics and hashes only. Patient/image manifests, medical files, private settings, checkpoints, detailed logs and one-epoch AP/loss artifacts remain under ignored local directories.

**49 focused tests passed**, and package consistency checks passed after the scale fix. The first test invocation encountered Windows sandbox temporary-directory cleanup denial; the successful verification used a project-local temporary directory. This did not require changing dataset or model code. The final checkpoint/CSV/frozen-manifest verification reused the existing completed run.

The corrected benchmark was tested before its fix was committed; its early receipt records the preceding HEAD and config hash. The final smoke ran from clean `ef79a55`. `provenance.json` distinguishes these records and hashes the installed trainer, validator, augmentation, metric and data-loading sources. All successful and failed attempts are retained. This phase ends at a reviewable candidate protocol; no official A-F results have been generated.
