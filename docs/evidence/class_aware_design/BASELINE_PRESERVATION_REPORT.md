# Standard baseline preservation

Verified on 2026-10-16. The existing official standard YOLOv8s completed all 80 epochs. This inspection ran no model inference or training and changed no baseline artifact.

| Item | Verified result |
|---|---|
| Selected validation checkpoint | Completed epoch 22 |
| Overall validation mAP50 / mAP50-95 | 0.59908 / 0.33270 |
| Fracture AP50 / AP50-95 | 0.92116190 / 0.51973089 |
| Optimizer attempts / successful / skipped | 47,433 / 47,416 / 17 |
| Completed artifacts / checkpoints | 36 / 12 |
| Resume segments | 0 |
| Observed scaler range in recorded AMP events | 64–8192 |

Best checkpoint SHA-256: `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`.

Frozen protocol SHA-256: `281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01`.

Frozen split SHA-256: `ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49`.

COCO initialization SHA-256: `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a`.

Private full-run inventory SHA-256: `9e2d9b5316d816e2275f96a436367552c42be9dd6f5a0afcd887826b12884982`.

The [aggregate verification receipt](baseline_preservation.json) records every checkpoint's byte count, SHA-256, sidecar and saved completed epoch. Each checkpoint was loaded on CPU from verified bytes, with matching frozen identity, eight-class mapping, FP32 live/EMA/best model parameters, finite state tensors and optimizer/scaler/scheduler state. Saved training parameters match the frozen protocol apart from the launcher's deliberate output-overwrite argument. Scheduler epoch, EMA/update counts, 80 CSV rows, per-class logs, completion receipt and all 17 logged AMP skips reconcile. The selected checkpoint's own metadata, metrics and per-class values match epoch 22, rather than relying on a filename or reported score alone.

A complete content inventory of the existing run was captured locally, then rechecked unchanged. All original checkpoints, sidecars, logs, native CSV, class metrics and configuration/provenance files remain at their existing paths. The frozen protocol lock and all covered baseline implementation files remain valid; exploratory code lives separately in `src/wrist_design/`. Dataset images, patient records, complete run logs, private inventory and model weights are not published.

These are validation-selected baseline results, not held-out test or clinical results. Per-class scores have limited independent support, especially bone lesion. The best epoch is selected using overall eight-class mAP50-95; fracture AP is reported at that same checkpoint, not selected separately. Baseline AP values do not enter candidate weights, crop geometry or simulation selection. The proposed intervention will later require fresh initialization from the same verified COCO checkpoint, never these trained baseline weights.
