# Dataset preparation and patient-level splits

**Phase 1B, 2026-10-15.** The common eight-class dataset is prepared and technically validated for the next feasibility stage. There are no training runs, model results, SAHI evaluations or adaptive-refinement implementations. The two-training by three-inference design is unchanged.

## Frozen split and support

The target patient proportions are 70/15/15 with **seed 42**. Every view and examination of each released `patient_id` stays in one split. Neither image identity nor `timehash` is an independent allocation unit.

| Split | Patients | Images | Image share | Target boxes | Empty labels |
|---|---:|---:|---:|---:|---:|
| train | 4,264 | 14,061 | 69.17% | 32,871 | 11 |
| val | 914 | 3,088 | 15.19% | 7,204 | 2 |
| test | 913 | 3,178 | 15.63% | 7,361 | 2 |
| Total | 6,091 | 20,327 | 100% | 47,436 | 15 |

The patient capacities are 4,264 / 914 / 913, obtained by largest-remainder rounding of the requested proportions. The image shares differ by less than one percentage point from the targets; patient independence takes priority over exact image counts. There are 9,307 / 2,082 / 2,161 images with fracture boxes and 4,754 / 1,006 / 1,017 without fracture boxes. Absence of fracture boxes or an empty detector label does not establish a healthy image.

The deterministic allocator uses patient-level presence of the eight classes, any empty-target image, and any image without fracture boxes. It repeatedly selects the rarest remaining feature and processes its patients in seed-shuffled order. It selects the split with the largest remaining target for that feature, then considers normalized deficits for the patient's other features, image deficit, patient capacity and seeded ties. Holdout feature targets use nearest-integer proportions, symmetrically for validation/test; the remainder goes to training. Actual patient capacities remain hard limits. A construction safeguard rejects image shares more than three percentage points from their targets before freezing.

An initial implementation sorted larger patient groups first, which biased the training image share to 77.70%. This construction was rejected before conversion and retained in ignored local outputs. Seed-shuffled patient order corrected that implementation defect, with a synthetic regression check. Seed 42 was never changed and no model outcomes were available or used. The final allocation was reproduced with reversed input order, then frozen before the pilot. No allocation changes followed conversion or validation.

In the table below, each cell is **objects / images / independent patients**. Categories overlap within images and patients, so their image/patient counts must not be summed.

| Detector ID and class | Train | Validation | Test |
|---|---:|---:|---:|
| 0: fracture | 12,422 / 9,307 / 2,511 | 2,808 / 2,082 / 538 | 2,860 / 2,161 / 538 |
| 1: metal | 600 / 518 / 109 | 96 / 88 / 24 | 123 / 102 / 24 |
| 2: text | 16,388 / 14,030 / 4,264 | 3,611 / 3,075 / 913 | 3,723 / 3,169 / 912 |
| 3: periostealreaction | 2,497 / 1,590 / 763 | 493 / 316 / 163 | 463 / 329 / 163 |
| 4: pronatorsign | 400 / 399 / 389 | 83 / 83 / 83 | 84 / 84 / 83 |
| 5: bonelesion | 35 / 32 / 16 | 5 / 5 / 4 | 5 / 5 / 4 |
| 6: softtissue | 324 / 308 / 283 | 72 / 67 / 61 | 68 / 64 / 61 |
| 7: boneanomaly | 205 / 147 / 67 | 36 / 21 / 14 | 35 / 24 / 14 |

Bone lesion has only **four patients and five boxes in each holdout**, despite deliberate patient-level representation. Bone anomaly has 14 patients in each holdout; metal has 24. These are useful but limited independent supports, especially for bone lesion, and future per-class AP will need cautious interpretation. No patients or images were duplicated to increase support.

The split-manifest SHA-256 is `ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49`. The manifest itself, image-to-patient associations and source-object records remain local under `outputs/dataset_preparation/`. `frozen_manifest_hashes.json` publishes only hashes and aggregate counts. The test partition is now held out from method development, tuning and model selection. Its present use was limited to split construction and technical data-integrity/loading checks.

## Annotation preparation

The student approved Supervisely as the source authority. The detector IDs are exactly the eight IDs in the table above, rather than the original nine-class YOLO order. Rectangles retain their original extent; polygons become their minimum enclosing axis-aligned boxes. Coordinates use the original width/height and continuous image edges. Malformed, non-finite, degenerate or out-of-bounds target geometry causes an error; no source clipping or correction is applied. Exported normalized coordinates retain round-trip double precision.

All 47,436 valid target objects are preserved, including **819 metal objects**, recovering the object absent from the original YOLO export. Eight `foreignbody` boxes are omitted while all eight affected images and their other target annotations are retained. Their four patients happen to belong to training under the fixed allocation; this excluded category was not a stratification target and no reallocation was made. These objects are unlabeled background for the detector, not ignore regions. All 20,327 axis annotations and image-level tags stay outside detector labels, including the 11 multipoint axis records noted in Phase 1A. Fifteen images receive zero-byte label files.

Private source records retain the original object index, source class, bounding envelope, original dimensions and source image/annotation hashes. Label row order follows the retained source-object order, so each output row can be traced to an original object. The immutable source annotations remain recoverable.

## Pixel preparation and pilot

Each original produces one **lossless-file-format 8-bit grayscale PNG** at the original dimensions, aspect ratio, orientation and pixel positions. This necessarily loses intensity precision relative to the 16-bit source. The fixed conversion is:

```python
prepared = ((source.astype(np.uint32) + 128) // 257).astype(np.uint8)
```

Since `65535 / 255 = 257`, this is nearest-integer linear mapping of the complete unsigned range: 0 maps to 0 and 65535 to 255. Integer source values cannot fall exactly halfway between output levels. `uint32` prevents overflow. PNG uses compression level 6 without optimization or image-dependent metadata. There is no percentile stretching, enhancement, resizing, cropping, rotation or offline augmentation.

A deterministic pilot selected **16 training images from 15 patients** before full conversion. It covered all eight classes, empty labels, small/large dimensions, narrow/wide aspect ratios, low/high mean intensities and substantial black background. Four private comparison sheets were reviewed against source high-byte display previews. Contours, texture and annotated regions remained reasonably visible; low-contrast cast cases and borders remained present in both views. No obvious conversion-induced degradation was observed. This was a technical check, not clinical adjudication or proof of diagnostic equivalence. All pilot pixels and boxes also passed independent checks. Repeating the pilot reused all 16 outputs unchanged.

Prepared images and labels are under `data/prepared/grazpedwri_detection/`, in `images/{train,val,test}` and `labels/{train,val,test}`. The local ignored `dataset.yaml` declares the eight classes, resolves this root explicitly and requests three loader channels. Files are published atomically, conflicting existing content is rejected, and a local SQLite journal commits completion in blocks of 128 images. Interrupted uncommitted work can be verified/recomputed without overwriting different data. Incomplete private files and the image-before-label/journal interruption case are covered by focused tests.

## Verification and loading

Independent validation used OpenCV to decode **all 20,327 originals and prepared images**, rather than the Pillow conversion decoder. Every prepared image is two-dimensional `uint8`, PNG bit depth 8/color type 0, with its original dimensions. Every pixel equals the independent float64 reference `rint(source * 255 / 65535)`: **zero mismatches**. A synthetic test also checks all 65,536 possible unsigned 16-bit values.

Every generated label was parsed and compared with an independently reconstructed envelope from the original Supervisely points. All class IDs, 47,436 objects and 15 empty labels are correct. The maximum observed coordinate reconstruction difference was `2.273736754432321e-13` pixels, below the `1e-8`-pixel numerical tolerance. All original image and annotation hashes match Phase 1A. The complete 81,314-file raw size/modification-time snapshot is unchanged. Every original is assigned once, every output has a label, and patient overlap is zero.

The actual **Ultralytics 8.4.177** detection dataset scanned all three splits without dropping images or objects. Its native-size `IMREAD_COLOR` path exactly replicated each checked grayscale plane into three equal channels. Each split produced a checked `2 x 3 x 320 x 320` uint8 batch with augmentation disabled. The 320 setting is only a technical loader check; it does not select the future training resolution. No model was constructed, weights loaded or training performed. See the [official loader reference](https://docs.ultralytics.com/reference/data/base/) for channel-loading behavior; the installed package was also inspected and exercised directly.

**42 automated tests passed**, including 17 new preparation checks for patient grouping, deterministic allocation, image-balance regression, category mapping, polygon/box geometry, exhaustive pixel arithmetic, protected writes and restart behavior. Runtime pins match `requirements.txt` and `pyproject.toml`; `pip check` reports no broken requirements. The implementation revision and library/zlib versions are recorded in `environment.json`.

## Evidence, limits and next stage

Compact public evidence accompanies this report: aggregate split statistics, `class_distribution.csv`, frozen-manifest hashes, full validation and loader receipts, pilot verification/review, and **training-only** support/geometry in `training_statistics.json`. Patient/image manifests, medical samples, detailed mappings, framework caches/settings and the rejected construction remain ignored. The public records contain no patient IDs, image filenames or absolute local paths.

Both future training conditions must use this same prepared base and frozen split. The training-only statistics may inform the later class-aware strategy; no sampling weights, repetition caps, crops or augmentation parameters were chosen. The eight-class training and the three inference methods remain later work.

The dataset is **ready for the planned feasibility stage**. The current PyTorch 2.14.1 / torchvision 0.29.1 build is CPU-only, so a compatible CUDA build must be installed/verified before the RTX 4050 benchmark at 768 and 1024. Input size, AMP, physical batch/accumulation and the official training protocol remain open. No GPU benchmark or model training was run here.

The earlier source-audit limits still apply: the 10,699 patient-study groups differ from the published 10,643 studies, and the four original image-ZIP checksums were not verified. Patient grouping does not depend on resolving that examination count. Intensity quantization, annotation uncertainty and rare independent patient support remain study limitations; preparation does not establish clinical reliability. Formal supervisor acceptance of the revised thesis scope remains pending.
