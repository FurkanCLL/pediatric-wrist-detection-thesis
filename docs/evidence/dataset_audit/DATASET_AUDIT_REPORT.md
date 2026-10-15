# GRAZPEDWRI-DX source audit

**Audit date:** 2026-10-15. **Scope:** source inspection only. No final splits, image conversion, prepared labels, training or model evaluation were produced. The two-training by three-inference design is unchanged.

## Source and counting method

The authoritative release is [GRAZPEDWRI-DX v2](https://doi.org/10.6084/m9.figshare.14825193.v2), published on 2022-05-06 under CC BY 4.0. Release metadata and published file checksums were retrieved from the [pinned Figshare API](https://api.figshare.com/v2/articles/14825193/versions/2). The local `dataset.csv` matches the official MD5, `f2e996300443654faa62983b7ef1b4c1`.

The official `folder_structure.zip` also matches its published MD5, `e484b434286428750e9c838a7ed6ada7`. Every one of its 60,986 files matches the extracted local counterpart byte for byte, including all annotation exports, class definitions and notebooks. The archive was read without extraction over raw data. Together with 20,327 images and the CSV, these account for all 81,314 local source files.

The primary annotation source is `supervisely/wrist/ann/`, with shapes defined in `supervisely/meta.json`. Detection statistics count each rectangle or polygon once; a polygon contributes its exterior bounding envelope for box-size analysis. Axis lines and image tags are counted separately. Pascal VOC and YOLOv5 are comparison exports, not additional objects. Source polygons, lines, images and metadata are preserved.

The scripts are under `src/wrist_audit/`. Run commands and installation instructions are in the root README. Small aggregate evidence accompanies this report. Detailed file hashes, image/object records, patient associations, issue records, pixel samples and visual-review selections remain under ignored `outputs/dataset_audit/`.

## Composition and annotation support

| Unit | Count |
|---|---:|
| Original image files | 20,327 |
| Metadata rows | 20,327 |
| Released patient IDs | 6,091 |
| Distinct `(patient_id, study_number)` groups | 10,699 |
| Source objects, including lines | 67,771 |
| Axis lines | 20,327 |
| Nine-category detection objects | 47,444 |
| Objects in the proposed eight targets | 47,436 |
| Images without detection boxes | 15 |

Class support below is dataset-wide descriptive evidence. It must not be used directly to freeze a class-aware training policy: recompute the relevant statistics on training data after the split is approved and frozen.

| Source class | Source YOLO ID | Proposed detector ID | Objects | Images | Patients |
|---|---:|---:|---:|---:|---:|
| fracture | 3 | 0 | 18,090 | 13,550 | 3,587 |
| metal | 4 | 1 | 819 | 708 | 157 |
| text | 8 | 2 | 23,722 | 20,274 | 6,089 |
| periostealreaction | 5 | 3 | 3,453 | 2,235 | 1,089 |
| pronatorsign | 6 | 4 | 567 | 566 | 555 |
| bonelesion | 1 | 5 | 45 | 42 | 24 |
| softtissue | 7 | 6 | 464 | 439 | 405 |
| boneanomaly | 0 | 7 | 276 | 192 | 95 |
| foreignbody | 2 | Excluded, proposed | 8 | 8 | 4 |

Image and patient counts overlap between categories and must not be summed. The human-readable targets “periosteal reaction”, “pronator sign”, “bone lesion”, “soft tissue” and “bone anomaly” correspond to the unspaced source spellings above. The eight targets are present, but the proposed detector IDs are not the source YOLO IDs. No mapping has been applied to raw labels.

Text has approximately 527 times as many objects as bone lesion. Bone-lesion support is only 24 patients, and metal has repeated views/examinations across 157 patients. Image-level oversampling or a nominally balanced image split would conceal this dependence. Rare-class AP will require careful support reporting and cautious interpretation; this audit does not predict model performance.

## Metadata and examination grouping

The metadata joins images by `filestem`. The CSV has a UTF-8 BOM, fractional ages and two records with gender code `O`; these are supported source representations. Filename ages encode completed years. The observed age range is 0.2–19.0 years, so the cohort must not be described as uniformly younger than 18.

`study_number` is patient-local. It forms 10,699 patient-study groups: 1,271 have one image, 9,319 have two, 18 have three, and 91 have four. Study numbering is contiguous within every patient. There are 143 groups with both lateralities and 110 with repeated side/projection pairs; these are review findings, not proof of erroneous records.

The [dataset publication](https://doi.org/10.1038/s41597-022-01328-z) reports 10,643 studies. The released metadata yields 56 more groups. The exact official CSV checksum matches, so this discrepancy must be retained rather than “corrected” to the published total. `timehash` gives 20,327 distinct patient-time pairs and is unsuitable as an examination key. The underlying reason for the examination-count difference is unresolved. It does not require an image-level split: patient grouping keeps all these views and examinations together.

Blank tag fields are not universally missing identities or clinical negatives. Source image tags include AO classification (14,158), initial examination (10,861), cast (5,776), osteopenia (2,473) and diagnosis uncertainty (537). In the CSV, `fracture_visible=1` occurs in 13,550 images and `metal=1` in 708, matching the respective source-box image counts. These fields do not turn the 15 empty-box images or the remaining non-fracture images into a verified healthy cohort.

## Geometry and annotation semantics

`bonelesion` and `periostealreaction` are polygon classes. The other seven detection categories are rectangles. Axis is defined as a line: 20,316 records have two points, ten have three and one has four. These 11 nonstandard axis records remain inside image bounds and are outside the eight detection targets. Converting polygon envelopes to detection boxes later will include some surrounding pixels, so the derived task is box detection, not segmentation. Axis lines and AO/other image tags should remain outside the eight-class box mapping.

Every VOC file has an empty `<size/>` element. All 47,444 VOC boxes match the primary source within one pixel; the maximum observed coordinate difference is 0.999972 pixels, consistent with integer export rounding. The VOC `difficult` and `truncated` flags are zero for all boxes. This does not establish clinical certainty; 537 source images separately carry the diagnosis-uncertainty tag.

The YOLO export has **47,443 boxes**, including **818 metal boxes**, whereas Supervisely and VOC both have 819 metal boxes. One source metal box is absent from YOLO. All remaining YOLO objects match within one pixel, with maximum observed coordinate difference 0.001476 pixels. There are 491 boxes with tiny border excursions attributable to normalized-coordinate rounding (accepted reporting tolerance 0.002 pixels). They are reported without modifying the source. **Recommendation:** derive future labels from Supervisely using its dimensions, retaining traceability; do not copy the existing YOLO export as if it were complete.

All 47,444 source detection envelopes have positive area and lie within their declared image edges. The following sizes use original-coordinate envelopes. The 1024 column is an idealized aspect-preserving long-edge scale, before padding or integer rounding; it is not an implemented image conversion or a chosen model input size.

| Source class | Median image-area share | Median short side at 1024 | Boxes with short side below 16 at 1024 |
|---|---:|---:|---:|
| fracture | 1.648% | 79.07 px | 0 |
| metal | 9.956% | 149.08 px | 4 |
| text | 0.456% | 38.11 px | 611 |
| periostealreaction | 1.447% | 37.68 px | 173 |
| pronatorsign | 6.674% | 91.78 px | 0 |
| bonelesion | 0.315% | 37.49 px | 0 |
| softtissue | 6.426% | 109.46 px | 0 |
| boneanomaly | 1.944% | 90.06 px | 0 |
| foreignbody | 0.207% | 30.49 px | 1 |

Approximately 5.0% of periosteal-reaction boxes fall below this illustrative short-side threshold, compared with 2.6% of text boxes. This describes scale variation; it does not establish a detection failure threshold or demonstrate that refinement will improve AP. Bone lesions combine low patient support with smaller median envelopes than fractures. Do not select class-aware targets, crop parameters or inference thresholds from eventual test outcomes.

Four original images were viewed for fracture, bone-lesion, foreign-body and empty-box coverage. They rendered legibly, with variable contrast and visible acquisition borders/markers. This was a technical spot check only, not clinical adjudication, a prevalence sample, or proof that all annotations are correct. The selections and review receipt remain local; no medical images are included in the public report.

## Image representation and integrity

All 20,327 image files passed PNG header/chunk-integrity inspection and SHA-256 hashing. Every file is 16-bit, single-channel grayscale (PNG color type 0, Pillow mode `I;16`). All image dimensions agree with Supervisely. Exact byte hashing found no duplicate images, including none across patient IDs. No files or records are missing from the image/metadata/three-annotation-format joins, and no duplicate stems or exact duplicate metadata rows were found.

| Image property | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Width | 212 px | 549 px | 1,664 px |
| Height | 324 px | 999 px | 2,328 px |
| Width/height ratio | 0.2480 | 0.5572 | 1.6793 |

There are 19,039 distinct dimension pairs. Width and height medians are separate marginal statistics, not the dimensions of a particular image. This variation supports preserving aspect ratio in later preparation.

The deterministic decoding subset contains **534 images (2.63%)**: a base of 512 ordered by SHA-256 of the image stem, extended for rare categories and empty-box coverage. All decode as two-dimensional `uint16` arrays; none are constant. All sampled minima are zero; maxima range from 65,534 to 65,535. Per-image means range from 7,655.60 to 41,281.02, with median 20,056.65. The median per-image 99th percentile is 47,905. The median zero-pixel fraction is 4.70%, with maximum 59.29%, showing that global statistics can be strongly affected by black background/borders. These are statistics over the sampled images, not pooled or exhaustive pixel statistics. Their near-full 16-bit range must not be mistaken for an already usable 8-bit detector input.

## Decisions needed before preparation

1. **Split policy:** a proposed starting point is 70/15/15 by patient, with one fixed seed and a documented patient-level multilabel allocation rule. Keep every examination and view of each patient together. Check rare-class patient support and exact duplicates before freezing manifests. Ratios and allocation policy require review; no split has been created. Report the achieved image proportions rather than promising exact image ratios.
2. **Foreign body:** keep its eight original annotations traceable. A simple proposed eight-class policy is to retain the images and omit only the foreign-body labels, explicitly accepting these out-of-scope objects as unlabeled background for the detector. This is not an ignore-region mechanism. If that is unacceptable, agree an explicit image-exclusion policy instead. Do not crop, mask, or relabel source images silently.
3. **Export authority:** approve deriving labels from Supervisely, preserving its additional metal annotation and polygon envelopes. Record the empty VOC dimensions, YOLO omission and border-rounding differences; no raw detection envelopes require clipping. Preserve axes separately and leave them out of detector labels. No source correction or clinical relabeling has been performed.
4. **Image preparation:** choose and validate a deterministic 16-bit-to-detector input policy later. This audit decodes source values without producing converted images. Preserve geometry and original intensities; hardware feasibility and the common augmentation/training policy remain separate later tasks.

The split must be frozen before model development, and test data must remain outside iterative method decisions. The rare-class allocation check is a construction check, not permission to try seeds until downstream test performance improves. The later class-aware policy must use training-only frequencies and geometry.

## Verification coverage and limits

The accompanying `aggregate_statistics.json` records exact coverage, geometry/intensity distributions, export comparison tolerances and finding counts. `source_release.json` records official release checksums and provenance coverage. `class_statistics.csv` contains the compact class table and geometry measures. `validation.json` records code revisions, test results and the two verification batches.

The implementation passed 25 focused synthetic tests, including BOM/metadata association, source-versus-proposed IDs, geometry, 16-bit decoding, empty VOC sizes, archive checks and public-evidence safeguards. Dependency import/package checks passed in the existing Python 3.12.5 environment with NumPy 2.3.5 and Pillow 12.3.0.

Validation used two explicit batches. The full local scan completed image integrity, hashing, metadata and primary geometry checks. Its original VOC parser assumed dimensions existed and stopped the export comparisons; the local receipt preserves those 20,327 tooling errors. The parser was corrected and regression-tested, then all VOC and YOLO exports were checked from the official archive already proven identical to the extracted files. `export_crosscheck.json` records that complete second batch. The aggregate export discloses the resolved parser defect and retains current source findings; it does not present the first attempt as an error-free run. There was no reason to repeat unchanged image hashing after an XML-parser-only correction.

The raw file-size/modification-time snapshot was unchanged during the full scan. Its local content-manifest SHA-256 is `d275fbeae0ad5644f25fb69ab337e2b5e4baa3e7293fb565813bc17b7832b5ca`. Hashes support reproducibility without publishing the individual file manifest.

The four original image ZIP checksums are not verified by this workflow. PNG integrity and a new local content manifest establish readability and a reproducible local snapshot; they do not prove byte identity with the remote image archives. No all-pairs near-duplicate search or radiologist review was performed. Pixel decoding is sampled, so chunk-integrity checks must not be presented as exhaustive pixel decoding. No clinical reliability or model performance is established.
