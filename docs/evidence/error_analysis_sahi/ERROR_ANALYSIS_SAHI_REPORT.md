# Baseline error analysis and SAHI feasibility

Phase 3B. Measurements: 2026-10-16; final technical review: 2026-10-17. **Neither predefined SAHI variant improves AP.** Both recover additional validation objects at the declared operating point, but introduce many more false positives, reduce fracture AP and increase latency. The evidence supports investigating selective, context-preserving refinement with guarded fusion. These are validation/development findings, not held-out or clinical results.

## 1. Scope and research questions

This phase examines the completed standard eight-class YOLOv8s checkpoint. It asks which errors dominate, which classes have measurable room for improvement, whether SAHI recovers objects or improves localization, what this suggests for future adaptive refinement, and whether any gain justifies its inference cost. It does not fit a model, evaluate held-out test images, approve a class-aware sampling policy, or implement the final adaptive method. The two-training by three-inference thesis design remains unchanged. This is preliminary engineering evidence, not final held-out Experiment B or a clinically validated diagnostic system.

## 2. Baseline checkpoint verification

The selected checkpoint is completed epoch 22 of the student's 80-epoch standard run. Its SHA-256 is `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`. A fresh read-only check validates all 12 checkpoint/sidecar pairs, finite saved state, epoch and frozen identity, and reconciles the complete 36-file baseline inventory with Phase 3A. Training completion remains 47,433 attempts, 47,416 successful updates and 17 AMP skips. The original run directory is never used as an inference output destination.

The inference code lives separately in `src/wrist_eval/`. The official trainer, its covered source files, protocol lock, checkpoint weights and class-aware A/B/C configurations are preserved. The eight IDs remain fracture, metal, text, periostealreaction, pronatorsign, bonelesion, softtissue and boneanomaly. These include findings and artifacts; they are not eight fracture subclasses.

## 3. Validation inference reproduction

The reference uses the frozen 3,088 validation images from 914 patients, containing 7,204 target objects. It loads the saved **EMA**, retains FP32 weights and unfused convolution/batch-normalization layers, and uses CUDA float16 autocast with half-precision normalized inputs, matching trainer-integrated validation. Image size is 768, batch 12, rectangular grouping, stride 32 and native validation padding 0.5. Test-time augmentation and dataset augmentation are disabled. Native multi-label, class-aware NMS uses confidence 0.001, IoU 0.7 and max_det 300.

The native reference reproduces mAP50 **0.5990822325**, mAP50-95 **0.3326975479**, fracture AP50 **0.9211619003** and fracture AP50-95 **0.5197308875**. Differences from the five-decimal training CSV are +0.0000022325 and -0.0000024521, below the predefined 0.0000051 tolerance (half the last printed decimal plus numerical allowance). All eight per-class values reproduce the saved history. Native fracture AP75 is 0.5345093187.

The reference uses zero loader workers so its read-only dataset subclass remains in-process on Windows, rather than the original four validation workers. Batch membership, order, pixels and transforms are retained; worker count changes loading cost. No trainer, backward pass or optimizer is invoked. The native label cache is read; a missing/stale cache raises instead of rewriting the prepared dataset. The effective full validator arguments are retained privately.

One initial pass stopped on an exported zero-height prediction. A bounded diagnostic identified a confidence-0.00144 box wholly inside letterbox padding. Native network-space AP retains such predictions; valid original-image exports explicitly count and omit boxes that collapse when clipped to the image. No confidence threshold, checkpoint or metric definition was tuned to resolve this. The failed log and diagnostic remain local. The corrected complete native pass is preserved separately.

For fair SAHI latency comparisons, a serial full-image bridge uses the identical numerical adapter as the tiled variants. Its full-image padding shape comes from the native reference's dimension-based rectangular batches; tiles use the same resize/half-stride-padding convention. Predictions are restored to original coordinates and evaluated against the original frozen annotations. Any bridge difference from native network-space metrics is reported explicitly below; this comparison does not substitute new values into the original training record.

The [saved-prediction bridge control](evaluation_bridge.json) resolves the metric difference without another GPU pass. Native network-space mAP50-95 is 0.33269755. Evaluating **the same predictions** after original-image clipping gives 0.33387575. Using original source boxes versus the loader's round-trip boxes gives identical AP; their maximum coordinate difference is only 0.00024414 pixels. Serial batch-one inference gives 0.33386896, just -0.00000679 from that control. Image-boundary clipping accounts for the material difference, principally text AP; it is not a newly improved checkpoint. Exactly two padding-only predictions are omitted from original-image exports, and 2,655 retained predictions touch an image boundary. All three strategies use the same clipped-original-coordinate evaluation below. Original native training/selection metrics remain preserved.

## 4. Per-class error decomposition

Official AP uses the installed, pinned Ultralytics `BaseValidator.match_predictions` and `ap_per_class`: confidence ranking, per-class one-to-one matching at ten float32 IoU thresholds 0.50–0.95, precision envelope and 101-point trapezoidal integration. The same implementation evaluates every inference variant. AP averages all confidence thresholds; it is distinct from recall at a chosen confidence.

Descriptive counts use confidence 0.05, 0.25 and 0.50, each at IoU 0.50 and 0.75. Predictions, in descending confidence order, claim the highest-IoU available compatible target, with stable source-order ties. Primary report counts use confidence 0.25 / IoU 0.50. Each prediction and target can be matched once. TP+FN equals target count; TP+FP equals retained operating-point prediction count.

For unmatched targets, residual stages consume remaining targets/predictions once, in this order: wrong class at the required IoU; same class below operating confidence but at least 0.001 and sufficient IoU; same class at operating confidence with IoU at least 0.1; then no residual candidate. These mutually exclusive descriptive labels depend on the declared ordering. They do not prove that an error has a single biological or network cause. A separate class-agnostic one-to-one confusion matrix reports spatial matches regardless of class, including background/misses. Unmatched predictions are classified as duplicate, wrong class, localization or background/unannotated by their overlap with annotations; these FP labels do not establish that annotations exhaust all visible pathology.

The baseline has **6,467 TP, 884 FP and 737 FN** at confidence 0.25 / IoU 0.50. Of the misses, 573 (77.7%) fall into the low-confidence residual category, 51 into localization, eight into classification and 105 have no residual candidate. This favors investigating recognition/ranking and false-positive control together with localization. Classification is a small fraction of this declared residual decomposition, although bone anomaly has five classification-category misses among its 32 FN. The class-agnostic matrix is a separate spatial view and need not give identical classification counts.

Fracture, periosteal reaction and pronator sign have substantial low-confidence candidates and decreasing AP at stricter IoU. Soft tissue is primarily a coverage/score problem at this operating point (17/72 TP, 47 low-confidence misses); its 17 matched boxes alone do not represent the 55 misses. Bone anomaly has four TP from 36 objects, with low-confidence, absent-candidate and classification failures; bone lesion has only one TP from five. A localization-only intervention cannot be presumed to solve those rare-class deficits.

| Class | GT | Patients | TP | FP | FN | Low conf | Loc | Class | No residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 2808 | 538 | 2463 | 346 | 345 | 269 | 19 | 1 | 56 |
| metal | 96 | 24 | 86 | 25 | 10 | 4 | 4 | 1 | 1 |
| text | 3611 | 913 | 3552 | 171 | 59 | 34 | 19 | 0 | 6 |
| periostealreaction | 493 | 163 | 298 | 295 | 195 | 172 | 8 | 0 | 15 |
| pronatorsign | 83 | 83 | 46 | 35 | 37 | 33 | 1 | 1 | 2 |
| bonelesion | 5 | 4 | 1 | 0 | 4 | 2 | 0 | 0 | 2 |
| softtissue | 72 | 61 | 17 | 9 | 55 | 47 | 0 | 0 | 8 |
| boneanomaly | 36 | 14 | 4 | 3 | 32 | 12 | 0 | 5 | 15 |

## 5. Fracture localization diagnostics

AP50 versus AP75, the full ten-threshold AP curve, matched-box IoU distributions, operating recall and size-specific misses are reported together. A decline in AP50-95 is not attributed entirely to box accuracy: confidence ordering, classification and detection coverage also contribute. Matched-IoU comparisons between methods use only objects detected by both methods at the specified operating point, with that survivor restriction stated.

Baseline fracture recall falls from **2,463/2,808 (87.71%) at IoU 0.50** to **1,736/2,808 (61.82%) at IoU 0.75**, at the same confidence 0.25. Its matched-IoU median is 0.8073 (5th/95th percentiles 0.6061/0.9276). Thus stricter localization is a real limitation alongside missed/low-confidence objects. Context/detail SAHI raise operating recall at IoU 0.75 to 63.996%/62.999%, but fracture AP75 falls: the additional precision/ranking penalty matters. At IoU 0.75, context recovers 150 fractures and loses 89; detail recovers 79 and loses 46. At IoU 0.50, mean IoU changes among fractures detected by both are -0.00460/-0.00647, with median zero. These methods do not establish a broad localization improvement.

## 6. False negatives and false positives

Recovery is evaluated per original target: both methods detect it, SAHI only, standard only, or neither. Those four counts sum to support. A duplicate cannot recover an additional target. The report also compares FP counts and FP/image, rather than interpreting more output boxes as improved recall. Low-confidence opportunities are descriptive evidence from already available outputs, not a chosen new operating threshold.

“No residual candidate” is not synonymous with complete absence of predictions. A stricter presence diagnostic finds **43 of the 737 baseline misses** have no prediction of any class overlapping at IoU >=0.1 even at confidence 0.001; 25 are fractures. At confidence 0.05, those counts become 195 misses and 130 fractures. Twenty-nine fracture misses lack even a same-class IoU-0.1 candidate at 0.001. These presence checks may share a nearby candidate across targets and are explicitly **not detection/recovery counts**. Pure prediction-guided ROI selection cannot assume every missed finding has an observable first-pass anchor.

Baseline FP decomposition is 388 background/unannotated, 322 localization, 163 duplicates and 11 classification. Context SAHI increases those to 926, 1,233, 1,004 and 46; detail to 1,886, 2,740, 1,437 and 265. Per-view/union NMS at 0.7 permits multiple boxes that each overlap one GT above the descriptive 0.5 threshold. Fragmented or imperfectly aligned predictions can therefore survive suppression and become duplicate/localization FP. This is consistent with the counts, but it is not a clinical adjudication of each extra box.

| Variant | Confidence | TP | FP | FN | FP/image | Fracture precision | Fracture recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sahi_context | 0.05 | 6875 | 11039 | 329 | 3.575 | 39.2% | 94.1% |
| sahi_context | 0.25 | 6599 | 3209 | 605 | 1.039 | 61.2% | 90.5% |
| sahi_context | 0.5 | 6123 | 1355 | 1081 | 0.439 | 74.8% | 84.7% |
| sahi_detail | 0.05 | 6849 | 17501 | 355 | 5.667 | 23.2% | 93.8% |
| sahi_detail | 0.25 | 6556 | 6328 | 648 | 2.049 | 45.3% | 89.6% |
| sahi_detail | 0.5 | 6055 | 3263 | 1149 | 1.057 | 65.0% | 83.1% |
| standard_serial | 0.05 | 6777 | 4351 | 427 | 1.409 | 71.0% | 92.0% |
| standard_serial | 0.25 | 6467 | 884 | 737 | 0.286 | 87.7% | 87.7% |
| standard_serial | 0.5 | 5942 | 249 | 1262 | 0.081 | 94.0% | 81.3% |

## 7. Object size, geometry and patient support

Predefined short-side strata at ideal aspect-preserving 768 input are <16, 16–<32, 32–<64, 64–<128 and >=128 pixels. These span the training-only periosteal median 27.75 and fracture median 59.26 pixels. Additional declared strata cover relative area (<0.005, 0.005–<0.02, >=0.02), relative border distance (<0.02, 0.02–<0.1, >=0.1), image aspect ratio (<0.4, 0.4–<0.6, 0.6–<0.8, >=0.8), original long side (<1024, 1024–1535, >=1536), object count (1, 2, 3, >=4), repeated instances of a class and non-text annotation overlap above IoU 0.1. Every stratum reports object and independent patient support, TP/FN, recall and matched IoU.

These are transparent descriptive bins, not clinically established cutoffs. Effective size uses 768/max(original dimensions); integer resize rounding introduces small differences in actual model pixels. Non-text box overlap is a geometric proxy, not expert-confirmed anatomical difficulty. Size-stratified recall is not presented as AP with ambiguous stratum-specific FP assignment.

Uncertainty uses 500 paired patient-bootstrap resamples, seed 314159. Each draw samples 914 patients with replacement, bringing all their images/examinations together; the same sampled patients are used for every method. The evaluator is unchanged. Percentile intervals are conditional on class presence; a resample missing a class has undefined AP for that class, and eight-class mAP is omitted for that replicate. Effective replicate counts are reported. These intervals do not correct validation selection or multiple comparisons. Five bone-lesion objects from four patients cannot support a reliable claim of improved generalization.

The largest relative small-box recall gains include periosteal objects below 16 pixels: **12/61 standard, 18/61 context, 21/61 detail**. Fractures at 16–<32 pixels improve from **17/28 to 20/28 and 21/28**; this is only three/four additional net detections. Most absolute fracture gains occur in the much larger 32–<64 group (1,633 objects). No validation fractures are below 16 pixels, so that stratum has undefined recall. Better recall in these bins does not offset the observed AP/FP regressions.

Geometry relationships are descriptive and confounded. Fracture recall is 83.1% in narrow images (W/H<0.4; n=201), compared with 88.0%/88.7% in the two middle aspect bins. Multiple fracture instances have 86.1% recall (n=1,424), versus 89.4% for single-instance images (n=1,384). Non-text overlap modestly lowers fracture recall (86.3% vs 88.0%), but periosteal recall is **higher** in overlapping cases (63.4% vs 54.3%); overlap is not a universal failure explanation. Border-near fracture strata contain only 22 objects total, including three at relative distance <0.02, so their low recall is not a reliable standalone border effect. Original-dimension, area, density and all other class-specific strata with patient counts remain in `error_analysis.json`. Empty strata have zero objects and undefined recall.

| Class | Short side at 768 | Objects | Patients | Standard recall | Context | Detail |
| --- | --- | --- | --- | --- | --- | --- |
| fracture | <16 | 0 | 0 | undefined | undefined | undefined |
| fracture | 16-<32 | 28 | 25 | 60.7% | 71.4% | 75.0% |
| fracture | 32-<64 | 1633 | 488 | 85.4% | 88.7% | 87.8% |
| fracture | 64-<128 | 1133 | 350 | 91.7% | 93.5% | 92.5% |
| fracture | >=128 | 14 | 12 | 92.9% | 92.9% | 92.9% |
| periostealreaction | <16 | 61 | 45 | 19.7% | 29.5% | 34.4% |
| periostealreaction | 16-<32 | 212 | 110 | 56.1% | 65.1% | 59.9% |
| periostealreaction | 32-<64 | 197 | 106 | 75.6% | 82.2% | 76.6% |
| periostealreaction | 64-<128 | 23 | 18 | 78.3% | 78.3% | 78.3% |
| periostealreaction | >=128 | 0 | 0 | undefined | undefined | undefined |

## 8. Representative qualitative inspection

Private overlays show the prepared 8-bit radiograph at its original spatial dimensions with original-coordinate ground truth, predictions, confidence, matching index and matched IoU. Declared categories cover correct/missed/poorly localized/false-positive fractures, small periosteal misses, pronator, lesion, soft-tissue and anomaly misses, multiple annotations, and recovered/lost examples for each tiled method. The first eligible image in a fixed SHA-256 order is selected for each category, with eligible counts and duplicate category selections retained. No manual substitution seeks a dramatic outcome. Images, filenames, patient IDs and detailed predictions remain ignored. The public receipt contains only anonymous case numbers and selection criteria. Technical overlay review is not clinical adjudication.

All 14 three-panel comparisons (12 unique source images, 42 method panels) were inspected for technical pixel/box alignment and correspondence to the declared errors. Case 5 illustrates a small periosteal target recovered by context slicing with additional nearby FP. Cases 6 and 9 show extra partial metal boxes despite retention of the correct global hardware box. Pronator, lesion and soft-tissue targets in cases 7–9 remain missed; case 10 retains the missed anomaly and adds a detail-view fracture-class FP near it. Cases 11 and 13 are the same selected image: their recovery is a text target, while the annotated fracture remains missed. Case 12 shows a marginal global fracture match lost after context union suppression, and case 14 loses a periosteal match with detail slicing. These examples support the aggregate recovery/fragmentation findings without estimating their prevalence. Crowded or border labels can overlap or be clipped in the panels; the private prediction records retain the exact values. See [the anonymous review receipt](qualitative_selection.json).

## 9. SAHI implementation and compatibility

[SAHI 0.12.8](https://pypi.org/project/sahi/0.12.8/) is added with seven pinned packages; the frozen Ultralytics 8.4.177 and PyTorch 2.14.1+cu126 environment is preserved. The [official integration](https://obss.github.io/sahi/models/ultralytics/) supports Ultralytics models. Here its model interface and conversion routines are retained, with a small `perform_batch_inference` adapter to preserve this project's unfused FP32 EMA/autocast and multi-label validation-NMS behavior. Generic `YOLO.predict` defaults would otherwise change that numerical policy. There is no second network or fine-tuning. Installed source hashes and dependency changes are recorded.

SAHI's [slicing utilities](https://obss.github.io/sahi/slicing/) generate overlapping image slices and shift local outputs to original coordinates. The prepared grayscale channels are replicated, without contrast enhancement. Published [SAHI methodology](https://arxiv.org/abs/2202.06934) motivates examining detail at multiple local views; its results on other domains do not predict this study's gains.

The verified dependency receipt is [compatibility_verified.json](compatibility_verified.json). The initial `compatibility.json` incorrectly classified direct-URL CUDA-wheel entries and pip's omitted freeze entry as additions; that metadata parsing error is corrected while preserving the initial receipt. The reviewed installation plan and frozen runtime checks confirm **seven actual additions and zero existing-package version changes**. [Runtime details](runtime_details.json) identify SAHI's auto-resolved **torchvision/CUDA** NMS backend, additional source hashes, native NMS defaults and the unchanged preparation-journal hash. CUDA NMS returns kept indices to the CPU, synchronizing that merge path; reported merge times include this bookkeeping.

## 10. Predefined SAHI configurations

The configuration was committed before any SAHI pilot AP was calculated. It is a bounded comparison, not a search. Training-only aspect ratio (median width/height 0.558) and object size motivate preserving each image's aspect ratio: each tile is ceil(f x original height) by ceil(f x original width), clipped to the source extent. Context uses f=0.667; detail uses f=0.5. Both have 20% overlap on each axis. They yield four and nine tiles respectively, plus one full-image pass, at sequential batch one. This is relative geometry, not a fixed square tile.

The 20% setting is the requested step overlap, not the overlap of every final neighboring tile. SAHI anchors the final tile at the image edge: the context grid consequently overlaps by approximately half a tile along each axis; the half-dimension grid has substantial additional overlap near its final row/column. This creates redundant processing and must be considered when interpreting the 5/10-pass cost. The negative or positive results apply to these predefined grids and postprocessing policy, not every possible SAHI configuration. No outcome-driven grid adjustment is made here.

Both use SAHI per-class NMS/IOU 0.7, explicitly forced, with no merge buffer, class remapping, NMM or test-time augmentation. The [API](https://obss.github.io/sahi/api/) documents the low-confidence automatic NMS switch and optional buffering; explicit settings prevent silent changes. Each tile first uses native NMS at confidence 0.001 / IoU 0.7 / max_det 300; after union NMS, descending-score outputs receive a final image-level cap of 300 because SAHI lacks that cap parameter. The ordinary comparator has one native NMS stage. The extra suppression stage and global prediction inclusion are part of the tiled strategy and are disclosed.

Measured grid geometry across validation confirms context adjacent-axis overlap of 50.07–50.73%, and detail overlap of 19.46–81.17%, due to integer steps and edge anchoring. Summed source tile area averages 1.7833x/2.2519x the original, before adding the full-image pass. These are source-pixel coverage sums, distinct from the five/ten network passes at 768 input.

## 11. Technical pilot

A predefined 24-image subset covers geometry extremes, first hash-ranked examples of every class, the smallest effective box for each class, dense annotations and then additional patients. It uses metadata/annotation characteristics for technical coverage, never model outcomes or ground-truth-guided inference crops. All three variants use exactly this subset. Synthetic tests verify complete slicing coverage, slice coordinate shifts and class-preserving duplicate suppression. The real pilot checks valid finite outputs, class IDs, original-image bounds, exact 1/5/10 model-pass accounting and nonnegative synchronized timings. Peak allocated memory is approximately 0.078 GiB for these batch-one runs; this is PyTorch tensor memory, not total GPU device occupancy.

Mean pilot end-to-end times extrapolate to about 219, 778 and 1308 seconds for all 3,088 images. These are estimates, not measured full-run durations. All variants pass the predefined three-hour-per-variant and 4 GiB allocated-memory gates, so the full validation set is retained. No convenient subset replaces the planned comparison. Pilot metrics remain preserved as technical subset evidence and do not select a configuration.

## 12. Full validation comparison

| Variant | mAP50 | mAP75 | mAP50-95 | Fracture AP50 | AP75 | AP50-95 | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sahi_context | 0.52577 | 0.26977 | 0.29315 | 0.86715 | 0.51120 | 0.49160 | 3209 | 605 |
| sahi_detail | 0.48217 | 0.23383 | 0.26256 | 0.82826 | 0.48395 | 0.46865 | 6328 | 648 |
| standard_serial | 0.59901 | 0.30228 | 0.33387 | 0.92117 | 0.53460 | 0.51974 | 884 | 737 |


Absolute AP differences in percentage points, relative to the common serial baseline:

| Variant | mAP50 | mAP75 | mAP50-95 | Fracture AP50 | AP75 | AP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| sahi_context | -7.32 | -3.25 | -4.07 | -5.40 | -2.34 | -2.81 |
| sahi_detail | -11.68 | -6.84 | -7.13 | -9.29 | -5.06 | -5.11 |

## 13. Recovered and lost detections

**sahi_context, confidence 0.25 / IoU 0.50.**

| Class | Both | Recovered | Lost | Neither | Mean IoU delta on both | Both n |
| --- | --- | --- | --- | --- | --- | --- |
| fracture | 2459 | 81 | 4 | 264 | -0.00460 | 2459 |
| metal | 86 | 3 | 0 | 7 | -0.03521 | 86 |
| text | 3546 | 18 | 6 | 41 | 0.00289 | 3546 |
| periostealreaction | 292 | 44 | 6 | 151 | -0.01848 | 292 |
| pronatorsign | 46 | 2 | 0 | 35 | -0.00093 | 46 |
| bonelesion | 1 | 0 | 0 | 4 | 0.00000 | 1 |
| softtissue | 17 | 0 | 0 | 55 | -0.00736 | 17 |
| boneanomaly | 4 | 0 | 0 | 32 | 0.00000 | 4 |

**sahi_detail, confidence 0.25 / IoU 0.50.**

| Class | Both | Recovered | Lost | Neither | Mean IoU delta on both | Both n |
| --- | --- | --- | --- | --- | --- | --- |
| fracture | 2461 | 55 | 2 | 290 | -0.00647 | 2461 |
| metal | 86 | 3 | 0 | 7 | -0.02129 | 86 |
| text | 3542 | 23 | 10 | 36 | -0.00923 | 3542 |
| periostealreaction | 297 | 20 | 1 | 175 | -0.00447 | 297 |
| pronatorsign | 46 | 0 | 0 | 37 | 0.00000 | 46 |
| bonelesion | 1 | 0 | 0 | 4 | 0.00000 | 1 |
| softtissue | 17 | 1 | 0 | 54 | -0.00926 | 17 |
| boneanomaly | 4 | 0 | 0 | 32 | 0.00000 | 4 |


| Variant | Class | Standard FP | SAHI FP | Shared | Introduced | Removed |
| --- | --- | --- | --- | --- | --- | --- |
| sahi_context | fracture | 346 | 1613 | 335 | 1278 | 11 |
| sahi_context | metal | 25 | 213 | 24 | 189 | 1 |
| sahi_context | text | 171 | 516 | 149 | 367 | 22 |
| sahi_context | periostealreaction | 295 | 786 | 288 | 498 | 7 |
| sahi_context | pronatorsign | 35 | 53 | 35 | 18 | 0 |
| sahi_context | bonelesion | 0 | 0 | 0 | 0 | 0 |
| sahi_context | softtissue | 9 | 17 | 9 | 8 | 0 |
| sahi_context | boneanomaly | 3 | 11 | 3 | 8 | 0 |
| sahi_detail | fracture | 346 | 3042 | 346 | 2696 | 0 |
| sahi_detail | metal | 25 | 729 | 25 | 704 | 0 |
| sahi_detail | text | 171 | 1967 | 149 | 1818 | 22 |
| sahi_detail | periostealreaction | 295 | 533 | 290 | 243 | 5 |
| sahi_detail | pronatorsign | 35 | 36 | 35 | 1 | 0 |
| sahi_detail | bonelesion | 0 | 0 | 0 | 0 | 0 |
| sahi_detail | softtissue | 9 | 12 | 9 | 3 | 0 |
| sahi_detail | boneanomaly | 3 | 9 | 3 | 6 | 0 |

FP correspondence uses a separate same-class one-to-one IoU-0.50 match between the methods' unmatched operating-point predictions. Shifted boxes can count as introduced/removed; these are spatial correspondences, not clinical lesion identities.

At confidence 0.25 / IoU 0.50, context SAHI has **148 recovered / 16 lost** targets (net +132); detail has **102 recovered / 13 lost** (net +89). Fracture accounts for 81/4 and 55/2; periosteal reaction for 44/6 and 20/1. Soft tissue gains zero/one additional target, and anomaly/lesion gain none. Spatial FP correspondence identifies 2,366 introduced and 41 removed FP for context, and 5,471 introduced and 27 removed for detail; their differences equal the total FP increases. This is a recall/precision trade-off, not a failure to retain most global TP. Fracture precision drops from 87.68% to 61.16%/45.27%, while recall rises from 87.71% to 90.46%/89.60%.

## 14. Per-class changes

There is no dependable class-level AP improvement. Metal AP50-95 falls particularly sharply (0.57460 to 0.39221/0.19861), while fracture falls from 0.51974 to 0.49160/0.46865. Periosteal reaction remains below standard despite recovering small targets. Context's bone-lesion AP50-95 difference is only +0.00003178, without an additional operating-point detection; that is not reliable evidence of benefit. The five-object/four-patient support makes any small lesion AP movement especially unstable.

**ap50; differences in percentage points.**

| Class | Standard | Context | Delta pp | Detail | Delta pp |
| --- | --- | --- | --- | --- | --- |
| fracture | 0.92117 | 0.86715 | -5.40 | 0.82826 | -9.29 |
| metal | 0.89041 | 0.63630 | -25.41 | 0.30790 | -58.25 |
| text | 0.98874 | 0.97215 | -1.66 | 0.90920 | -7.95 |
| periostealreaction | 0.56600 | 0.44720 | -11.88 | 0.51282 | -5.32 |
| pronatorsign | 0.57107 | 0.53858 | -3.25 | 0.56235 | -0.87 |
| bonelesion | 0.41500 | 0.40289 | -1.21 | 0.39855 | -1.64 |
| softtissue | 0.29604 | 0.25774 | -3.83 | 0.26987 | -2.62 |
| boneanomaly | 0.14361 | 0.08413 | -5.95 | 0.06845 | -7.52 |

**ap75; differences in percentage points.**

| Class | Standard | Context | Delta pp | Detail | Delta pp |
| --- | --- | --- | --- | --- | --- |
| fracture | 0.53460 | 0.51120 | -2.34 | 0.48395 | -5.06 |
| metal | 0.55916 | 0.38342 | -17.57 | 0.18978 | -36.94 |
| text | 0.82387 | 0.81427 | -0.96 | 0.72826 | -9.56 |
| periostealreaction | 0.09958 | 0.06736 | -3.22 | 0.08119 | -1.84 |
| pronatorsign | 0.13085 | 0.11907 | -1.18 | 0.12782 | -0.30 |
| bonelesion | 0.19500 | 0.19877 | +0.38 | 0.19500 | +0.00 |
| softtissue | 0.05690 | 0.05442 | -0.25 | 0.05427 | -0.26 |
| boneanomaly | 0.01827 | 0.00967 | -0.86 | 0.01039 | -0.79 |

**ap50_95; differences in percentage points.**

| Class | Standard | Context | Delta pp | Detail | Delta pp |
| --- | --- | --- | --- | --- | --- |
| fracture | 0.51974 | 0.49160 | -2.81 | 0.46865 | -5.11 |
| metal | 0.57460 | 0.39221 | -18.24 | 0.19861 | -37.60 |
| text | 0.69775 | 0.69161 | -0.61 | 0.62843 | -6.93 |
| periostealreaction | 0.21166 | 0.16069 | -5.10 | 0.18957 | -2.21 |
| pronatorsign | 0.23866 | 0.21921 | -1.95 | 0.23391 | -0.48 |
| bonelesion | 0.24050 | 0.24053 | +0.00 | 0.23721 | -0.33 |
| softtissue | 0.12450 | 0.11205 | -1.25 | 0.11263 | -1.19 |
| boneanomaly | 0.06355 | 0.03728 | -2.63 | 0.03146 | -3.21 |


## 15. Latency and GPU memory

Hardware is the same RTX 4050 Laptop GPU (6,141 MiB), with CUDA 12.6 / driver 560.94 on Windows. Python is 3.12.5, torch 2.14.1+cu126, torchvision 0.29.1+cu126 and Ultralytics 8.4.177. Core versions agree with the [frozen environment](../gpu_feasibility/environment.json) and the current compatibility receipt. Chrome/Codex graphics processes remained present; no concurrent model workload was observed. WDDM per-process memory was unavailable, so these are desktop measurements with the limitations in [runtime_details.json](runtime_details.json).

Steady-state per-image timing starts before image decoding and stops after final merging/capping. It excludes subsequent annotation matching, AP computation, JSON export and model initialization; a separate loop wall time includes matching/export overhead. CUDA is synchronized at boundaries; events measure the model forward separately, excluding preprocessing, NMS and merging. The adapter records preprocessing, forward wall time, NMS/restoration and SAHI's slice/merge costs. Their boundaries are reported explicitly rather than treating SAHI's unsynchronized generic duration as kernel-only time. Mean/median/p95/p99, throughput, model passes, pre/post-merge counts and peak allocated/reserved memory accompany accuracy.

Each variant uses the same GPU and a fresh process, with a three-call fixed synthetic warmup. Cold loading, warmup and the first measured image remain separately inspectable. Variable image shapes can still incur first-shape overhead. Peak allocation includes resident model tensors; reserved memory is the caching allocator high-water mark and can persist across images. Neither equals total-device memory, which also includes the display/driver. Runs are sequential without another model workload. These are instrumented batch-one costs, not maximally optimized serving throughput or the batch-12 native validation throughput.

| Variant | Mean ms | Median | p95 | p99 | Ratio | GPU forward ms | Passes | Merge ms | Alloc GiB | Reserve GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sahi_context | 238.1 | 237.7 | 300.1 | 338.2 | 3.82x | 160.6 | 5 | 7.42 | 0.077 | 0.111 |
| sahi_detail | 419.2 | 433.4 | 519.6 | 559.1 | 6.73x | 300.1 | 10 | 8.29 | 0.077 | 0.111 |
| standard_serial | 62.3 | 61.8 | 77.5 | 85.6 | 1.00x | 33.7 | 1 | 0.37 | 0.077 | 0.111 |

| Variant | Mean pre-union boxes | Mean final boxes | Images/s | Load s | Warmup s | Sum inference s | Loop wall s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sahi_context | 58.9 | 34.6 | 4.20 | 2.98 | 0.72 | 735.3 | 740.0 |
| sahi_detail | 80.3 | 56.2 | 2.39 | 2.47 | 0.81 | 1294.6 | 1299.6 |
| standard_serial | 13.9 | 13.9 | 16.04 | 3.00 | 0.77 | 192.5 | 196.6 |

Pre-union counts follow native per-view NMS and padding-only removal; they are not raw grid-anchor counts. Final boxes include all scores >=0.001. Load time includes EMA loading and adapter setup, excluding process/Python import startup.

Across all 3,088 images, standard/context/detail execute **3,088 / 15,440 / 30,880** model passes. Peak tensor allocation stays below 0.08 GiB; allocator reservation peaks at 0.111328 GiB (114 MiB). This small inference allocation does not describe training VRAM. Maximum final outputs are 98/211/270, so the final 300-box cap never binds. The serial comparator's reported 0.37 ms merge field is final CPU sorting/capping bookkeeping; it has no additional SAHI union-NMS stage.

## 16. Implications for adaptive local refinement

The following are candidate concepts, not approved algorithms or measured adaptive results. No final adaptive inference pipeline is implemented in this phase.

| Concept | Observable inference-time inputs | Illustrative budget | Main limitation |
|---|---|---|---|
| Confidence/size-guided refinement | Small predicted boxes, an intermediate score band, local overlap/disagreement | Global + one or two local views: 2–3 passes | Cannot directly propose a region for a finding with no useful first-pass prediction; false positives also have low confidence |
| Class-aware context with protected global detections | Predicted category/size, neighboring boxes, confidence and global/local agreement | Global + one or two broad-context views: 2–3 passes | Scores are not calibrated localization-quality estimates; local fragments can overrule a correct global box unless fusion protects it |
| Budgeted hybrid with limited fallback | Prediction cues plus predeclared image-only structural or coarse-coverage proposals | Global + up to three local/fallback views: <=4 passes | Coverage remains incomplete, and fallback views can add false positives or lose anatomical context |

These pass counts are design illustrations, not adaptive latency measurements. Actual preprocessing, crop geometry and fusion can change cost; measured SAHI costs provide context rather than a guaranteed linear speedup. Ground truth is available only for offline coverage/error diagnosis. Future ROI selection must consume model predictions or image pixels alone. A claim that a target is covered by an oracle-centered window is not evidence that an inference-time proposal mechanism can find it.

The strongest candidate for a **new, bounded development step** combines class-aware broad context and protected reliable global detections with a small ROI budget, then tests whether limited fallback coverage adds enough value to justify its FP/cost risk. Uniform detail is not supported as a default accuracy improvement by these results. Fusion/ranking needs explicit attention: more correct operating-point detections coexist with much lower AP, and mean matched fracture IoU does not improve.

Of context's 81 recovered fractures, 77 were baseline low-confidence residual misses; 68 have a same-class IoU-0.1 first-pass candidate at confidence >=0.05, and 54 have a nearby same-class prediction smaller than 64 effective pixels. For periosteal reaction, all 44 recoveries were low-confidence misses, 42 have a >=0.05 same-class nearby candidate, and 41 have a small candidate. However, **13/81 recovered fractures have no prediction of any class overlapping at IoU 0.1 at confidence 0.05**; detail has 16/55 such recoveries. Presence at confidence 0.001 is much broader but also includes many weak FP candidates. These GT-conditioned proximity figures are diagnostic opportunity estimates, not measured ROI-proposal precision/coverage or a deployable selection rule. They do not prove that an inference-time system can select every useful region.

Global/local disagreement requires extra inference and cannot be treated as a free first-pass signal. A future selective method must measure proposal burden, true recovery, lost global detections, FP and matched IoU under a declared budget, with no GT-guided proposals. The SAHI crop fractions differ from Candidate A's broader training crops and sampling mix, so this study neither validates nor directly rejects that training intervention. No candidate inherits class-aware training weights from validation errors.

## 17. Limits and validation-selection bias

The baseline was selected by validation mAP50-95, and this same partition now guides development expectations. These metrics are not unbiased independent performance estimates. Two configurations and three declared confidence points bound the investigation; no result-based tile, merging or sampling search is performed. Annotation boundaries, overlapping categories, omitted foreign-body labels and the original quantized 8-bit preparation affect interpretation. Residual error labels and axis-aligned IoU are engineering diagnostics, not a radiologist's error adjudication. Rare independent patient counts, especially bone lesion and anomaly, remain material limits. Bootstrap intervals do not undo model/method selection.

The bootstrap conditions on this single fixed checkpoint and does not measure training-seed variability. Native auxiliary precision/recall values use the framework's best mean-F1 confidence, whereas the report's operating counts use the declared fixed thresholds. CUDA-event forward time is an elapsed stream interval, not a profiler sum of individual kernel durations. Sequential method order, variable image shapes, Windows display activity and laptop power/thermal behavior can affect timing; no optimized serving-throughput claim follows.

Class-aware candidates A/B/C remain exactly as proposed in Phase 3A. Their weights, crop thresholds and caps cannot be optimized against these validation findings. The eventual A–F thesis conditions still use two checkpoints and three inference methods; those labels are distinct from the class-aware candidate labels. Held-out test contents remain unopened for this phase. After both checkpoints and inference protocols are approved and frozen, the final comparison can use the reserved test set once. Formal supervisor acceptance of revised scope remains pending.

Paired 95% patient-bootstrap intervals for AP50-95 differences, in percentage points:

| Class / overall | Context minus standard | Detail minus standard |
| --- | --- | --- |
| overall | [-4.83, -3.11] (n=492) | [-8.13, -6.07] (n=492) |
| fracture | [-3.38, -2.27] (n=500) | [-5.71, -4.56] (n=500) |
| metal | [-23.91, -11.91] (n=500) | [-45.15, -30.28] (n=500) |
| text | [-1.08, -0.18] (n=500) | [-7.73, -6.04] (n=500) |
| periostealreaction | [-6.39, -4.01] (n=500) | [-3.34, -1.38] (n=500) |
| pronatorsign | [-3.19, -0.77] (n=500) | [-0.64, -0.20] (n=500) |
| bonelesion | [-0.14, +0.18] (n=492) | [-0.97, +0.00] (n=492) |
| softtissue | [-2.77, +0.43] (n=500) | [-3.00, +0.61] (n=500) |
| boneanomaly | [-4.61, -0.01] (n=500) | [-5.96, -0.03] (n=500) |

All 500 paired resamples completed. Eight omit bone lesion, leaving 492 defined eight-class mAP replicates; fracture is present in all 500. Both overall and fracture AP50-95 difference intervals remain below zero for both predefined variants. This is evidence of a regression within this development analysis, conditional on the fixed checkpoint and observed patients. The rare-class intervals and repeated validation use do not establish out-of-sample or clinical effects. Full intervals, including AP50/AP75, are in [patient_bootstrap.json](patient_bootstrap.json).

## 18. Verification, reproducibility and next phase

**Verification passed.** The full suite ran 110 tests in 7.817 seconds (91 prior tests plus 19 focused inference/analysis tests); `pip check` reports no broken requirements. Tests cover checkpoint/config identity, class mapping, deterministic selection, slicing coverage, coordinate restoration, merging, invalid outputs, AP matching, TP/FP/FN accounting, recovery, stratification and patient resampling. The independent [accounting receipt](accounting_verification.json) reconciles every variant's 3,088 images / 7,204 objects, finite outputs, exact model-pass counts and recovery partitions. A bounded [deterministic replay](determinism_replay.json) on the first three predefined pilot images per method has maximum array difference **zero** for all methods. This nine-image identity check is separate from performance measurement; no successful full experiment was repeated after interruption.

The [preservation receipt](preservation_checks.json) confirms all 36 baseline artifacts unchanged, all 6,176 validation image/label SHA-256 values correct, the 121,973-file raw/prepared stat inventory unchanged, frozen source/protocol validity and unchanged proposed class-aware candidates. Earlier preparation/training/design source, configurations and evidence also have no Git changes from the pre-phase revision `bd83049`. These checks supplement validation-only path guards; they do not constitute a new exhaustive cryptographic re-audit of every raw file. No held-out image/label contents were evaluated, model was trained or final adaptive pipeline was added.

The implementation used for final verification is commit **8098012a6f1ae5bdea22b7ffaf4336323e3db310**. [Implementation provenance](implementation_provenance.json) records normalized source hashes, private prediction-file hashes, best-checkpoint SHA-256 and canonical configuration SHA-256 `86ed8fd6450b77734e6cdea4eff5dec5eed59d5f0ea9790f0e37c507f9336c13`. The earlier meaningful commits are **46f3194** (validation-only implementation and predeclared configuration) and **205d49a** (SAHI feasibility and paired diagnostics). The following documentation/evidence commit contains this completed report; its identity is available in Git history and the delivery summary. All are pushed to `origin/main` without force. Private radiographs, prediction records and raw logs remain excluded.

Completed verification commands, run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m wrist_eval.bridge
.\.venv\Scripts\python.exe -m wrist_eval.analyze
.\.venv\Scripts\python.exe -m wrist_eval.bootstrap
.\.venv\Scripts\python.exe -m wrist_eval.qualitative
.\.venv\Scripts\python.exe -m wrist_eval.verify
.\.venv\Scripts\python.exe -m pip check
```

The [README reproduction section](../../../README.md#validation-error-analysis-and-sahi-feasibility) records the preceding reference, pilot, gate and full-run commands. They require the private frozen data, completed baseline and existing preservation receipts. Protected attempt directories deliberately refuse reuse. Inspect saved receipts before any deliberate rerun. Saved-prediction analysis, bootstrap and bridge do not invoke neural inference; `verify` includes the bounded replay. Technical visual review was performed after overlay generation, separately from automated checks. [Final verification](verification.json) and [class CSV](class_comparison.csv) provide compact evidence alongside [the aggregate analysis](error_analysis.json), the three `full_*.json` summaries and [size-stratum CSV](size_strata_comparison.csv).

**Recommendation for the next separately authorized phase:** retain SAHI as the established comparator, with these two predefined results preserved as negative development evidence. Do not adopt either current uniform tiling policy as an accuracy default or freeze a final inference protocol from this study. A bounded prototype of broad-context, prediction-guided refinement with explicit FP/ranking guards has a measurable rationale: context slicing recovers 81 fractures and 44 periosteal targets, mostly from low-confidence baseline candidates. Protecting global matches alone is insufficient because most survive already; controlling introduced false positives is central. Measure candidate burden, missed-region coverage, genuine recoveries/losses, AP and cost, with an explicit image-only fallback allowance and no GT-derived inference ROIs. No adaptive gains are claimed before that experiment exists.

Class-aware policy review remains a separate student decision. Candidate A's train-only exposure/diversity evidence supports review, but neither it nor B/C has measured trained performance. Rare-patient concentration, crop context/annotation retention, combined sampling/cropping effects and actual compute/update comparability remain unresolved. Validation errors must not determine new training weights or crop thresholds. Keep the common 80-epoch protocol, two-checkpoint by three-inference design and held-out test isolation. Stop here for student review; no class-aware training, candidate alteration, final adaptive implementation or held-out evaluation is started.
