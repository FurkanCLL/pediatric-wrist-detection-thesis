# Class-aware design: bounded exposure gains with broad fracture coverage

Phase 3A, Step 1. Verified 2026-10-16. **Recommendation: Candidate A, subject to student approval.** No production loader integration, official class-aware training, new YOLO inference, SAHI/adaptive evaluation or held-out test evaluation was performed. Candidate letters A–C here are design labels, distinct from the thesis's six inference conditions A–F.

Candidate A is the smallest tested intervention. It redistributes 2% of the fixed draw budget, preserves approximately 98% of distinct fracture-positive and negative images per epoch, and represents every training patient. Its mean post-crop object exposures rise about 7.7% for periosteal reaction, 14.0% for pronator sign, 22.5% for soft tissue and 28.1% for bone anomaly. Bone lesion is coverage-only. These are **sample-distribution measurements, not expected AP improvements**.

The crop component is deliberately limited: about 51 accepted crops per epoch, 0.36% of draws. Most accepted crop parents contain small periosteal-reaction boxes. This may be too weak to produce a separately measurable cropping benefit. Candidate B increases exposure further with less distinct-image coverage. Candidate C fails the proposed 95% distinct-image guardrails and increases patient concentration; it is not recommended.

## 1. The standard baseline is verified and preserved

The [baseline preservation report](BASELINE_PRESERVATION_REPORT.md) and [checkpoint receipt](baseline_preservation.json) reconcile all 36 existing artifacts and 12 checkpoints, including actual bytes, sidecar hashes, saved identity/epoch, FP32 states, training arguments, scheduler position, numerical accounting and measured per-class history. The selected `best.pt` is completed epoch **22**:

| Validation metric | Value at the selected checkpoint |
|---|---:|
| Overall mAP50 | 0.59908 |
| Overall mAP50-95 | 0.33270 |
| Fracture AP50 | 0.92116190 |
| Fracture AP50-95 | 0.51973089 |

Best checkpoint SHA-256: `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`. The 80-epoch run made **47,433 attempts, 47,416 successful optimizer updates and 17 skips**, without a resume. All 17 skips reconcile with AMP events; recorded scales ranged from 64 to 8,192. The best epoch was selected by overall validation mAP50-95, not fracture AP alone.

All checkpoints, logs, CSV, configuration and provenance remain in place. The frozen 80-epoch protocol and every covered baseline source hash remain valid. Exploratory code is isolated in `src/wrist_design/`; the official launcher, trainer, checkpoint code, prepared data and protocol lock were not edited. Metrics here describe an existing validation-selected baseline. They do not establish clinical reliability, held-out performance or a policy objective. The policy code receives no validation AP values.

## 2. Training support, geometry and co-occurrence

The [training analysis](training_analysis.json) is recomputed from the hash-verified frozen training records: 14,061 images, 4,264 patients, 32,871 boxes and 11 empty detector-label images. All quoted support figures match the supplied values.

| Class | Objects | Images | Patients | With fracture | Multiple categories (incl. text) |
|---|---|---|---|---|---|
| fracture | 12,422 | 9,307 | 2511 | 9,307 (100.0%) | 9296 |
| metal | 600 | 518 | 109 | 442 (85.3%) | 518 |
| text | 16,388 | 14,030 | 4264 | 9,287 (66.2%) | 9542 |
| periostealreaction | 2,497 | 1,590 | 763 | 1,587 (99.8%) | 1590 |
| pronatorsign | 400 | 399 | 389 | 375 (94.0%) | 399 |
| bonelesion | 35 | 32 | 16 | 18 (56.2%) | 32 |
| softtissue | 324 | 308 | 283 | 226 (73.4%) | 308 |
| boneanomaly | 205 | 147 | 67 | 52 (35.4%) | 147 |

Co-occurrence is substantial but unequal. Periosteal reaction co-occurs with fracture in 1,587/1,590 images; pronator sign in 375/399; soft tissue in 226/308; bone anomaly in 52/147. Metal and bone anomaly coexist in 39 images; pronator sign and soft tissue in 55. Thus an unstratified rare-class bonus can alter both fracture exposure and other classes' exposure. Text occurs in 14,030 images, so a generic “multiple categories” count mostly includes text rather than independent radiographic findings. The full eight-class co-occurrence matrix is retained in the analysis, with pre/post simulation matrices in [simulation_summary.json](simulation_summary.json).

Median short sides at idealized aspect-preserving 768 input are 59.26 px for fracture, 27.75 for periosteal reaction, 68.74 for pronator sign, 28.11 for bone lesion, 84.48 for soft tissue and 64.72 for bone anomaly. Metal's median is 113.15 px. These distributions motivate extra detail for small findings, not uniform cropping of every uncommon class. Image aspect ratio ranges from 0.249 to 1.679, median 0.558; median images per patient is 2, maximum 30. A fixed square crop or unbounded per-image replacement would ignore these structures.

There are 9,307 images with fracture boxes and 4,754 without. The corresponding patient sets contain 2,511 and 2,325 patients, with 572 in both sets. “Without fracture boxes” and “empty labels” do not mean clinically healthy. Source taxonomy, polygon-envelope boxes, omitted foreign-body objects, axes and image tags retain the original preparation semantics.

## 3. Why use a coverage-preserving hybrid

Keeping every image once and adding repetitions cannot fit the same 14,061-draw budget. Some common full-image views must be omitted when additional views are introduced. The design measures that loss rather than calling repeated exposure new coverage.

For comparison, the training-only analysis calculates exact one-epoch expectations for unrestricted sampling with replacement. These are mathematical reference cases, not additional proposed training runs:

| Reference with replacement | Expected unique images | Expected unique patients | Expected lesion image draws | Expected empty-label draws |
|---|---|---|---|---|
| uniform_replacement | 8,888.43 | 3,839.88 | 32.00 | 11.00 |
| sqrt_rarity_sum | 7,547.29 | 3,230.29 | 262.78 | 0.00 |
| inverse_frequency_sum | 4,766.44 | 2,350.93 | 1,775.14 | 0.00 |

The rarity references use image weights `sum_c I_c^(-alpha)` over present classes, with alpha 0.5 or 1. They give zero weight to empty-label images and can repeatedly select the same few rare patients. Uniform replacement also loses considerable unique coverage even though its expected class exposure is unchanged. Our candidates instead retain a large distinct-image component, explicit patient/status anchors, bounded extra tickets and empty-label coverage. Their weights are not claimed to be optimal.

## 4. Explicit candidate sampling policies

The [candidate configuration](../../../configs/class_aware_candidates.json) is proposed and unapproved. All three candidates follow the same procedure:

1. Protect one full-image view of **every image containing metal, periosteal reaction, pronator sign, bone lesion, soft tissue or bone anomaly**, and every empty-label image.
2. Within fracture-positive and fracture-negative strata, retain at least one full image for every represented patient. Fill the remaining coverage component without replacement.
3. Allocate the requested extra budget proportionally between the two strata. Final fracture-positive/negative draws remain exactly 9,307/4,754.
4. Extra draws may target only classes 3, 4, 6 and 7. Bone-lesion patients receive no extra draws. Use bounded weighted tickets, with maximum image draws and a shared patient extra allowance across both strata.
5. When eligible tickets cannot fill the extra quota under the caps, recover omitted original images once. Do not relax caps or force rare repetitions. Shuffle the completed epoch plan deterministically.

For target class c, define `a_c = min(weight_cap, sqrt(1590 / I_c) * min(1, P_c / 100))` if it has at least 32 independent training patients; otherwise zero. The reference 1,590 is training periosteal-reaction image support. Image priority is **the maximum** eligible `a_c`, not a sum. Fracture, text, metal and bone lesion have no direct bonus weight. The support floor separates bone lesion's 16 patients from the next-smallest target's 67; the reliability factor tempers bone anomaly. These thresholds are transparent engineering hypotheses, not statistically certified sample-size cutoffs.

The eligible bonus pool contains **2,356 images from 1,262 patients** after excluding the entire 16-patient bone-lesion cohort. Artifact labels and multiple co-occurring findings cannot add their weights into an extreme priority. Exponential random priorities produce a weighted permutation of bounded image tickets; patient caps then constrain acceptance. The resulting probabilities are therefore conditional, and exposure is measured by simulation rather than inferred directly from the weights.

| Parameter | A: conservative | B: balanced | C: stronger |
|---|---|---|---|
| Requested extra budget | 2% = 281 | 5% = 703 | 10% = 1,406 |
| Max image draws/epoch | 2 | 2 | 3 |
| Patient extra fraction / absolute cap | 0.25 / 2 | 0.50 / 3 | 1.00 / 6 |
| Class priority cap | 1.5 | 2.0 | 3.0 |
| Max crop attempts/epoch | 281 | 703 | 1406 |
| Min crop dimension fraction | 0.8 | 0.7 | 0.6 |
| Context multiple | 4 | 4 | 3 |
| Desired short side at 768 | 44.45 px | 53.34 px | 59.26 px |
| Priority: periosteal / pronator / soft tissue / anomaly | 1.000 / 1.500 / 1.500 / 1.500 | 1.000 / 1.996 / 2.000 / 2.000 | 1.000 / 1.996 / 2.272 / 2.204 |

For a patient with N_p original training images, extra allowance is `min(absolute_cap, ceil(fraction * N_p))`. This is **not a strict fractional increase bound**: a one-image patient can receive one extra draw and therefore reach 2x exposure. We use ceiling because most patients have only two images; floor-based fractions would exclude many patients from any extra opportunity. Actual concentration and patient-relative exposure are reported below. Every candidate still sees all 4,264 patients, including all 2,511 fracture-positive and 2,325 fracture-negative patient memberships, each epoch.

## 5. Cropping is an additional view with explicit fallbacks

Only an additional rare-class draw can be cropped. Its parent already has a full-image draw in the coverage component. Every selected parent therefore has a full view; no crop substitutes for that parent's only view. At most one crop attempt is made per patient per epoch. The configured fractions cap **attempts**, not accepted crops, and failures are not replaced by extra attempts to reach an accepted-crop quota.

Select an eligible small target by bounded class priority, then short-side size, with deterministic coordinate ties. The desired short side is a fraction of the training fracture median at 768: 0.75x, 0.90x or 1.00x for A/B/C. It is a trigger and zoom target, not a promise that the constrained crop achieves that size.

Let s be the target's original short side after idealized 768 scaling, and let w/W and h/H be its width/height relative to the source. The common crop-dimension fraction is:

`f = max(minimum_fraction, s / desired_short_side, context_multiple * max(w/W, h/H))`.

If f exceeds 0.95, use the full image. Otherwise crop approximately fW by fH, round dimensions outward and clamp the centered rectangle inside the original image. This preserves the source aspect to rounding precision without invented pixels. A/B/C minimum dimension fractions are 0.80/0.70/0.60, retaining at least approximately 64%/49%/36% of original image area. Context spans are 4x/4x/3x the target box. This is a geometric span, not proof of adequate anatomy; margins can be asymmetric at original image borders.

Annotation rules apply to **all eight detector classes**:

- Every source fracture box and the selected target must be fully retained. Otherwise reject the crop and keep the full image.
- A fully outside non-fracture object has no visible pixels and is omitted from crop labels, with its count recorded.
- A partially visible non-target box must retain at least 90% of its area. Keep its exact clipped intersection and source index. If any visible fragment falls below the threshold, reject the entire crop rather than deleting its label.
- Retained boxes need a short side of at least 2 px at idealized 768 crop scaling. Reject malformed, non-finite or degenerate source boxes; do not silently repair them.
- Bone-lesion-positive images are never crop targets; the sampler prevents crop draws from the entire bone-lesion patient cohort.

| Candidate | Accepted unique crop parents | Median zoom | Median target short side before → after | Actual attempts → crops/epoch | Crops / all draws |
|---|---|---|---|---|---|
| A | 643 | 1.249x | 19.98 → 24.60 px | 263.73 → 51.08 | 0.363% |
| B | 610 | 1.427x | 21.51 → 28.63 px | 542.02 → 103.01 | 0.733% |
| C | 861 | 1.562x | 24.58 → 36.90 px | 777.35 → 222.57 | 1.583% |

The unique-parent, zoom and target-size columns describe the eligible geometry pool; the attempt/crop columns are draw-weighted epoch means. Candidate A's accepted pool contains 611 periosteal-reaction targets, 26 bone-anomaly, four soft-tissue and two pronator-sign targets. Thus its cropping component mostly provides modest detail for thin periosteal boxes; sampling supplies most of the exposure increase for the other classes. Candidate B has fewer geometrically acceptable parents than A despite a higher allowance: tighter frames can fail fracture/context/annotation checks. Larger crop budgets are not automatically safer or more effective.

All prototype pixels come from prepared training PNGs. No raw or prepared file is rewritten. These simulations are **before the common random augmentation**. Installed Ultralytics applies affine transformations and box-candidate filtering afterwards; its default filtering includes a 2 px size threshold and an area-retention test. We have not replayed those transformations or integrated a custom dataset. Consequently these are nominal source/crop label exposures, not reconstructed final post-augmentation labels from the completed baseline. Integration will need explicit order, worker-RNG and annotation checks after approval.

## 6. Full 80-epoch simulation results

All four distributions were simulated for 80 epochs at seeds **42, 1042 and 2042**, specified before simulation. Official training seed stays 42. Each run contains 1,124,880 draws; the full analysis checks **960 epochs and 13,498,560 draws**. Sorted source identities and seed/epoch RNG streams make results independent of input-record order. The standard reference is the actual native sampler's nominal behavior: shuffled full coverage without replacement, with drop-last disabled. It is not a new baseline run.

The following values summarize seed 42 over all 80 epochs. Counts marked fixed hold each epoch; crop counts are means. Complete per-epoch metrics for all seeds are in [epoch_exposures.csv](epoch_exposures.csv); percentile/minimum/maximum and cumulative results are in [simulation_summary.json](simulation_summary.json).

| Policy | Requested / realized rare extras | Unique images (fixed) | Unique fracture-positive (fixed) | Unique no-fracture-box (fixed) | Patients (fixed) | Full / crop draws, mean |
|---|---|---|---|---|---|---|
| standard | 0 / 0 | 14,061 (100.00%) | 9,307 (100.00%) | 4,754 (100.00%) | 4264 | 14,061.00 / 0.00 |
| A | 281 / 281 | 13,780 (98.00%) | 9,121 (98.00%) | 4,659 (98.00%) | 4264 | 14,009.92 / 51.08 |
| B | 703 / 625 | 13,436 (95.56%) | 8,842 (95.00%) | 4,594 (96.63%) | 4264 | 13,957.99 / 103.01 |
| C | 1406 / 1248 | 12,813 (91.12%) | 8,376 (90.00%) | 4,437 (93.33%) | 4264 | 13,838.42 / 222.57 |

All candidates retain 100% of nominal fracture-positive and negative draws. That alone would hide C's substantial distinct-image loss. A preserves approximately 98% of both image strata; B retains 95.00% of distinct positive images and 96.63% of distinct negatives; C retains approximately 90.00% and 93.33%. The 95% distinct-image guardrails are explicit conservative design checks in addition to the initial 95% draw-exposure goal, not clinically justified performance thresholds.

Fracture object exposure after cropping equals its pre-crop exposure in **every simulated epoch**. Sampling can still change the number of boxes represented by a fixed number of positive images. Across all 240 epochs per candidate, A represents 12,402–12,450 fracture boxes per epoch versus 12,422 at baseline, B 12,387–12,477, and C 12,380–12,486. A's worst observed ratio is 99.84%. Every fracture-positive patient remains represented, but neither counts nor patient coverage guarantee preservation of AP or error patterns.

### Per-class exposures, including clipping/omissions

Cells show mean **before → after crop** and the post-crop change from standard. Object labels and source images are separate denominators; categories overlap.

| Objects exposure | Standard | A | B | C |
|---|---|---|---|---|
| fracture | 12,422 | 12,429.29 → 12,429.29 (+0.06%) | 12,431.26 → 12,431.26 (+0.07%) | 12,438.11 → 12,438.11 (+0.13%) |
| metal | 600 | 618.73 → 618.73 (+3.12%) | 635.31 → 635.31 (+5.89%) | 670.09 → 669.09 (+11.51%) |
| text | 16,388 | 16,389.05 → 16,357.50 (-0.19%) | 16,391.76 → 16,312.46 (-0.46%) | 16,385.89 → 16,203.29 (-1.13%) |
| periostealreaction | 2,497 | 2,690.30 → 2,690.30 (+7.74%) | 2,941.96 → 2,941.96 (+17.82%) | 3,378.32 → 3,378.32 (+35.30%) |
| pronatorsign | 400 | 456.00 → 456.00 (+14.00%) | 544.65 → 544.65 (+36.16%) | 690.27 → 690.27 (+72.57%) |
| bonelesion | 35 | 35.00 → 35.00 (+0.00%) | 35.00 → 35.00 (+0.00%) | 35.00 → 35.00 (+0.00%) |
| softtissue | 324 | 396.99 → 396.99 (+22.53%) | 479.73 → 479.73 (+48.06%) | 650.31 → 650.31 (+100.71%) |
| boneanomaly | 205 | 262.54 → 262.54 (+28.07%) | 306.40 → 306.40 (+49.46%) | 409.94 → 409.94 (+99.97%) |

| Image exposure | Standard | A | B | C |
|---|---|---|---|---|
| fracture | 9,307 | 9,307.00 → 9,307.00 (+0.00%) | 9,307.00 → 9,307.00 (+0.00%) | 9,307.00 → 9,307.00 (+0.00%) |
| metal | 518 | 533.54 → 533.54 (+3.00%) | 548.41 → 548.41 (+5.87%) | 578.94 → 577.94 (+11.57%) |
| text | 14,030 | 14,029.70 → 14,005.70 (-0.17%) | 14,029.24 → 13,966.51 (-0.45%) | 14,028.41 → 13,879.44 (-1.07%) |
| periostealreaction | 1,590 | 1,713.72 → 1,713.72 (+7.78%) | 1,875.42 → 1,875.42 (+17.95%) | 2,150.01 → 2,150.01 (+35.22%) |
| pronatorsign | 399 | 454.82 → 454.82 (+13.99%) | 543.27 → 543.27 (+36.16%) | 688.62 → 688.62 (+72.59%) |
| bonelesion | 32 | 32.00 → 32.00 (+0.00%) | 32.00 → 32.00 (+0.00%) | 32.00 → 32.00 (+0.00%) |
| softtissue | 308 | 378.59 → 378.59 (+22.92%) | 457.76 → 457.76 (+48.62%) | 621.19 → 621.19 (+101.68%) |
| boneanomaly | 147 | 189.40 → 189.40 (+28.84%) | 222.09 → 222.09 (+51.08%) | 296.80 → 296.80 (+101.90%) |

Protected rare-image and independent-patient coverage remains complete. Cropping removes mostly fully outside text annotations: A omits a mean 31.55 text boxes/epoch and clips 2.275 non-target annotations; B 79.30 and 4.713; C 182.60 and 9.263. C also omits a mean one metal box/epoch. There are no omitted fracture boxes. Full-image parent views remain available, but the distribution shift for text/metal can affect the overall eight-class selection metric and must not be ignored.

### Repetition, concentration and robustness

Image repetition percentiles include all original images, including zero draws in an epoch. Patient repetition uses all 4,264 patients. Patient-relative repetition divides a patient's draws by its original image count; equal patient counts alone would conceal oversampling.

| Policy | Image repeats/epoch: mean p95 / mean p99 / max | Patient draws/epoch: mean p99 / max | Total image repeats: min / p05 / median / p95 / p99 / max | Total patient relative: min / p99 / max |
|---|---|---|---|---|
| standard | 1.00 / 1.00 / 1 | 12.00 / 30 | 80 / 80 / 80 / 80 / 80 / 80 | 1.000x / 1.000x / 1.000x |
| A | 1.00 / 2.00 / 2 | 12.05 / 32 | 70 / 75 / 79 / 89 / 114 / 136 | 0.931x / 1.288x / 1.600x |
| B | 1.00 / 2.00 / 2 | 12.39 / 33 | 63 / 71 / 77 / 102 / 131 / 160 | 0.887x / 1.475x / 2.000x |
| C | 2.00 / 3.00 / 3 | 12.94 / 35 | 53 / 63 / 73 / 125 / 175 / 240 | 0.808x / 1.944x / 2.000x |

For seed 42, A has a mean 102.2 patients/epoch above 1.25x original exposure, and 1.4 patients/epoch at 2x, maximum four in an epoch. B has 223.8 and 2.6, while C has 373.1 and 131.7. This is why C's complete patient coverage is insufficient evidence of a balanced intervention. Over the complete budget, A's maximum patient-relative exposure is 1.60x, B/C 2.00x. All candidates empirically visit every one of the 14,061 images and 4,264 patients over 80 epochs; this measured union is not a guarantee for an arbitrary short simulation.

Complete union coverage does not mean equal exposure to every view: in seed 42, A’s least-sampled image appears 70 times rather than 80, and its least-exposed patient receives 93.1% of that patient’s original cumulative draw count. B reaches 63 views / 88.75%, and C 53 / 80.78%. These tails remain a limitation even though every patient is represented every epoch.

The stress seeds give similar mean exposure gains, without selecting a favorable seed:

| Post-crop object gain: range of seed means | A | B | C |
|---|---|---|---|
| periostealreaction | 7.74%–7.77% | 17.82%–18.11% | 35.30%–35.64% |
| pronatorsign | 14.00%–14.12% | 35.80%–36.22% | 71.82%–72.57% |
| softtissue | 22.40%–22.74% | 48.02%–48.28% | 99.41%–101.03% |
| boneanomaly | 27.92%–28.07% | 49.02%–49.77% | 99.32%–99.97% |

Across all seeds A accepts 36–70 crops per epoch; B 82–122; C 196–253. A's per-epoch crop acceptance is only about 19% of attempted crop modes, with the rest falling back. Common reasons include no small eligible target, insufficient zoom with context, visible annotation fragments and fracture retention. Detailed fallback totals and pre/post co-occurrence matrices remain available, including negative results.

## 7. Bone lesion remains a limited-independent-support category

All candidates retain exactly **32 bone-lesion images, 35 boxes and 16 patients per epoch**, with no extra lesion image repetitions or lesion-patient bonus draws. That is 2,560 source-image appearances and 2,800 box appearances over 80 epochs, not additional independent evidence. The largest patient contributes seven images (21.875% of lesion images) and ten boxes (28.57% of lesion boxes). This existing concentration cannot be solved by repeated crops or oversampling.

There is a separate limitation: those 16 patients have 66 training images in total. Their non-lesion views can participate in the common-image omissions. For A, cohort draw count averages 98.13% of the original cohort total, with a minimum 95.45% in seed 42; all lesion-positive images and all 16 patients remain present. The worst A cohort ratio is also 95.45% across all three seeds; its mean ranges from 98.13% to 98.92%. B/C have larger reductions in those other views (all-seed minima 90.91% and 86.36%). We report this distinction instead of describing lesion-patient coverage as unchanged exposure for every view. Protecting all 66 cohort views would be a different candidate requiring another declared simulation and approval.

Bone-lesion validation support is tiny; its reported baseline AP is descriptive only and never used in this policy. No improvement in lesion generalization, or reduction in memorization, can be concluded from these simulations.

## 8. Bounded prototype and automated verification

Fifteen deterministic training cases cover all eight classes, accepted targets of all four eligible classes, image borders, overlapping/multiple boxes, the narrowest and widest aspect ratios, and all observed fallback categories. The same parents are compared across A/B/C: **45 comparisons**. Prepared YOLO labels were decoded independently and matched to frozen original-coordinate boxes. Crop pixels equal the exact source subarray; every retained coordinate is independently reconstructed from crop edges; source PNG and label hashes remain unchanged.

All 15 comparison sheets were technically inspected. Accepted overlays remained aligned; fallback panels retained the complete original view and labels. Several tighter crops lose hand or shaft context even when annotation rules pass, so visual acceptance is not a clinical-context certification. Source low contrast/cast texture is preserved. Some non-target boxes undergo allowed clipping, and text/metal boundaries can prevent a tighter crop. The original and repeated prototype runs produced **60 identical PNG hashes**. Images, transformed labels, source identities and patient mappings remain under ignored `outputs/class_aware_design/visual_review_verified/`.

**91 unit tests pass**, including 17 new design tests and 74 existing tests; `pip check` passes. Tests cover transformation/visibility, all retained labels, invalid/degenerate geometry, borders, wide/narrow aspect, minimum size, full-target/fracture retention, fallback, empty labels, pixel/source immutability, deterministic ordering, image/patient caps, saturation, held-out-row rejection and continued validity of the frozen baseline lock. One early synthetic assertion incorrectly assumed that a three-epoch simulation must cover every common image; it was corrected to validate the actual union, preserving the distinction between measured coverage and a guarantee.

An independent reconciliation of saved arrays checks all 960 epochs, patient and image caps, lesion coverage, exact draw/status totals, retained label counts and the full-view-per-parent rule. It reproduces boundary epochs from reversed records. See [simulation verification](simulation_verification.json), [prototype verification](prototype_verification.json) and [test receipt](verification.json).

## 9. Recommendation and decisions for review

**Recommend Candidate A as the next proposed policy.** It provides measurable exposure increases for three underrepresented findings and a smaller increase for periosteal reaction, with less distinct-view loss and cumulative repetition than B/C. It preserves every patient and remains comfortably above the distinct-image and fracture-object exposure guardrails. B is a plausible stronger alternative, but no simulated count establishes that its additional repetitions will improve AP. C is unsuitable under the stated diversity checks.

Approval is needed for the full concrete proposal: 2% requested redistribution; max two draws/image; per-patient extra cap `min(2, ceil(0.25*N_p))`; status-stratified and patient-anchored coverage; no bone-lesion cohort extras; maximum rather than summed rarity; and A's conservative crop parameters, including the low realized crop rate, 90% non-target visibility and full fracture/target retention. The separate 95% distinct-image guardrails and allowed singleton 2x cases also need acknowledgment. None of these settings is final merely because its hash is recorded.

Next implementation requires explicit authorization after review. It should add a separate class-aware dataset/sampler without altering the preserved baseline's locked implementation, then verify transform order, native augmentation, worker determinism, draw budgets and performance. This step does not authorize a smoke epoch or official training. Both future models must retain the same architecture, verified COCO initialization, frozen splits, eight labels, 768/batch 6/steady accumulation 4, AdamW/schedule/common augmentation, AMP safeguards, 80 epochs, 14,061 draws and the same validation/checkpoint rule. The nominal attempt budget is 47,433; actual successful counts must be recorded. Do not add epochs to compensate for AMP skips. The class-aware model must never initialize from the trained standard checkpoint.

The study evaluates the combined sampling-and-cropping strategy. Without additional approved ablations, any later difference cannot be attributed separately to sampling or cropping. Simulated coverage cannot establish AP preservation, clinical safety or generalization. The held-out test remains reserved for the final locked comparison.

## 10. Reproduction and identities

Implementation checkpoint: `0db961a`; baseline preservation checkpoint: `3baadbb`. Proposed candidate configuration SHA-256: `fb36c515662976eee1d35079dad0aeeabc6f5a7ba60a750118238b7594b76edb`. [Design provenance](design_provenance.json) records source hashes, pinned environment, frozen training-container/split/preparation identities, baseline identity, aggregate hashes and deterministic draw-array hashes. The record is **proposed**, not an approved training lock.

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m wrist_design.baseline
.\.venv\Scripts\python.exe -m wrist_design.analyze
.\.venv\Scripts\python.exe -m wrist_design.simulate
.\.venv\Scripts\python.exe -m wrist_design.geometry
.\.venv\Scripts\python.exe -m wrist_design.verify
.\.venv\Scripts\python.exe -m wrist_design.prototype --attempt visual_review_repeat
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The baseline command is read-only and checks the original private inventory. The simulator rewrites only its own design outputs, never baseline/data artifacts. Use a fresh prototype attempt name; existing visual directories are protected. A new prototype run requires a new actual visual review before its receipt is marked reviewed. Repeating simulation seeds is a reproducibility check, not permission to search seeds or optimize validation AP. No command above starts model training.
