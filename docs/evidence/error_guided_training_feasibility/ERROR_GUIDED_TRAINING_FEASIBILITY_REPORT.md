# Training-only error mining favors a simpler rarity-only intervention

Phase 4A engineering feasibility report, 2026-10-17. **Recommendation: SIMPLIFY — Rarity-only.** This is a proposal for student review, not permission to train or a replacement of the historical A-F thesis design.

## Executive summary

- Complete training-only baseline inference finds 6,479 annotated objects with detection/localization difficulty, including 4,118 in the five bonus-priority clinical classes. There are 3,087 eligible difficult images; 1,208 are outside rarity priority. Difficulty therefore contains some additional information, but it is strongly related to rarity (image Spearman .654).
- All 720 simulated epochs / 10,123,920 source-image draws pass individual coverage, repetition and patient/cohort guardrails. Both proposed samplers realize 625 replacements per epoch (4.445%); 77 requested slots are restored to unique no-fracture-box views because that rarity pool has only 160 cap-safe extras.
- PAEG-CAT changes only 105.4 bonus memberships per epoch on average, 0.750% total variation in source frequencies; changed ranking also changes 4.39% of ordered draw positions. It adds 51.0 priority-target hard-positive presentations versus rarity-only, about 1.08%, while reducing exposure to several intended rare classes. That marginal increment does not justify a separate full error-guided model on the available evidence.
- Prefer a later student-approved comparison of the preserved standard baseline against full-image patient-aware rarity-only sampling. The simpler policy already raises rare annotation exposure by roughly 18%–49%. Neither intervention has a measured training/AP benefit. Preserve this result and stop: no production integration, official fitting, validation/test inference or matrix revision occurs in this phase.

## 1. Authorization, inputs and completed work

The direct Phase 4A request permits training-only inference/analysis, isolated sampler prototypes and simulations. Existing project guidance, baseline/checkpoint records, A/B/C design/simulations, Phase 3B findings and negative CG-ALR findings were reviewed as context. Previous validation findings do not enter any difficulty calculation, weight or training-example choice. No new validation/test image or prepared-label contents are opened. The shared frozen metadata container is hash-verified and filtered to train before analytical/image/label use; that container includes other split records, which are discarded rather than used as mining inputs.

The complete frozen training population is 14,061 images, 4,264 patients and 32,871 target boxes, with 9,307 fracture-positive and 4,754 no-fracture-box images. The latter is an annotation stratum, not confirmed normal anatomy. Source/prepared geometry, labels, class mapping, split, augmentation and protocol stay unchanged. All 14,061 training image and 14,061 training label hashes match the conversion journal.

Implementation is isolated in `src/wrist_mining/`: `common.py` rejects non-training paths; `inference.py` binds the checkpoint/config/source and saves protected per-image predictions; `difficulty.py` reuses existing one-to-one categories; `analysis.py` computes training distributions/checkpoint sensitivity; `qualitative.py` creates private audit sheets; `sampling.py` implements bounded paired tickets; `simulate.py` records source-index plans; `verify.py` independently checks predictions and all saved plans. The official trainer and pinned evaluator are unchanged. No dependency is added.

| Input | Identity |
| --- | --- |
| Mining config canonical SHA-256 | d1cbbb1c535aa82ed99ecfe623b912ee8c9c173c32e1abd71724654cc088cbf7 |
| Sampler proposal canonical SHA-256 | c7d09a95f1d3242cbf32127880d7c51bf0ff295974c4fb70304cd5f8ff2283f2 |
| Frozen split SHA-256 | ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49 |
| Common 80-epoch protocol SHA-256 | 281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01 |
| Training difficulty feature-content SHA-256 | 6cfd57c56455150491a3cf00dfed1a737727096da7b506e1da0bcaa77a3ad22b |
| Best training-prediction file SHA-256 | 7415e46e4a02fe86f5d90fdaedb8e3389dc317379921842281926d83d3f57f3b |
| Historical class-aware proposal canonical SHA-256 | fb36c515662976eee1d35079dad0aeeabc6f5a7ba60a750118238b7594b76edb |

| Checkpoint | Completed epoch | Images inferred | SHA-256 |
| --- | --- | --- | --- |
| best.pt | 22 | 14061 | 31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b |
| epoch010.pt | 10 | 1087 | 656abbcd4a28278d03c4f42e7f11ef4c9eb1e7cfe3decf941560d0dc6e9cf6a5 |
| epoch040.pt | 40 | 1087 | 7a26b5fe2bd19d24a824522d4f0c4facc1f21765cb35d5bacec64e5bf1d05cc9 |

| Mining receipt | Loop incl read/hash/export/replay, s | Decode-to-output mean, ms | Peak allocated / reserved, MiB |
| --- | --- | --- | --- |
| best.pt | 318.504 | 20.049 | 79.70 / 116.00 |
| epoch010.pt | 23.426 | 19.648 | 79.11 / 112.00 |
| epoch040.pt | 22.435 | 18.975 | 79.11 / 112.00 |

These diagnostic loop times exclude initial label-hash preflight, model load and synthetic warmup; synchronized per-image times exclude file read/hash. Checkpoint sensitivity adds only two small existing-model passes, with no fitting. These measurements are not comparable speed claims against previous validation sessions and are not training-cost estimates.

`best.pt` is the preserved epoch-22 EMA selected during the completed 80-epoch baseline. Existing epochs 10/40 were chosen by training time before mining, without inspecting their scores. Their predefined subset is the union of 1,024 hash-ordered training images and up to 32 images for each clinical class: 1,087 distinct images. These are existing checkpoints, not additional models.

Full-image inference reuses the unchanged Phase 3B adapter: unfused FP32 EMA, normalized FP16 inputs and CUDA FP16 autocast, 768 maximum dimension, confidence .001, multi-label class-aware NMS .70, max_det 300 and verified original-coordinate restoration. Default half-stride padding uses training dimensions only; no saved validation shape profile is used. Each checkpoint has three synthetic warmups and three exact repeated training predictions. All outputs are finite/valid and all bounded replay arrays are bitwise equal. Total actual YOLO mining forwards are 16,253, including nine warmups and nine replays; no successful full inference is repeated.

## 2. What the mining categories mean

At confidence .25, use the unchanged descriptive matcher at IoU .50 and .75: confidence-ordered correct-class predictions match each annotation at most once. Remaining categories consume residual predictions/annotations once, in the established order: classification explanation, correct-class low confidence, localization with IoU>=.10, then no residual candidate. These are mutually exclusive descriptive reasons at each threshold, not independent clinical diagnoses. Native AP implementation is unchanged; training AP is not presented as independent performance.

An annotation is a hard positive if it is not matched at .50 or .75. Define `d_g=1` if unmatched at .50; otherwise `.5` if unmatched at .75; otherwise 0. Thus many hard positives are localized loosely enough for .50 but fail the stricter .75 criterion; they must not all be called completely missed objects. Per-image/class score is the mean over that class's annotations. Unmatched >=.25 predictions are potential false alarms; >=.50 counts are reported separately. No unmatched prediction is automatically a verified negative.

### Class-wise in-sample counts

HP below means the union of detection/localization failures. FP means annotation-relative unmatched prediction at .25/IoU .50. All eight targets are retained, including artifacts and non-fracture findings.

| Class | Objects | Patients | TP .50 | FN .50 | FN .75 | HP fraction | Potential FP | FP score>=.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 12422 | 2511 | 11976 | 446 | 1706 | 13.73% | 705 | 197 |
| metal | 600 | 109 | 577 | 23 | 85 | 14.17% | 99 | 30 |
| text | 16388 | 4264 | 16189 | 199 | 2265 | 13.82% | 685 | 255 |
| periostealreaction | 2497 | 763 | 1629 | 868 | 1754 | 70.24% | 1025 | 37 |
| pronatorsign | 400 | 389 | 273 | 127 | 250 | 62.50% | 171 | 7 |
| bonelesion | 35 | 16 | 28 | 7 | 11 | 31.43% | 7 | 1 |
| softtissue | 324 | 283 | 63 | 261 | 277 | 85.49% | 44 | 4 |
| boneanomaly | 205 | 67 | 98 | 107 | 131 | 63.90% | 34 | 2 |

| Class: reason for FN at .50 | Classification | Low confidence | Localization | No residual candidate |
| --- | --- | --- | --- | --- |
| fracture | 1 | 409 | 21 | 15 |
| metal | 0 | 12 | 8 | 3 |
| text | 1 | 110 | 66 | 22 |
| periostealreaction | 0 | 790 | 22 | 56 |
| pronatorsign | 1 | 125 | 0 | 1 |
| bonelesion | 1 | 4 | 0 | 2 |
| softtissue | 3 | 257 | 0 | 1 |
| boneanomaly | 5 | 91 | 1 | 10 |

Across all classes there are 30,833 matched / 2,038 missed annotations at .50, and 26,392 matched / 6,479 missed at .75. For the five priority classes (fracture, periosteal reaction, pronator sign, soft tissue, bone anomaly), there are 4,118 hard annotations, including 1,809 missed at .50. Bone lesion is analyzed but excluded from bonus priority because only 16 independent training patients exist. Simulation fields named `clinical_*` refer to these **five priority classes**, not every clinical target; the unchanged lesion cohort contributes no extra exposure.

The model already fits many training fracture boxes at .50, whereas soft tissue/periosteal/pronator/anomaly remain difficult even in sample. This supports class imbalance/label/visibility concerns, not proof that every difficult box is informative or wrong. Correct-class weak predictions, loose localization and no residual candidate remain distinct in private per-image details.

### Object size

Cells are hard annotations / total annotations using idealized short side at 768, with original pixels/geometry preserved. These bins are descriptive, not sampling thresholds. Exact counts are in `size_strata.csv`.

| Class | <16 px | 16–<32 | 32–<64 | 64–<128 | >=128 |
| --- | --- | --- | --- | --- | --- |
| fracture | 0/0 | 96/163 | 1141/7231 | 448/4898 | 21/130 |
| metal | 3/4 | 25/33 | 34/90 | 20/220 | 3/253 |
| text | 840/1642 | 1265/7504 | 153/6648 | 7/593 | 0/1 |
| periostealreaction | 356/363 | 921/1174 | 436/853 | 41/107 | 0/0 |
| pronatorsign | 0/0 | 2/2 | 114/159 | 131/236 | 3/3 |
| bonelesion | 2/2 | 5/19 | 2/11 | 0/0 | 2/3 |
| softtissue | 0/0 | 0/0 | 60/67 | 189/228 | 28/29 |
| boneanomaly | 0/0 | 22/30 | 45/72 | 57/87 | 7/16 |

## 3. Patient concentration, rarity and annotation reliability

Detailed image/patient distributions remain private. Public patient distributions include all 4,264 patients, retaining zero counts rather than presenting only difficult patients. Class-positive patient hard-fraction distributions and concentration relative to all annotations are in `training_mining.json`; reason counts per patient are summarized in `patient_error_distributions.json`.

| Class | Patients with hard objects | Effective hard-object patients | Top 1% patient share of hard objects | Median hard fraction among class-positive patients |
| --- | --- | --- | --- | --- |
| fracture | 1028 | 685.5 | 14.36% | 0.000 |
| metal | 41 | 29.3 | 100.00% | 0.000 |
| text | 1357 | 875.3 | 12.27% | 0.000 |
| periostealreaction | 670 | 419.6 | 19.95% | 0.750 |
| pronatorsign | 246 | 242.2 | 18.80% | 1.000 |
| bonelesion | 9 | 8.1 | 100.00% | 0.350 |
| softtissue | 245 | 219.9 | 27.08% | 1.000 |
| boneanomaly | 61 | 30.1 | 86.26% | 1.000 |

For priority-class hard objects, 1,583 patients contribute; effective support is about 849 patients. The top 43 of all 4,264 patients account for 12.41% of these hard objects; the top 10% account for 57.92%. All-annotation shares are 6.58% and 37.97%, respectively. Difficulty is more concentrated than raw annotation volume, so patient caps are substantive safeguards. Eligibility excludes whole lesion patients, leaving 1,579 patients with nonzero sampler difficulty; this explains the small denominator difference.

There are 3,087 nonzero eligible image scores: 1,879 with rarity priority and 1,208 without it. The latter contain 1,261 hard fracture boxes in error-eligible images. Image Spearman correlation between R and D is .654. Additional information exists, particularly fracture localization, but high rare-class hard rates explain substantial redundancy. Absent classes have zero score, so this correlation is descriptive of the training population and not a causal class-effect estimate.

The release marks `diagnosis_uncertain=1` on 360 training images; 13,701 entries are blank. Blank is unmarked/missing, not a diagnosis of certainty. Of marked images, 52 have nonzero eligible difficulty and 28 have potential FP predictions. This image-level flag is incomplete as a box-quality measure and never enters the sampler. Absence of a mark in every selected audit case cannot prove clean annotations.

### Bounded potential-hard-negative audit

The 2,770 unmatched predictions at .25 include 654 duplicates, 36 classification explanations, 949 localization explanations and 1,131 background-or-unannotated predictions. Only 533 have confidence>=.50; confidence alone cannot establish that their regions are truly negative. Most of the pool is therefore already explained by overlap, extent or class semantics rather than reliable background negatives.

All 24 predefined figures were actually inspected in six private review sheets: 16 potential-FP cases (two/class from distinct patients), six hard positives (one/clinical class), two positive controls. No absent category or hand-picked replacement was needed. This is a class-balanced bounded technical audit, not a representative prevalence estimate or expert clinical adjudication.

Observed issues include whole-hardware versus component-box granularity, duplicate narrow periosteal boxes, a tiny text-marker extent mismatch, an apparent image-edge stripe classified as text, bone-lesion predictions over anomaly annotations and unresolved cortical/soft-tissue regions. Several unmatched predictions refer to the same visible object as an annotation with different extent or class, so labeling them negative would be misleading. The six hard-positive examples include four .50-to-.75 localization failures and two weak-confidence explanations. Controls are correctly localized annotated findings, not healthy-image controls.

There are **zero clinically adjudicated negatives**. Recommend excluding all potential-FP-derived negative/difficulty terms from the initial sampler. Keep annotations unchanged; neither relabeling nor removing suspicious cases is performed. Annotated hard positives are more traceable than inferred negatives, but difficult annotation extents/category boundaries can still amplify noise.

## 4. Quantifying the in-sample limitation

All 14,061 mining images and all 4,264 patients appeared in baseline fitting. The selected checkpoint completed 22 epochs, nominally 309,342 source-image presentations before augmentation. Thus 100% of these difficulty estimates are in sample, and the teacher was chosen by earlier validation metrics. No current-phase validation/test outcome is used for scoring, but the inherited validation-selected checkpoint dependence remains. Rarity-only avoids this extra teacher-selection dependency.

The same-trajectory sensitivity subset is enriched for rare classes and not population weighted. It cannot establish external generalization, independent-model reliability or unbiased error rates. The fixed epochs 10/40 are correlated snapshots, not out-of-fold predictions.

| Existing checkpoint vs best22 | Image D Spearman | Nonzero-D Jaccard | Top-decile Jaccard |
| --- | --- | --- | --- |
| epoch010.pt_versus_best | 0.7423 | 0.5929 | 0.3797 |
| epoch040.pt_versus_best | 0.7209 | 0.5726 | 0.4342 |

| Class | Hard-object Jaccard: epoch10 / best | Hard-object Jaccard: epoch40 / best |
| --- | --- | --- |
| fracture | 0.3654 | 0.3798 |
| metal | 0.4444 | 0.8889 |
| text | 0.7171 | 0.7952 |
| periostealreaction | 0.7517 | 0.6552 |
| pronatorsign | 0.6857 | 0.6250 |
| bonelesion | 0.3333 | 0.4545 |
| softtissue | 0.9000 | 0.6667 |
| boneanomaly | 0.7209 | 0.3548 |

Global rank correlation is moderate, but top-decile membership changes considerably. Discrete scores create large boundary ties (96/193/67 tied cases for best/epoch10/epoch40 under the recorded cutoff), resolved by a fixed hash. A specific top-hard list is less reliable than its broad difficulty pattern. The sampler uses bounded stochastic priority rather than treating that list as independent truth. No extra model or out-of-fold training is attempted.

## 5. Exact proposed sampler

The unchanged historical A/B/C proposal remains separate. `configs/error_guided_sampler_proposed.json` and `SAMPLER_PROPOSAL.md` record the current full-image feasibility proposal before simulations, at commit `bc5cb6a`.

- Candidate-B rarity: for c in {3,4,6,7}, `r_c=min(2,sqrt(1590/I_c)*min(1,P_c/100))`, with zero below 32 independent patients. `R_i=max_c(r_c*presence_ic)/2`, bounded [0,1].
- Annotation difficulty: `d_g=1` for FN .50, otherwise `.5` for FN .75, otherwise 0. `D_ic=mean(d_g)` in each present class, absent classes zero. `D_i=max_c D_ic` for c in {0,3,4,6,7}; whole bone-lesion patients have zero bonus priority.
- Compare `W_rarity=R_i` against `W_error=.5R_i+.5D_i`. Equal coefficients are a simple feasibility hypothesis between bounded components, not an optimum or a 50/50 draw quota. No FP count or uncertainty flag enters either weight.
- Draw a common coverage skeleton using existing patient/status anchors and rare/empty-image protection, strengthening it to retain every original view of every lesion patient. Determine feasible realized per-stratum extra counts without difficulty. Both candidates reallocate exactly those slots using the same per-image uniforms `u_i(seed,epoch)` and exponential key `-log(u_i)/W_i`, same cap-feasible allocator and final permutation stream. Alpha=0 gives exactly identical paired plans in a focused test.
- Maximum two draws/image. Patient extras `min(3,ceil(.5*N_p))`, zero for lesion patients. Ceiling permits singleton 2x exposure; it is not a strict 50% relative bound. An exact two-stratum residual-capacity check prevents weighted ordering from starving a required stratum; it adds no patient preference weight.
- Exactly 14,061 draws and original 9,307/4,754 fracture-status draw totals per epoch; all patients; >=8,842/4,517 distinct images in those strata; full images only, original geometry/labels, unchanged augmentation/seed/protocol. No targeted cropping is used.

Strict 95% unique coverage caps a nominal 5% redistribution at 702, not rounded 703. In the no-fracture-box rarity pool, 195 eligible images across 138 patients permit only 160 extras under the patient caps. Thus both candidates realize 465 fracture-positive plus 160 no-fracture-box extras =625, restoring 77 unique no-fracture-box views. Both retain identical coverage sets and matched actual budgets. A full 702 intervention is not silently claimed. A larger replacement budget would conflict with the present coverage/pool/cap constraints rather than being a harmless coefficient change.

| Class priority | Periosteal | Pronator | Soft tissue | Bone anomaly |
| --- | --- | --- | --- | --- |
| Raw r_c (before /2) | 1.000000 | 1.996237 | 2.000000 | 2.000000 |

## 6. Matched 80-epoch simulations

Three fixed policies x 80 epochs x seeds 42,1042,2042 =720 epochs and 10,123,920 source-image draws. Each policy/seed has 1,124,880 draws. All source-index plans are private and hash-bound. Every epoch passes; no failed epoch is hidden by an average. These are nominal source annotation exposures before native random affine/box-candidate filtering, not actual model training or independent new patient information.

| Policy | Unique images/epoch | Unique fracture / no-fracture views | Patient coverage | Realized replacements | Full-image draws |
| --- | --- | --- | --- | --- | --- |
| Standard uniform | 14061 | 9307 / 4754 | 4264 | 0 | 14061 |
| Patient-aware rarity-only | 13436 | 8842 / 4594 | 4264 | 625 | 14061 |
| PAEG-CAT (.5R+.5D) | 13436 | 8842 / 4594 | 4264 | 625 | 14061 |

| Every-epoch guardrail | Rarity-only | PAEG-CAT |
| --- | --- | --- |
| Fracture-positive distinct coverage | 95.00376% | 95.00376% |
| No-fracture-box distinct coverage | 96.63441% | 96.63441% |
| All training patients | 4,264 / 4,264 | 4,264 / 4,264 |
| Maximum source-image draws | 2 | 2 |
| Maximum patient extras | 3 | 3 |
| Largest observed patient draws | 33 | 33 |
| Largest relative patient draw ratio | 2x | 2x |
| All 66 views of 16 lesion patients | Exactly once each epoch | Exactly once each epoch |

### Per-class object exposure

Values are means across all 240 seed/epoch allocations per policy. Gains are relative to standard nominal source exposure, not AP changes.

| Class | Standard | Rarity-only | Gain | PAEG-CAT | Gain | PAEG minus rarity |
| --- | --- | --- | --- | --- | --- | --- |
| fracture | 12422.00 | 12429.10 | +0.06% | 12435.52 | +0.11% | +6.43 |
| metal | 600.00 | 635.06 | +5.84% | 641.53 | +6.92% | +6.47 |
| text | 16388.00 | 16387.11 | -0.01% | 16385.82 | -0.01% | -1.29 |
| periostealreaction | 2497.00 | 2946.71 | +18.01% | 2892.47 | +15.84% | -54.24 |
| pronatorsign | 400.00 | 544.67 | +36.17% | 507.71 | +26.93% | -36.96 |
| bonelesion | 35.00 | 35.00 | +0.00% | 35.00 | +0.00% | +0.00 |
| softtissue | 324.00 | 479.36 | +47.95% | 466.25 | +43.90% | -13.11 |
| boneanomaly | 205.00 | 306.08 | +49.31% | 300.62 | +46.64% | -5.46 |

| Class-positive image exposure | Standard | Rarity-only | PAEG-CAT | Independent patient support |
| --- | --- | --- | --- | --- |
| fracture | 9307.00 | 9307.00 | 9307.00 | 2511 |
| metal | 518.00 | 548.54 | 553.07 | 109 |
| text | 14030.00 | 14029.28 | 14029.14 | 4264 |
| periostealreaction | 1590.00 | 1875.74 | 1838.90 | 763 |
| pronatorsign | 399.00 | 543.32 | 506.41 | 389 |
| bonelesion | 32.00 | 32.00 | 32.00 | 16 |
| softtissue | 308.00 | 457.71 | 445.45 | 283 |
| boneanomaly | 147.00 | 221.58 | 217.75 | 67 |

Independent rare-class patient coverage is complete in both samplers because rare source images are protected. More image/box exposure does not increase these patient counts. Current rarity-only is full-image only and protects the entire lesion cohort; the historical A/B crop experiments remain intact and are not treated as the same intervention.

| Exposure metric per epoch | Standard | Rarity-only | PAEG-CAT | PAEG vs rarity |
| --- | --- | --- | --- | --- |
| clinical_hard_positive_exposure | 4118.00 | 4726.42 | 4777.45 | +1.08% |
| clinical_missed50_exposure | 1809.00 | 2184.38 | 2225.55 | +1.88% |
| hard_positive_exposure | 6479.00 | 7106.70 | 7157.21 | +0.71% |
| potential_fp_exposure | 2770.00 | 2949.43 | 2939.37 | -0.34% |
| source_uncertainty_mark_exposure | 360.00 | 380.11 | 381.77 | +0.44% |

Potential-FP exposure counts how often source images carrying baseline unmatched predictions are revisited; no negative label is added and this quantity is never a selection reward. Source-uncertainty exposure counts marked images, not known mislabeled objects. Their increase over uniform is a co-occurrence risk that remains under rarity-only too. Neither policy proves it avoids all questionable annotations.

### Contribution of difficulty beyond rarity

| Paired selection diagnostic, all 240 allocations | Result |
| --- | --- |
| Different bonus images/epoch | 105.425 (range 84–127) |
| Bonus-set Jaccard | 0.7115 (range 0.6622–0.7630) |
| Whole-epoch source-frequency total variation | 0.7498% |
| PAEG bonus images without rarity priority | 85.87/625 |
| Mean difficulty of bonus draws | 0.6043 → 0.6662 |
| Added priority hard-object presentations | 51.03 / epoch (1.08%) |
| Added genuinely missed-at-.50 priority presentations | 41.18 / epoch (1.88%) |

| Ordering diagnostic, distinct from source-frequency overlap | Result |
| --- | --- |
| Different ordered positions per epoch | 617.604 / 14,061 (4.392%) |
| Different ordered positions: min/max | 603 / 624 |

The 0.750% value is total variation of source frequencies, not ordered-plan disagreement. Weighted ranking reorders common bonus images too, so 4.39% of ordered positions differ even though the shared coverage component is identical. Batch composition/augmentation pairing can consequently change. Their learning effect is unmeasured; this report does not infer AP equivalence or a negligible gradient effect from the small frequency shift.

The error component does select harder bonus images and reach some fracture examples outside rarity priority. It is not identically redundant. About one in six bonus memberships changes, corresponding to less than 1% total variation in full-epoch source frequencies. Ordered-plan changes are larger, as reported above. The total priority hard-positive increase over rarity-only is about 1.08%, with missed-at-.50 exposure about 1.89% higher. These are measured exposure changes, not estimates of their learning effect.

Rarity-only already increases priority hard-positive exposure about 14.77% over uniform; PAEG reaches about 16.01%. It trades away some periosteal/pronator/soft-tissue/anomaly exposure for harder within-class and fracture views. There is no evidence here that the trade will improve generalization. Shared random image keys and exact zero-difficulty controls make the measured difference attributable to the specified priority change rather than unrelated lottery noise.

### Across-seed and cumulative tails

Means/minima over epochs below describe sampler randomness, not independent patient/model confidence intervals. The same data and teacher are reused; 240 allocations must not be interpreted as 240 independent experiments.

| Seed | Rarity priority HP mean | PAEG priority HP mean | Increment | Rarity / PAEG missed-.50 mean |
| --- | --- | --- | --- | --- |
| 42 | 4724.262 | 4776.238 | +51.975 | 2184.850 / 2226.438 |
| 1042 | 4727.688 | 4777.675 | +49.988 | 2184.425 / 2224.812 |
| 2042 | 4727.312 | 4778.438 | +51.125 | 2183.863 / 2225.412 |

| Policy/seed | Image cumulative min / median / p99 / max | Patient cumulative relative min / p99 / max |
| --- | --- | --- |
| standard/42 | 80.00 / 80.00 / 80.00 / 80.00 | 1.0000 / 1.0000 / 1.0000 |
| rarity_only/42 | 62.00 / 77.00 / 129.40 / 160.00 | 0.8938 / 1.4750 / 2.0000 |
| paeg_cat/42 | 62.00 / 77.00 / 130.00 / 160.00 | 0.8938 / 1.4750 / 2.0000 |
| standard/1042 | 80.00 / 80.00 / 80.00 / 80.00 | 1.0000 / 1.0000 / 1.0000 |
| rarity_only/1042 | 62.00 / 77.00 / 131.40 / 160.00 | 0.9000 / 1.4750 / 2.0000 |
| paeg_cat/1042 | 62.00 / 77.00 / 129.00 / 160.00 | 0.9000 / 1.4750 / 2.0000 |
| standard/2042 | 80.00 / 80.00 / 80.00 / 80.00 | 1.0000 / 1.0000 / 1.0000 |
| rarity_only/2042 | 63.00 / 77.00 / 129.80 / 160.00 | 0.9062 / 1.4812 / 2.0000 |
| paeg_cat/2042 | 63.00 / 77.00 / 130.00 / 160.00 | 0.9062 / 1.4812 / 2.0000 |

| Reviewed cases: cumulative source draws | Standard | Rarity-only | PAEG-CAT |
| --- | --- | --- | --- |
| Potential-FP cases | 80.00 | 85.40 | 86.04 |
| Annotated hard-positive cases | 80.00 | 97.83 | 96.78 |
| Positive controls | 80.00 | 78.00 | 78.00 |

For potential-FP audit cases, cumulative draws range67–145 under rarity-only and67–153 under PAEG. The fracture extent-disagreement case2 rises from69.67 to80.67 mean presentations; the lesion/anomaly class-conflict case11 rises142 to148. These are actual reviewed ambiguous-case repetitions, not proof of wrong annotations or a representative noise rate. Error weighting does not guarantee avoidance of questionable regions.

| Class | Rarity max-patient object share, mean | PAEG max-patient object share, mean | Rarity / PAEG effective patients, mean |
| --- | --- | --- | --- |
| fracture | 0.40% | 0.41% | 1338.35 / 1337.36 |
| metal | 4.57% | 4.67% | 65.92 / 65.36 |
| text | 0.22% | 0.22% | 2564.07 / 2564.08 |
| periostealreaction | 0.95% | 0.98% | 454.69 / 454.14 |
| pronatorsign | 0.78% | 0.77% | 338.57 / 339.65 |
| bonelesion | 28.57% | 28.57% | 8.11 / 8.11 |
| softtissue | 1.09% | 1.08% | 239.20 / 239.37 |
| boneanomaly | 7.05% | 7.15% | 32.13 / 31.68 |

Every source image and patient appears over the complete simulation, but individual views fall to 62–63 presentations instead of 80, while some reach 160. Some patients reach 2x cumulative exposure. This residual memorization/underexposure risk survives the per-epoch caps. The entire lesion cohort stays exactly at its original exposure, including 35 lesion boxes per epoch. Rare-label concentration is not solved merely by retaining every patient.

Audit-case cumulative exposures remain in `simulation_summary.json` as anonymous case ordinals. Those records show how repeated exposure affects the actual reviewed questionable regions; they are not a basis for relabeling or suppressing a case. The incomplete source uncertainty mark cannot stand in for an annotation-quality classifier.

### Is 5% too conservative?

It is globally small but not uniformly weak: rarity-only changes roughly 18%–49% of rare annotation exposure, consistent with the earlier Candidate-B exposure evidence. A small overall redistribution can therefore be a meaningful class-aware intervention. The **extra error-guided effect** has 0.750% total variation in source frequencies and changes 4.39% of ordered positions, while adding about 1% hard-positive exposure beyond rarity. Given this limited incremental exposure and the teacher/annotation limitations, another error-specific 80-epoch model is not justified on the available evidence. Its actual learning effect remains unknown.

This does not prove all 5% methods are ineffective. Neither a larger error coefficient nor a larger budget is optimized here. The 95% per-stratum unique constraint already caps replacements near 5%; raising it to 10% would require a new explicit coverage trade-off, as the earlier stronger Candidate C illustrates. Do not silently relax safeguards to manufacture a stronger distribution change.

## 7. Implementation feasibility and realistic cost

Read-only source inspection identifies a small later integration path, documented in `LOADER_FEASIBILITY.md`: keep the unchanged full-image YOLODataset/labels/transforms, supply deterministic repeated source indices through a separate training-condition sampler/launcher, and log actual consumed plans and hashes. Do not rewrite `im_files` or label arrays. The official standard trainer remains untouched.

The pinned InfiniteDataLoader persists a repeated iterator and prefetches across epochs. In single-GPU mode the trainer does not automatically call a custom sampler's `set_epoch`. A naive callback can therefore serve an old plan across an epoch boundary. Prefer a separately reviewed finite-loader training subclass with explicit pre-iterator epoch selection, or prove a reset/drain scheme. Worker lifecycle/RNG changes are material engineering differences and must be disclosed before integration. They are not implemented against real training data here.

Keep physical batch six, 2,344 batches/epoch, steady accumulation four, original transforms and seed42, original labels, verified COCO initialization and original validation/checkpoint selection policy in any later approved run. All 80-epoch policies have 1,124,880 draws and nominal 47,433 optimizer attempts including warmup ramp. Actual AMP skips/successful updates can differ with data; exact successful-update equality is not promised. The new condition must have an additive implementation/policy/mining provenance manifest and separate output directory. Never initialize from the trained mining teacher.

A later authorized bounded real-training smoke must check two-worker index/label pairing across epochs, augmentation variation, deterministic plan replay, consumed counts, loss/gradient/scaler/update health and immutable source pixels. The current synthetic tests and simulations establish none of those real loader/optimization outcomes. Production integration, smoke fitting and official training remain outside this phase.

The completed baseline measured 30,809.86 seconds of train+validation epoch work, **8.56 hours** for 80 epochs, excluding additional startup/checkpoint/loader overhead. For the same hardware/protocol reserve roughly **9–10 hours per future run** as a planning allowance, not a new measurement. PAEG plus a rarity-only ablation would require two additional models: about 17.12 hours of baseline-equivalent epoch work, roughly 18–20 hours with allowance, and an explicitly revised experiment budget. A single rarity-only model needs one such budget. Mining inference timing must not be used as a substitute for training cost.

## 8. Decision and scientific contribution

**GO — PAEG-CAT:** Not recommended in the tested form. There is additional annotated-positive difficulty (1,208 outside-rarity difficult images), moderately stable broad ranking and a guardrail-compliant sampler simulation. But about 1% extra priority hard-positive exposure, less than 1% source-frequency change (with 4.39% ordered-position changes), unstable top-hard memberships, in-sample/validation-selected-teacher dependence and uncertain annotation boundaries do not justify an additional error-specific full model. If a future revised study pursued it, a matched rarity-only ablation would be necessary to attribute benefit; that model/matrix change is not authorized now.

**SIMPLIFY — Rarity-only:** Preferred. The bounded full-image patient-aware rarity policy already changes intended rare-class exposure substantially while preserving all patients, >=95% distinct coverage in both strata, caps, original geometry and lesion protection. It avoids teacher-derived difficulty/noise and is simpler to implement/audit. Recommend **standard versus patient-aware rarity-only sampling** for a later student-approved training comparison. No AP gain is predicted; this proposal is not an official policy freeze or authorization to launch.

**REDESIGN — Different intervention:** Current simulations do not show both approaches infeasible or uniformly too weak: rarity-only produces meaningful nominal class redistribution under all safeguards. A broader redesign is therefore not required by this evidence. It may be needed if the supervisor requires a stronger distinct research contribution or if a later approved training result fails, but no new loss, architecture or methodology is developed here.

The defensible contribution is an auditable patient-aware fixed-budget data-selection study, a controlled test of incremental model-error guidance, explicit rejection of unverified hard-negative labels and measured limitations. It is not invention of rarity sampling, proof of detection improvement or clinical validation. Formal revised thesis-title/task/scope alignment remains pending. The historical two-checkpoint by three-inference A-F design and prior negative inference evidence remain intact; the recommended possible revision must be discussed and approved rather than silently adopted.

## 9. Verification, preservation and reproduction

All 171 tests pass, including 11 training-boundary/error tests and eight sampler/analysis tests. Tests cover one-to-one category partitioning, .50/.75 difficulty, low confidence, wrong class, empty labels, non-training/path rejection, tied ranks, deterministic subsets, exact alpha0 pairing, lesion protection, infeasible quotas and capacity logic checked against exhaustive small fixtures. Simulations are not training tests.

Independent saved-output verification checks all full/subset predictions for training membership/valid original coordinates, replays 32 saved difficulty cases without GPU inference, and checks every one of 720 saved index plans: exact budget/index range, source caps, all patients, per-stratum quotas and >=95% coverage, all 16 lesion patients' views exactly once, paired budgets/coverage and all eight class/HP exposure totals. All 10,123,920 simulated draws reconcile. No epoch failure is hidden by an average.

All 36 baseline artifacts retain their hashes. Raw/prepared stat inventories match 121,973 entries; frozen official source/protocol, A/B/C proposals and prior source/config/evidence diffs are unchanged. No validation/test image/label contents are opened for this study, no annotation is changed and no official training occurs. New private predictions, IDs, plans, per-patient distributions and all medical figures remain under ignored `outputs/error_guided_training_feasibility/`. Aggregate JSON/CSV contains no private image IDs.

Tooling notes: the first uncertainty parser expected explicit Boolean/0 values; inspecting training-only source values established blank/1 semantics before analysis, without altering source metadata or weights. Initial preservation predates newline normalization of the authorization-text hash; `preservation_pre_gpu.json` binds the final config used by all inference. No scientific parameter changed and no inference was repeated for either bookkeeping correction. Tests use the previously verified workspace temporary directory workaround; no dependencies or frozen source are changed.

Completed runs: `inference/best_01`, `epoch10_01`, `epoch40_01`, and `simulations/matched_01`. Successful results and active pointers are retained. Do not repeat successful GPU mining to regenerate aggregates.

```powershell
.\.venv\Scripts\python.exe -m wrist_mining.verify
.\.venv\Scripts\python.exe -m wrist_mining.preservation after
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import tempfile,pathlib,unittest;tempfile.tempdir=str(pathlib.Path('outputs/phase3c2_test_tmp').resolve());unittest.main(module=None,argv=['unittest','discover','-s','tests','-q'])"
```

Original inference commands were `python -m wrist_mining.inference --checkpoint best.pt --attempt best_01`, and the analogous `epoch010.pt/epoch10_01` and `epoch040.pt/epoch40_01`. Completed inference attempts verify identity and return without GPU work. An incomplete existing attempt refuses overwrite, preserving its evidence. `python -m wrist_mining.analysis analyze` and `stability` recompute CPU aggregates from saved predictions; `qualitative` refuses the existing private figure directory. `python -m wrist_mining.simulate --attempt matched_01` refuses the existing completed plan directory. Use saved results, not unnecessary reruns.

Public evidence includes `training_mining.json`, `size_strata.csv`, `patient_error_distributions.json`, three checkpoint inference receipts, `checkpoint_stability.json`, `qualitative_audit.json`, the mining/sampler predefinitions, `capacity_diagnostics.json`, 720-row `epoch_exposures.csv`, 240-row `selection_overlap.csv`, `ordered_plan_comparison.json`, `simulation_summary.json`, `independent_verification.json`, preservation/test/provenance receipts and `decision.json`. Exact per-seed tails and anonymous audit-case repetition remain inspectable in the aggregate summary.

Meaningful pushed checkpoints: `b6ad105` records tested training-only mining/predefinition before inference; `bc5cb6a` records mined/stability/audit evidence and the exact sampler proposal before simulations. The final verified report/evidence commit follows and is reported with delivery. No force push is used.

**Stop after reporting.** No training, held-out evaluation, historical matrix change or further methodology development is authorized by this recommendation.
