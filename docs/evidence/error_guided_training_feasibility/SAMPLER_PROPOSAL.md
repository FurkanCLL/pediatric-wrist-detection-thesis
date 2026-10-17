# Matched full-image sampler proposal

Recorded after training-only mining/technical review and before the 80-epoch simulations. Simulation permission is separate from official model-training approval. No historical A-F design, A/B/C policy or training protocol is replaced.

## Why test error guidance

The verified in-sample diagnostics identify 3,087 error-eligible images; 1,208 have no existing rarity priority and contain 1,261 hard fracture annotations. Image difficulty and rarity have Spearman correlation .654: related, but not identical. Same-trajectory epoch-10/40 sensitivity gives correlations .742/.721 and top-decile Jaccard .380/.434 on 1,087 predefined training cases. These are moderately stable difficulty patterns with substantial rank/tie instability, not independent truth. The 24-case technical audit finds duplicate, extent, taxonomy and unresolved finding/annotation issues; no clinical negative is certified. Use annotated-positive difficulty only, retaining every source label unchanged.

## Exact three-policy comparison

1. **Standard:** original shuffled full-image population without replacement, exactly 14,061 draws.
2. **Patient-aware rarity-only:** Candidate-B rarity formula and repetition limits, full images only.
3. **PAEG-CAT:** same coverage skeleton, realized bonus budget, caps, full images and random image keys, changing only the bounded priority weight.

For class c in {periostealreaction, pronatorsign, softtissue, boneanomaly}, use the unchanged Candidate-B priority

`r_c = min(2, sqrt(1590 / training_image_support_c) * min(1, training_patient_support_c / 100))`,

with zero priority below 32 independent training patients. Define `R_i=max_c r_c * present_ic / 2` in [0,1]. The max prevents additive co-occurrence amplification. Entire bone-lesion patients have zero bonus priority.

For each source annotation g, use the existing .25 one-to-one match results: `d_g=1` if unmatched at IoU .50; otherwise `.5` if unmatched at .75; otherwise 0. `D_ic` is the mean d over an image's annotations of class c, absent classes zero. `D_i=max_c D_ic` over c in {fracture, periostealreaction, pronatorsign, softtissue, boneanomaly}, with every view of a bone-lesion patient excluded from bonus eligibility. The object-level reasons still distinguish weak confidence, classification, localization and no residual candidate; they are not independent generalization labels.

Use **`W_rarity=R_i` and `W_error=.5*R_i+.5*D_i`**. Equal coefficients are an interpretable feasibility hypothesis between two bounded components, not an optimized optimum or a 50/50 quota of bonus draws. No potential-FP count or diagnosis-uncertainty mark enters either weight. No alternative coefficient, larger budget or new policy will be searched in this phase.

## Shared coverage and exact matching of the intervention budget

Retain the earlier Candidate-B independent patient/status anchors, full rare/empty-label-image protection and conservative same-status coverage/fallback logic. Additionally protect **all original views of all 16 bone-lesion patients**, not only images containing lesion boxes. This cohort receives exactly one draw per source image in every simulated epoch and no bonus.

The requested 5% replacement is capped at `floor(.05*9307)+floor(.05*4754)=702` draws (4.9925%), because rounding to 703 cannot meet both 95% distinct-image stratum thresholds. Use the existing bounded rarity allocator to determine a difficulty-independent shared coverage skeleton and a feasible realized bonus quota per fracture stratum. If rarity capacity is insufficient, restore omitted unique views using the documented existing fallback; report the smaller realized intervention. This is not a promise that all 702 requested slots can be filled.

Reallocate those same realized per-stratum bonus slots under **both** R and W_error. Both use the same sorted source population, same per-source-image uniform `u_i(seed,epoch)` and weighted exponential priority `-log(u_i)/W_i`. Both use the same capacity-feasible ticket allocator, identical skeleton and final permutation stream. An alpha=0 test gives exactly identical paired plans, ruling out a comparison driven just by unrelated random priorities. Maximum one extra ticket per image means at most two draws/image.

Patient extras are capped at `min(3,ceil(.5*N_p))` across both strata, with zero extras for lesion patients. Ceiling can permit a singleton to reach 2x exposure; this is an explicit absolute/ceiling cap, not a strict 50% relative bound. The allocator preserves feasibility of the two remaining stratum quotas at each accepted ticket: individual stratum capacities and their joint patient capacity must suffice. This is a small exact capacity safeguard, not an additional patient weighting heuristic.

Every epoch must have exactly 14,061 draws, original 9,307/4,754 fracture-positive/no-fracture-box draw counts, all 4,264 patients, at least 8,842/4,517 distinct images in those strata, image/patient caps and exact lesion-cohort preservation. Stop on a violation; do not relax a failed guardrail. Both candidates have identical unique source-image sets each epoch, so compare **bonus membership and full draw multiplicities**, not the misleading 100% overlap of their shared coverage set.

## Simulation and interpretation

Simulate exactly these three policies for 80 epochs at seeds 42,1042,2042: 720 nominal epochs and 10,123,920 image draws. Store all source-index plans privately; publish only per-epoch/seed aggregates and hashes. Assess hard-positive exposure including genuinely missed-at-.50 objects separately from localization-only .50-to-.75 cases. Record class/image/patient exposure, source uncertainty marks, potential-FP co-exposure, audit-case repetition, realized replacement fraction, extra-set overlap/total variation and every-epoch guardrail minima.

No image geometry, labels, augmentation, fitting seed, initialization, nominal optimizer budget or official trainer changes. Simulated source exposure is before native random affine/box filtering. More repeated annotations do not add independent patient information. A favorable exposure result cannot establish an AP gain. Recommendation must weigh distinctiveness, rank/annotation reliability and roughly one full baseline-equivalent training cost per future model; final experiments remain a student decision.
