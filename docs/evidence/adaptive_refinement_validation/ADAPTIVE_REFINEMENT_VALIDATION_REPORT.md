# Adaptive Refinement Implementation and Validation

Phase 3C.2, 2026-10-17. **Complete validation/development comparison; negative method-selection outcome.** The student approved CG-ALR v0.1, corrected V2 geometry, numerical defaults, safeguards and decision gates before implementation. No new training or held-out test evaluation occurred.

## 1. Decision and scope

Neither V1 nor V2 meets the primary success or secondary trade-off rules. Both slightly decrease overall mAP50-95 against standard, with paired intervals spanning zero, and both exceed the all-class and fracture false-positive limits. Prefer the preserved standard serial baseline as the current accuracy/cost default. Do not select an adaptive method for thesis C/F from these results alone. A final common adaptive rule remains a separate student decision; the two-checkpoint by three-inference study structure is unchanged.

At confidence .25 / IoU .50, both variants recover 38 annotated objects and lose none, but add 171 false positives: total FP rises 19.34%, and fracture FP rises 16.47%. V2 makes 64 guarded coordinate updates; its very small AP advantage over V1 does not rescue either variant's standard-comparison failures. No parameter adjustment, optimization cycle, class-aware integration/training or test evaluation follows this report.

## 2. Authorization and reproducible identities

The historical proposed file remains byte-for-byte unchanged, including its unapproved flags. The separate approved execution manifest records this student's authorization and binds the proposal, design revision and checkpoint. Its approval applies to this validation study, not official training or final inference selection. Hashes labeled canonical use sorted compact JSON; normalized source hashes use UTF-8 text with normalized line endings.

| Item | Identity |
| --- | --- |
| Historical proposed config canonical SHA-256 | a2e96ff54fdee455aefcb698b8a49008572a22decdb535cc2e65cd6d73997cc6 |
| Approved execution manifest canonical SHA-256 | 713f3d695be8b1c92ced9d11552528c8a1700017dcf730e542dba3fb2738702b |
| Student request normalized SHA-256 | 5ec6a00496f58b345219ac477d5ab39c683114f704f2cf9b0e55e17a6c536104 |
| Design revision | 05f38a78709b2501f70819a5535ad48188663df2 |
| Best checkpoint SHA-256 | 31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b |
| Frozen training protocol SHA-256 | 281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01 |
| Frozen patient split SHA-256 | ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49 |
| Validation global shape-profile canonical SHA-256 | cebc48530898e8c74ac0c355487264a7dbb3ca671e8942a2216531137d43770f |
| Sealed full prediction bundle SHA-256 | f9a48a7181df0192ca9f873fd44c5eebc6034580a232c8e457ab8578629d1575 |
| Full shared-view accuracy run revision | 5d288abdc688abf52916de2a3deb5e9cf15a9f17 |
| Standalone timing run revision | a8002afb102e9a37b59b8ccf68a99faeaa6ce41a |

| Immutable inference source | Normalized SHA-256 |
| --- | --- |
| src/wrist_eval/adaptive.py | 482898228b8d0a9ba6d84a5cf298b12707dafa9c09766e8b11984166dcbeae5e |
| src/wrist_eval/adaptive_config.py | 7357d1fc5d1215f8d4743088b28435da32bfed2a07eb4ef46ad9f0ad3e631a7c |
| src/wrist_eval/adaptive_study.py | 62389e76002c466d1a62827c0c6eecb216f51a08d2a2ed2f90b639de63de077f |

The checkpoint is the saved FP32, unfused epoch-22 EMA from the completed 80-epoch standard baseline. Each run loads one model object and uses it for all global/local views. Algorithm planning/fusion receives no training-condition flag; a future class-aware checkpoint can be injected through the same checked loader without changing the algorithm. That checkpoint does not exist yet and has not been evaluated here.

Validation comprises 3,088 original-coordinate images, 914 patients and 7,204 object annotations. Frozen patient groups, labels, prepared pixels and eight-class IDs remain unchanged. Global dimensions use the same metadata-only rectangular shape profile as preserved standard inference. This profile contains no GT boxes or outcome-selected predictions. A future arbitrary-image service still requires an explicit geometry policy.

## 3. Final implementation

- `src/wrist_eval/adaptive.py`: pure deterministic candidate scoring, context geometry, two-ROI planning, local-evidence filtering, protected one-to-one matching and V1/V2 fusion. It has no checkpoint, filesystem or GT dependency.
- `adaptive_config.py`: validates the approved manifest and unchanged historical proposal. `adaptive_study.py`: checked checkpoint injection, existing Phase 3B adapter, bounded GPU runner, original-coordinate bundles and actual pass accounting.
- `adaptive_analysis.py`: saved-prediction metrics, operating points, recovery/FP correspondence, patient bootstrap and numerical diagnostics. `adaptive_validation.py`: independent output invariants and predeclared decision gates.
- `adaptive_timing.py`: complete standalone variant execution with exact sealed-prediction comparison and synchronized timing. `adaptive_evidence.py`: preservation receipts and private technical overlays.

The settings are exactly the approved defaults: retain the global .001 pool; consider only fracture/periosteal anchors at .05<=p<.25 and effective short side>=1 pixel; score S=.6Q+.4Z with Q=clip((p-.05)/.20) and Z=clip(1-short/reference), references 64/32. Deterministic score/confidence/size/original-row ties and one class head before pooled refill govern priority.

Crop dimensions begin at 75% of each original dimension and expand only to contain the outward-rounded anchor envelope with its two-global-input-pixel margin. Integer placement stays in the original image. Coalesce only at ROI IoU>=.70 with envelope containment; never enlarge a crop union. There are zero, one or two local views, and at most three actual forwards per invocation. Local coordinates use the verified adapter restoration and one original-image offset. Evidence within two resized content pixels of an artificial crop edge is rejected; physical image edges are exempt.

Local target evidence at p>=.25 undergoes local-only same-class NMS at IoU>.70. Same-class global/local matching requires IoU>=.50; eligible mutable anchors and protected global p>=.25 competitors share deterministic greedy one-to-one assignments across both ROIs. Covered anchors require expanded-envelope containment in the evidence's source ROI. Protected competitors cannot change.

V1 changes only matched mutable scores to min(p_original+.10,p_local,.35), rounded down when necessary so float32 never exceeds the exact cap. V2 inherits exactly those matches and scores; local p>=.50 permits 25% coordinate blending. Each proposed float32 box is clipped/validated and vetoed if its overlap with another current same-class box exceeds max(.70, its previous overlap). Geometry is processed by ascending original row ID, without rematching after veto.

All global rows/class IDs remain present; no independent local rows, global deletion or final global NMS is allowed. Unrelated/protected fields and all .50-operating predictions are preserved. Existing duplicates can remain. Final order is stable score/original-ID order when changes occur, with original order retained on a no-change path.

## 4. Verification and bounded GPU pilot

The full pre-GPU suite passed 146 tests, including 36 new algorithm tests. Six later analysis/decision tests bring the final suite to 152 passing tests; package consistency also passes. Coverage includes scoring endpoints/ties, diversity/refill/coalescing, narrow/odd/border geometry, pinned scale-box restoration, artificial/physical edges, class agreement, protected competition, one-to-one reuse prevention, float32 caps, blend/veto, unrelated outputs, empties and pass limits. Adversarial fixtures demonstrate that confirmation can worsen ranking and blending can reduce GT IoU. Synthetic correctness does not establish detection benefit.

The initial full-suite attempt stalled/errored under the default Windows temporary directory. Both logs remain under ignored outputs. The identical suite passed with a workspace temporary directory, without scientific code/protocol changes; the exact environment cause was not isolated. The pre-GPU receipt remains separate from final tests.

The 24 fixed Phase 3B pilot cases all match saved standard global predictions bitwise. ROI counts are 15/5/4 for zero/one/two views; 37 physical model calls, mean 1.5417. Three selected cases replay exactly, costing four more calls. Six fixed cases per variant provide complete-path timing calibration, costing 18 calls. Three synthetic warmup calls bring total pilot GPU work to 62 calls. No per-image invocation exceeds three.

Pilot peak allocated memory is 83,314,176 bytes (0.0776 GiB); reserved is 121,634,816 bytes. Initial composed mean times were 99.075/99.070 ms for V1/V2; they remain diagnostics, superseded by standalone timing below. Load/warmup took 2.509/1.170 s. All technical feasibility gates pass. Six private final-output overlays and two source/crop geometry sheets were actually inspected; source crop bytes and coordinate alignment are verified. The pilot contains one V2 collision veto and no accepted geometry change; synthetic tests cover acceptance, and full validation later exercises 64 accepted updates.

Independent full-bundle accounting passes all 3,088 images: saved-global equality, finite valid outputs, exact global membership/protected fields, one-to-one assignment, confidence caps, .50 invariance, V1 coordinate preservation, V2 collision bounds and equal V1/V2 score/class streams. The single shared accuracy run makes 4,373 physical calls plus three warmup calls; V1/V2 share the identical view bundle and differ only in CPU fusion. No shared GPU cost is divided between reported methods.

## 5. Common original-coordinate AP

All AP values below are fractions, not percentages. The pinned Ultralytics matcher and AP implementation are identical across methods. Preserved standard and SAHI AP dictionaries reproduce exactly from saved predictions. These original-coordinate values retain Phase 3B's documented padding/clipping bridge; they should not be silently substituted for the original trainer's network-coordinate validation scores.

| Method | mAP50 | mAP75 | mAP50-95 | Fracture AP50 | Fracture AP75 | Fracture AP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| Standard serial | 0.59900535 | 0.30227745 | 0.33386896 | 0.92117499 | 0.53459554 | 0.51973720 |
| SAHI Context | 0.52576800 | 0.26977334 | 0.29314538 | 0.86714508 | 0.51119950 | 0.49159838 |
| CG-ALR V1 | 0.59861937 | 0.30234178 | 0.33374752 | 0.92132656 | 0.53482221 | 0.51980528 |
| CG-ALR V2 | 0.59861937 | 0.30236096 | 0.33375909 | 0.92132656 | 0.53497564 | 0.51989787 |

### All eight classes

Each cell is AP50 / AP75 / AP50-95. Exact values and primary operating-point class counts are in `class_metrics.csv`.

| Class | Standard serial | SAHI Context | CG-ALR V1 | CG-ALR V2 |
| --- | --- | --- | --- | --- |
| fracture | 0.921175 / 0.534596 / 0.519737 | 0.867145 / 0.511200 / 0.491598 | 0.921327 / 0.534822 / 0.519805 | 0.921327 / 0.534976 / 0.519898 |
| metal | 0.890409 / 0.559162 / 0.574598 | 0.636297 / 0.383416 / 0.392209 | 0.890475 / 0.559166 / 0.574876 | 0.890475 / 0.559166 / 0.574876 |
| text | 0.988744 / 0.823867 / 0.697748 | 0.972154 / 0.814272 / 0.691608 | 0.988758 / 0.823963 / 0.697783 | 0.988758 / 0.823963 / 0.697783 |
| periostealreaction | 0.565996 / 0.099578 / 0.211656 | 0.447199 / 0.067358 / 0.160685 | 0.563268 / 0.099809 / 0.210497 | 0.563268 / 0.099809 / 0.210497 |
| pronatorsign | 0.571072 / 0.130848 / 0.238662 | 0.538580 / 0.119074 / 0.219206 | 0.570684 / 0.130835 / 0.238579 | 0.570684 / 0.130835 / 0.238579 |
| bonelesion | 0.415000 / 0.195000 / 0.240500 | 0.402895 / 0.198774 / 0.240532 | 0.415000 / 0.195000 / 0.240500 | 0.415000 / 0.195000 / 0.240500 |
| softtissue | 0.296036 / 0.056896 / 0.124503 | 0.257745 / 0.054425 / 0.112047 | 0.295759 / 0.056868 / 0.124389 | 0.295759 / 0.056868 / 0.124389 |
| boneanomaly | 0.143610 / 0.018274 / 0.063548 | 0.084129 / 0.009668 / 0.037277 | 0.143685 / 0.018271 / 0.063551 | 0.143685 / 0.018271 / 0.063551 |

### Explicit paired differences and 95% patient intervals

Each row is method minus comparator. Differences are absolute AP fractions; multiply by 100 for percentage points. Intervals use 500 paired patient draws, seed 314159, with all examinations/views retained together. Eight samples omit overall eight-class mAP because the rare bone-lesion class is absent: overall intervals have 492 valid replicates; fracture/operating intervals have 500. Class absence is undefined, never zero AP. All per-class/operating intervals remain in `patient_bootstrap.json`.

| Comparison | Metric | Point delta | 95% interval |
| --- | --- | --- | --- |
| V1_minus_standard_serial | map50 | -0.00038598 | [-0.000926, +0.000258] |
| V1_minus_standard_serial | map75 | +0.00006433 | [-0.000280, +0.000328] |
| V1_minus_standard_serial | map50_95 | -0.00012144 | [-0.000355, +0.000074] |
| V2_minus_standard_serial | map50 | -0.00038598 | [-0.000926, +0.000258] |
| V2_minus_standard_serial | map75 | +0.00008351 | [-0.000260, +0.000335] |
| V2_minus_standard_serial | map50_95 | -0.00010987 | [-0.000354, +0.000087] |
| V1_minus_sahi_context | map50 | +0.07285137 | [+0.053757, +0.087863] |
| V1_minus_sahi_context | map75 | +0.03256844 | [+0.016939, +0.047372] |
| V1_minus_sahi_context | map50_95 | +0.04060214 | [+0.031157, +0.048139] |
| V2_minus_sahi_context | map50 | +0.07285137 | [+0.053757, +0.087863] |
| V2_minus_sahi_context | map75 | +0.03258762 | [+0.016944, +0.047376] |
| V2_minus_sahi_context | map50_95 | +0.04061371 | [+0.031163, +0.048143] |
| V2_minus_V1 | map50 | +0.00000000 | [+0.000000, +0.000000] |
| V2_minus_V1 | map75 | +0.00001918 | [+0.000000, +0.000059] |
| V2_minus_V1 | map50_95 | +0.00001157 | [+0.000001, +0.000031] |

| Comparison | Fracture metric | Point delta | 95% interval |
| --- | --- | --- | --- |
| V1_minus_standard_serial | ap50 | +0.00015157 | [-0.000073, +0.000639] |
| V1_minus_standard_serial | ap75 | +0.00022668 | [-0.000234, +0.000463] |
| V1_minus_standard_serial | ap50_95 | +0.00006808 | [-0.000114, +0.000357] |
| V2_minus_standard_serial | ap50 | +0.00015157 | [-0.000073, +0.000639] |
| V2_minus_standard_serial | ap75 | +0.00038010 | [-0.000160, +0.000659] |
| V2_minus_standard_serial | ap50_95 | +0.00016068 | [-0.000084, +0.000483] |
| V1_minus_sahi_context | ap50 | +0.05418147 | [+0.047279, +0.060743] |
| V1_minus_sahi_context | ap75 | +0.02362271 | [+0.009230, +0.037572] |
| V1_minus_sahi_context | ap50_95 | +0.02820690 | [+0.022796, +0.033864] |
| V2_minus_sahi_context | ap50 | +0.05418147 | [+0.047279, +0.060743] |
| V2_minus_sahi_context | ap75 | +0.02377614 | [+0.009280, +0.037611] |
| V2_minus_sahi_context | ap50_95 | +0.02829949 | [+0.022844, +0.033886] |
| V2_minus_V1 | ap50 | +0.00000000 | [+0.000000, +0.000000] |
| V2_minus_V1 | ap75 | +0.00015343 | [+0.000000, +0.000471] |
| V2_minus_V1 | ap50_95 | +0.00009260 | [+0.000007, +0.000247] |

V1/V2 beat the preserved SAHI Context configuration, but neither establishes benefit over standard. V2-minus-V1 mAP50-95 has a positive paired lower bound, yet the point difference is only +0.00001157 (0.001157 percentage points), and both fail the standard-comparison rules. This is insufficient to select V2 under the approved conditional rule.

**Confidence-tie limitation:** non-target boxes/scores/class streams and native TP indicators are bitwise unchanged, but some displayed non-target AP values drift slightly. Source inspection found the pinned `ap_per_class` uses a full-stream `np.argsort(-conf)` with no stable tie rule. Changing target scores/order changes how equal-score non-target TP/FP rows are ordered before class filtering. `confidence_tie_diagnostics.json` verifies equal non-target TP/confidence streams and identical class-isolated AP, while exposing changed TP ordering after that global sort. Keep the frozen common evaluator; do not interpret this numerical drift as detection improvement. Tiny AP differences and bootstrap limits remain conditional on these pinned conventions.

## 6. Operating points and recovered/lost objects

The descriptive matcher is confidence-ordered, class-aware and one-to-one, distinct from native AP matching. All-class precision/recall below use pooled TP/FP/FN; fracture P/R use that class's counts. FP is annotation-relative, not clinical adjudication.

| Method | Conf/IoU | TP | FP | FN | P | R | FP/image | Fracture P | Fracture R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Standard serial | 0.05/0.50 | 6777 | 4351 | 427 | 0.6090 | 0.9407 | 1.4090 | 0.7104 | 0.9199 |
| Standard serial | 0.05/0.75 | 5200 | 5928 | 2004 | 0.4673 | 0.7218 | 1.9197 | 0.5003 | 0.6478 |
| Standard serial | 0.25/0.50 | 6467 | 884 | 737 | 0.8797 | 0.8977 | 0.2863 | 0.8768 | 0.8771 |
| Standard serial | 0.25/0.75 | 5002 | 2349 | 2202 | 0.6805 | 0.6943 | 0.7607 | 0.6180 | 0.6182 |
| Standard serial | 0.50/0.50 | 5942 | 249 | 1262 | 0.9598 | 0.8248 | 0.0806 | 0.9403 | 0.8130 |
| Standard serial | 0.50/0.75 | 4773 | 1418 | 2431 | 0.7710 | 0.6625 | 0.4592 | 0.6746 | 0.5833 |
| SAHI Context | 0.05/0.50 | 6875 | 11039 | 329 | 0.3838 | 0.9543 | 3.5748 | 0.3924 | 0.9412 |
| SAHI Context | 0.05/0.75 | 5303 | 12611 | 1901 | 0.2960 | 0.7361 | 4.0839 | 0.2811 | 0.6741 |
| SAHI Context | 0.25/0.50 | 6599 | 3209 | 605 | 0.6728 | 0.9160 | 1.0392 | 0.6116 | 0.9046 |
| SAHI Context | 0.25/0.75 | 5093 | 4715 | 2111 | 0.5193 | 0.7070 | 1.5269 | 0.4327 | 0.6400 |
| SAHI Context | 0.50/0.50 | 6123 | 1355 | 1081 | 0.8188 | 0.8499 | 0.4388 | 0.7480 | 0.8469 |
| SAHI Context | 0.50/0.75 | 4878 | 2600 | 2326 | 0.6523 | 0.6771 | 0.8420 | 0.5344 | 0.6051 |
| CG-ALR V1 | 0.05/0.50 | 6777 | 4351 | 427 | 0.6090 | 0.9407 | 1.4090 | 0.7104 | 0.9199 |
| CG-ALR V1 | 0.05/0.75 | 5200 | 5928 | 2004 | 0.4673 | 0.7218 | 1.9197 | 0.5003 | 0.6478 |
| CG-ALR V1 | 0.25/0.50 | 6505 | 1055 | 699 | 0.8604 | 0.9030 | 0.3416 | 0.8601 | 0.8825 |
| CG-ALR V1 | 0.25/0.75 | 5025 | 2535 | 2179 | 0.6647 | 0.6975 | 0.8209 | 0.6074 | 0.6232 |
| CG-ALR V1 | 0.50/0.50 | 5942 | 249 | 1262 | 0.9598 | 0.8248 | 0.0806 | 0.9403 | 0.8130 |
| CG-ALR V1 | 0.50/0.75 | 4773 | 1418 | 2431 | 0.7710 | 0.6625 | 0.4592 | 0.6746 | 0.5833 |
| CG-ALR V2 | 0.05/0.50 | 6777 | 4351 | 427 | 0.6090 | 0.9407 | 1.4090 | 0.7104 | 0.9199 |
| CG-ALR V2 | 0.05/0.75 | 5201 | 5927 | 2003 | 0.4674 | 0.7220 | 1.9194 | 0.5006 | 0.6481 |
| CG-ALR V2 | 0.25/0.50 | 6505 | 1055 | 699 | 0.8604 | 0.9030 | 0.3416 | 0.8601 | 0.8825 |
| CG-ALR V2 | 0.25/0.75 | 5025 | 2535 | 2179 | 0.6647 | 0.6975 | 0.8209 | 0.6074 | 0.6232 |
| CG-ALR V2 | 0.50/0.50 | 5942 | 249 | 1262 | 0.9598 | 0.8248 | 0.0806 | 0.9403 | 0.8130 |
| CG-ALR V2 | 0.50/0.75 | 4773 | 1418 | 2431 | 0.7710 | 0.6625 | 0.4592 | 0.6746 | 0.5833 |

| Comparison at .25 | IoU | Both | Recovered | Lost | Neither |
| --- | --- | --- | --- | --- | --- |
| V1_minus_standard_serial | 0.5 | 6467 | 38 | 0 | 699 |
| V1_minus_standard_serial | 0.75 | 5002 | 23 | 0 | 2179 |
| V2_minus_standard_serial | 0.5 | 6467 | 38 | 0 | 699 |
| V2_minus_standard_serial | 0.75 | 5002 | 23 | 0 | 2179 |
| V1_minus_sahi_context | 0.5 | 6477 | 28 | 122 | 577 |
| V1_minus_sahi_context | 0.75 | 4804 | 221 | 289 | 1890 |
| V2_minus_sahi_context | 0.5 | 6477 | 28 | 122 | 577 |
| V2_minus_sahi_context | 0.75 | 4804 | 221 | 289 | 1890 |
| V2_minus_V1 | 0.5 | 6505 | 0 | 0 | 699 |
| V2_minus_V1 | 0.75 | 5025 | 0 | 0 | 2179 |

Against standard at IoU .50, recovery is 15 fracture and 23 periosteal objects, with zero lost; at IoU .75 it is 14 fracture and nine periosteal objects, also zero lost. V1 and V2 recover exactly the same objects at both predefined thresholds. Relative to SAHI Context they recover 28 but lose 122 objects overall at IoU .50, including seven fracture recovered/69 lost and 15 periosteal recovered/30 lost. Thus improved AP over SAHI does not mean superior recall on every target.

Every standard detection already above .25 is protected. Both-detected localization IoU deltas against standard have mean, median, p05 and p95 exactly zero for both target classes at IoU .50 and .75. V2 changes only originally weak anchors; its accepted geometry updates are analyzed separately below. At .50 confidence all outputs are exactly baseline-invariant. No non-target operating counts change against standard.

## 7. False positives, ranking and actual refinement

There are 459 confidence updates in 360 images (11.66% of all validation images). Of these, 250 start below .15 and cannot reach .25 under the approved +.10 limit. The other 209 cross .25: 38 become descriptive TPs and 171 become FPs. Only 18.18% of threshold promotions are annotated matches. No new detection rows are created: these are existing weak rows promoted in confidence. Local confirmation therefore fails to discriminate enough true from false anchors.

At the primary point, all-class TP/FP/FN changes from 6,467/884/737 to 6,505/1,055/699. Fracture changes from 2,463/346/345 to 2,478/403/330; periosteal from 298/295/195 to 321/409/172. All-class FP +171 exceeds the integer limit 972; fracture FP +57 exceeds limit 380. Periosteal FP rises 38.64%. Small target recall gains coexist with lower precision and worse periosteal AP50/AP50-95.

| Comparison | 95% interval for FP count ratio | 95% interval for fracture FP count ratio |
| --- | --- | --- |
| V1_minus_standard_serial | [+1.159041, +1.227546] | [+1.122135, +1.212895] |
| V2_minus_standard_serial | [+1.159041, +1.227546] | [+1.122135, +1.212895] |
| V1_minus_sahi_context | [+0.308577, +0.349925] | [+0.223054, +0.276183] |
| V2_minus_sahi_context | [+0.308577, +0.349925] | [+0.223054, +0.276183] |
| V2_minus_V1 | [+1.000000, +1.000000] | [+1.000000, +1.000000] |

Both adaptive variants have total FP ratio 1.19344 and fracture FP ratio 1.16474 versus standard; both lower uncertainty bounds exceed 1.10. This fails the tolerance even accounting for patient uncertainty. The six non-target classes retain exact prediction fields and within-class order, as independently checked; apparent AP drift is the tie-order limitation above.

| ROI / evidence stage | Count |
| --- | --- |
| Zero / one / two local-ROI images | 2,074 / 743 / 271 (67.16% / 24.06% / 8.78%) |
| Images with any local ROI | 1,014 (32.84%) |
| Mean model calls per image | 1.41612694 |
| Eligible fracture / periosteal anchors | 827 / 2,046 = 2,873 |
| Excluded: non-target / outside confidence band | 14,650 / 25,272 |
| Selected ROI seeds / coalesced anchors / budget drops | 1,285 / 1,512 / 76 |
| Rejected local rows: low score / non-target / artificial boundary | 27,923 / 5,229 / 85 |
| Eligible local rows / suppressed in union / retained | 2,665 / 658 / 2,007 |
| Matched protected / mutable / unmatched retained evidence | 1,448 / 459 / 100 |
| Refined images without any final update | 654 / 1,014 (64.50%) |
| V2 accepted geometry / collision veto | 64 / 18 |

No actual candidate failed the effective-size/no-zoom geometry rules. Class priority and coalescing concentrate 2,873 anchors into 1,285 local calls. Most local evidence is rejected by confidence/class rules, while 1,448 of 1,907 matches are consumed by protected detections. Only 459 mutable matches update scores (15.98% of eligible anchors). The protection rules work as intended, but leave limited mutable evidence and cannot create a missing anchor.

V2's 64 accepted geometry changes occur in 64 images (2.07%); 18 attempts are collision-vetoed without rematching. This limited footprint explains the very small V2/V1 AP difference. Descriptive geometry diagnostics compare each changed box's best same-class annotated IoU, counting zero if none exists; these are not one-to-one TP claims and never guide inference.

| Geometry diagnostic | Value |
| --- | --- |
| equal | 18 |
| improved | 29 |
| updated_score_below_0.25 | 35 |
| worsened | 17 |

| Class | Changes | Mean IoU delta | Median | p05 | p95 | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 56 | +0.005415 | +0.000000 | -0.016215 | +0.035890 | -0.034757 | +0.084495 |
| periostealreaction | 8 | +0.014867 | +0.019789 | -0.006596 | +0.036245 | -0.007587 | +0.039190 |

Of the 64 blends, best annotated IoU improves for 29, worsens for 17 and is unchanged for 18; 35 updated boxes remain below .25 confidence. These changes do not alter primary-point recovered/lost counts between V1 and V2.

## 8. Measured inference cost and GPU memory

Hardware/software match preserved Phase 3B: RTX 4050 Laptop GPU, 6 GiB, driver 560.94, CUDA 12.6; Python 3.12.5, torch 2.14.1+cu126, torchvision 0.29.1+cu126, Ultralytics 8.4.177, SAHI 0.12.8, NumPy 2.3.5. Prepared grayscale is replicated to three equal channels; model weights remain FP32, with normalized float16 inputs and CUDA float16 autocast. No augmentation or training is used.

**Timing fallback and deviation:** the predefined pilot measured complete paths before AP, but the large discrepancy was recognized during final review after the accuracy analysis. On the same six pilot cases, composed versus standalone means were 115.752 versus 70.035 ms (V1) and 115.632 versus 65.439 ms (V2), differences of about 65%/77%. Warm shape/cache/device-state effects are plausible; contributors were not isolated. The stage-composition validation should have been resolved earlier. Section 16's preapproved fallback was committed at `a8002af` before the additional full timing run. Original timing evidence is retained, and accuracy outputs/configuration were already sealed.

The authoritative adaptive values below measure **complete single-variant paths on all 3,088 images**, starting before decode and ending after fusion/sort/cap assertions and CUDA synchronization. Each image's adaptive outputs, global outputs, ROI plan and call count match the sealed accuracy bundle exactly. This is timing/replay evidence, not a new accuracy trial or variant. No shared GPU work is divided. Model load, three synthetic warmups, JSON export and post-hoc identity/GT/AP checks are excluded from per-image end-to-end and reported separately.

For the paired <=2x cost gate, decode plus the already-required global call is also timed through valid original-coordinate predictions, before ROI planning. It is a contemporaneous one-pass control without an extra forward. This control shares the adaptive run's device history and metadata setup; it is not a separate randomized standard benchmark. Historical standalone standard/SAHI values remain descriptive cross-session comparators; desktop state and caching limit speedup claims. V1 then V2 were measured in fixed order, so small timing differences are not evidence of algorithmic speed differences.

| Method | Mean ms | Median ms | p95 ms | p99 ms | Images/s (1/mean) |
| --- | --- | --- | --- | --- | --- |
| Standard serial (historical) | 62.334 | 61.769 | 77.481 | 85.603 | 16.04 |
| SAHI Context (historical) | 238.129 | 237.725 | 300.102 | 338.180 | 4.20 |
| CG-ALR V1 (standalone) | 58.452 | 54.773 | 125.732 | 156.251 | 17.11 |
| CG-ALR V2 (standalone) | 60.844 | 55.446 | 132.547 | 159.505 | 16.44 |

| Variant | Paired global mean ms | Extra mean ms | Paired ratio | Historical standard ratio | Load s | Warmup s | Loop incl export s | Peak allocated / reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | 44.776 | 13.676 | 1.305 | 0.938 | 2.879 | 0.870 | 182.399 | 79.11 / 114.00 |
| V2 | 46.035 | 14.809 | 1.322 | 0.976 | 1.005 | 0.111 | 189.868 | 122.87 / 154.00 |

| Variant mean stage ms | Decode | Planning | Preprocess all views | GPU forward (events) | View NMS/restoration | Local evidence/matching/fusion |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | 11.762 | 0.272 | 5.026 | 33.226 | 5.338 | 0.806 |
| V2 | 11.931 | 0.293 | 5.158 | 34.974 | 5.476 | 0.892 |

Stage counters have distinct wall/CUDA-event boundaries; their sums need not equal whole-path wall time. Python/SAHI orchestration and crop/view handling are included in whole-path time. Peak allocated/reserved are PyTorch tensor/allocator measurements, not total process/driver VRAM. No OOM or nonfinite output occurs. Both remain below the approved 4 GiB allocated-memory ceiling. V2 was measured second in the same process and has a higher retained allocation than V1; this is reported conservatively as observed session memory, not intrinsic extra geometry memory. Adapter/model reference retention and allocator history are plausible contributors; no additional GPU rerun was made to isolate them. The initial shared full run took 216.807 s including export, with peak allocated/reserved 82,949,632 / 119,537,664 bytes; composed means 68.015/67.987 ms remain diagnostic.

Standalone V1 and V2 each make 4,373 physical calls plus three warmups: 8,746 measured calls and six warmups total. Including the pilot (62 calls) and shared full run (4,376 including warmup), the entire phase uses 13,190 actual model calls. Every individual adaptive invocation still respects its maximum of three; sharing the accuracy bundle reduces experimental compute, not reported per-method cost.

## 9. Predeclared decision gates

Section 17 rules are applied without threshold revisions. The fracture uncertainty tolerance is -.005 absolute AP, the FP point limit is 1.10 and the complete-path mean-cost limit is 2x paired standard. These are research engineering gates, not clinical noninferiority or safety thresholds.

| Gate | V1 | V2 |
| --- | --- | --- |
| all_class_fp_ratio | FAIL | FAIL |
| fracture_ap_points_preserved | PASS | PASS |
| fracture_fp_ratio | FAIL | FAIL |
| fracture_interval_tolerance | PASS | PASS |
| mean_latency_ratio | PASS | PASS |
| overall_map50_95_interval_positive | FAIL | FAIL |
| overall_map50_95_positive | FAIL | FAIL |
| overall_map50_nondecreasing | FAIL | FAIL |

| Outcome | V1 | V2 |
| --- | --- | --- |
| primary_success | False | False |
| secondary_tradeoff_success | False | False |
| passes | False | False |

Technical validity passes. Fracture AP point estimates and the predeclared uncertainty tolerance pass; that does not establish a clinically meaningful gain. Overall AP benefit, nondecreasing mAP50 and both FP limits fail. The secondary rule also fails because overall AP point change is negative and FP preservation fails. The conditional V1/V2 selection rule therefore does not activate, even though V2 has a tiny positive paired difference versus V1. Recommended qualifying adaptive variant: **none**.

## 10. Technical qualitative review

All 13 private figures were actually inspected: six pilot prediction triptychs, two pilot source/crop sheets and five full-validation triptychs. Full categories are first unused frozen-order examples of V2 accepted geometry, collision veto, recovered target, promoted FP and ROI without update; there is no outcome-selected replacement. No lost-target example exists at the primary point, consistent with zero lost targets. Receipts remain aggregate; radiographs, identifiers and detailed coordinates remain Git-ignored.

The recovered example shows a weak fracture crossing .25 at roughly .29 confidence with IoU about .79, retaining V1 geometry. The promoted-FP example adds periosteal evidence near the shaft while existing findings remain aligned. The no-update example spends local calls without recovering an annotated fracture miss. Protected/veto examples preserve global output geometry. The selected accepted-V2 example's changed low-score box is below the fixed .25 display threshold; that triptych alone cannot prove the blend, so the numeric geometry diagnostics and synthetic tests supply that evidence. Source/crop sheets show expected pixel/context geometry. None of this is a clinician review or certification that all context is medically sufficient.

## 11. Preservation, limitations and stop

Final integrity checks preserve all 36 baseline artifacts and the existing standard/SAHI prediction hashes, the frozen training protocol, original proposed adaptive config and class-aware A/B/C candidates. Raw/prepared stat inventories match all 121,973 entries; all 3,088 validation image hashes and 3,088 validation label hashes match. Protected pre-phase source/config/evidence diffs are empty and normalized Phase 3B source hashes match. Raw/prepared files were not modified, and held-out image/label contents were not opened; metadata/stat inventory checking is distinct from test evaluation. No old repository work occurs. No dependency is added.

This is a single-checkpoint validation study using a validation-selected model and validation-informed method design. Patient bootstrap conditions on this checkpoint/data/metric convention, not training-seed variation or unseen-hospital performance; it does not correct multiple comparisons or validation reuse. Bone-lesion support is only five boxes across four validation patients, and absent-class resamples limit overall intervals. Annotation-relative false positives may include uncertain/unlabeled findings; they are not clinically adjudicated.

The anchored design cannot create a target without a weak global candidate. Broad 75% crops offer modest extra scale, protected competition consumes most evidence, and conservative caps leave many candidates below the operating threshold. Protected rows guarantee preservation but cannot repair existing confident errors. Local score confirmation is not calibrated independent evidence: the same checkpoint repeats correlated errors. V2's collision guard controls overlaps, not GT accuracy, and actual geometry changes are rare. The numerical tie limitation and late timing-fallback recognition are explicit; the complete-path boundary follows approved Section 16 and the common accuracy evaluator remains unchanged.

Stop at this completed two-variant comparison. Preserve this negative evidence. Student review is needed before any new methodological adjustment, class-aware integration/training, final common adaptive-method selection or held-out evaluation. The revised institutional thesis scope still awaits formal supervisor alignment.

## 12. Evidence and reproduction

Tracked aggregate receipts in this directory:

- `automated_tests.json`, `final_tests.json`, `pilot_summary.json`, `pilot_gate.json`, `pilot_qualitative.json`, `pilot_geometry.json`.
- `full_summary.json`, `independent_accounting.json`, `full_qualitative.json`, `utilization_and_ranking.json`, `geometry_diagnostics.json`, `confidence_tie_diagnostics.json`.
- `metrics_comparisons.json`, `class_metrics.csv`, `patient_bootstrap.json`, `fp_ratio_uncertainty.json`, `decision_gates.json`.
- `timing_protocol.json`, `standalone_timing.json`, `preservation_before.json`, `preservation_after.json`, `preservation_final.json`, `publication_tests.json` and `publication_verification.json`.

Private directories are `outputs/adaptive_refinement/pilot/pilot_01`, `full/full_01`, `standalone_timing/timing_01` and `analysis`; the active pointers and hashes bind their preserved files. Interrupted test logs and every successful pilot/full/timing result are retained. Existing attempt names refuse overwrite. Do not repeat successful GPU work to regenerate this report.

Offline rechecks (no GPU inference):

```powershell
.\.venv\Scripts\python.exe -c "import tempfile,pathlib,unittest;tempfile.tempdir=str(pathlib.Path('outputs/phase3c2_test_tmp').resolve());unittest.main(module=None,argv=['unittest','discover','-s','tests','-q'])"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_validation accounting
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis fp_uncertainty
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis geometry
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis ties
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_validation decisions
```

`adaptive_analysis analyze` and `bootstrap` reproduce accuracy aggregates/intervals from private saved predictions, without inference, but recompute already-completed CPU work. For a deliberate, separately warranted GPU reproduction, original commands were `adaptive_study pilot --attempt pilot_01`, `adaptive_study full --attempt full_01` and `adaptive_timing --attempt timing_01`, all under `python -m wrist_eval.`; they refuse those existing attempt directories. Pilot/source/approval gates precede full validation. No command here authorizes further algorithm experiments.

Meaningful pushed checkpoints: `c397328` implements/tests approved core and approval manifest; `5d288ab` verifies the GPU pilot/crop geometry; `a8002af` seals full accuracy/uncertainty/accounting and commits the standalone timing fallback before its execution. The final documentation/provenance commit follows these checkpoints and is reported with delivery. No force push is used.
