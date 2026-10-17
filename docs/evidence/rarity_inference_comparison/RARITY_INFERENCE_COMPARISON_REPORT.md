# Phase 4E: Rarity-Only SAHI Context and CG-ALR V2 Evaluation

Completed validation study, 17–18 October 2026. Both implementations pass technical verification. **Neither method demonstrates an improvement meeting the predefined scientific criteria with the rarity-only checkpoint.** E substantially reduces overall and fracture AP. F produces a very small, uncertain overall AP change and improves fracture AP slightly, while exceeding both frozen FP limits. These are valid negative/mixed findings, not implementation failures.

This report completes the requested bounded six-condition validation comparison. It does not select a final method, develop CG-ALR V3, perform Phase 5A synthesis, launch training or evaluate held-out images/labels/predictions. The student reviews these results before any next phase. Formal supervisor alignment of the revised thesis scope remains pending.

## 1. Objective, frozen inputs and identities

Compare E versus D, F versus D, E versus B and F versus C, plus the two prespecified interactions. A/B/C/D inference is reused from preserved artifacts; their common metrics are recomputed rather than copied from instructions. F makes its own required global call and compares it to saved D; it does not substitute D rows for a mismatched model output.

Population: **3,088 radiographs, 914 patients, 7,204 annotated objects, eight classes**, unchanged original-coordinate annotations and frozen validation membership. All views/examinations remain linked to their patients. Raw 16-bit sources and derived images/labels are preserved. Images are decoded without validation augmentation. Empty boxes do not establish a healthy radiograph. Targets include non-fracture findings and image artifacts.

Implementation was tested and committed/pushed **before GPU execution** at `2d9174864972714f0f313968362ad24995640983`. Starting revision was `323f221ae6978b9bea0e7398c976642e62e48036`. The [pre-inference contract](IMPLEMENTATION_PROTOCOL.md), [test gate](automated_tests.json), [preflight receipt](preflight.json) and additive E/F manifests preserve the authorization and numerical identities.

| Artifact | SHA-256 |
| --- | --- |
| Rarity epoch-26 best checkpoint | cbac08863558a1f96b79bda26c198d2f27c853ff2019678c6104ebd8d0108b1e |
| Standard epoch-22 best checkpoint | 31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b |
| Saved D predictions | b653b3d5e1df5ca227b29588e6085bb115d8530c99e934495d6e1d630a2b661d |
| Historical SAHI/common config (canonical) | 86ed8fd6450b77734e6cdea4eff5dec5eed59d5f0ea9790f0e37c507f9336c13 |
| Historical adaptive proposal (canonical) | a2e96ff54fdee455aefcb698b8a49008572a22decdb535cc2e65cd6d73997cc6 |
| Historical adaptive execution (canonical) | 713f3d695be8b1c92ced9d11552528c8a1700017dcf730e542dba3fb2738702b |
| New E execution (canonical) | 9aa7a854cd2b38a43d3072bf64541b0a1170f655f3ef780b04ad02ae3dea716e |
| New F execution (canonical) | e20a420fef6249945aeeb98c992acbcc29ccde675ac8fb8a6e3456ad388a1c42 |

Historical configs/manifests still point to their original checkpoint. The original proposal's historical unapproved flags remain unchanged; separate execution manifests bind approved numerical methods to rarity. The frozen common training protocol remains `281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01`. Checkpoint hash, sidecar, epoch, initialization/protocol/condition provenance, architecture, eight-class mapping, finite tensors and FP32 unfused EMA compatibility pass; the EMA contains 57 BatchNorm modules. One identical rarity model object supplies all global/local calls and remains unchanged afterward.

## 2. E: unchanged SAHI Context

Reuse the historical `sliced.predict` implementation: input 768, preserved aspect ratio and rectangular global geometry, slice fractions .667/.667, overlap .20/.20, four context slices plus global, sequential batch one, FP32 unfused EMA with FP16 input/autocast, confidence .001, per-view NMS IoU .70, class-aware SAHI union NMS IoU .70 and max 300 final detections. No NMM, Detail configuration, alternate threshold or merging policy is introduced.

The historical outcome-independent 24-image pilot passes with 120 calls, exact replay of its first 3 images, valid finite original-coordinate output and feasibility bounds. No pilot AP is calculated. The full run passes on 3,088 images with **15,440 calls**. Every image has exactly 4 slices/5 calls, all 24 pilot predictions reproduce exactly, and model state is unchanged. Detailed aggregate pilot/full receipts are [pilot E](pilot_E.json) and [full E](full_E.json).

## 3. F: unchanged CG-ALR V2

Inject rarity into the existing `ViewRunner` and unchanged `adaptive.py`. The pure algorithm receives no training-condition flag or ground truth. Preserve the .6Q+.4Z score, fracture/periosteal candidates with .05<=confidence<.25, reference short sides 64/32, original ranking/diversity/coalescing and .75 crop geometry. Local evidence retains its .25 confidence, artificial-edge, union-NMS and same-class IoU>=.50 matching rules. Confidence increases remain bounded by original+.10, local score and .35. V2 uses .25 local coordinate weight only with local confidence>=.50 and the original collision veto. At most 2 ROIs/3 actual calls are allowed. No global row is added or deleted; protected rows/classes remain unchanged.

The same 24-image pilot passes with 39 calls and exact first 3 replay. Full F passes on 3,088 images with **4,303 calls**. Every global float32 array has identical shape and bytes to D, checked immediately before local calls and again from saved bundles. All 24 pilot outputs reproduce exactly. Every saved bundle passes independent membership, coordinate restoration, score-cap, protected-output, one-to-one evidence, quarter-blend and collision checks plus exact replay through frozen planning/fusion. Confidence-.50 output is unchanged from D. See [pilot F](pilot_F.json), [full F](full_F.json) and [verification](verification.json).

Full usage: 0/1/2 ROIs on **2,121/719/248 images**; 1,215 local views, mean **1.393459 calls/image**. Of 2,004 eligible local-evidence rows, 1,883 match a global row: 429 mutable matches produce 429 confidence updates; 1,454 protected matches leave their rows unchanged. 121 evidence rows remain unassigned. V2 accepts 90 geometry changes and vetoes 26 collisions. Eligible evidence and matched protected evidence must not be counted as additional recovered detections.

## 4. Common original-coordinate AP

AP uses the unchanged pinned framework matcher and 101-point interpolated integration at ten IoUs .50:.05:.95. Overall AP is the macro mean of the same eight classes. Values below are fractions in [0,1]; differences later use percentage points (pp), 100 times the fraction difference. These are common-evaluator validation results, separate from historical native training fitness.

| ID | Training / inference | mAP50 | mAP75 | mAP50-95 |
| --- | --- | --- | --- | --- |
| A | standard / standard | 0.599005 | 0.302277 | 0.333869 |
| B | standard / SAHI Context | 0.525768 | 0.269773 | 0.293145 |
| C | standard / CG-ALR V2 | 0.598619 | 0.302361 | 0.333759 |
| D | rarity-only / standard | 0.611845 | 0.315562 | 0.337551 |
| E | rarity-only / SAHI Context | 0.517918 | 0.271754 | 0.288736 |
| F | rarity-only / CG-ALR V2 | 0.612083 | 0.315752 | 0.337693 |

Per-class AP50-95 and validation support:

| Class | Objects / patients | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 2808 / 538 | 0.519737 | 0.491598 | 0.519898 | 0.512034 | 0.476008 | 0.512379 |
| metal | 96 / 24 | 0.574598 | 0.392209 | 0.574876 | 0.586509 | 0.392489 | 0.586579 |
| text | 3611 / 913 | 0.697748 | 0.691608 | 0.697783 | 0.706043 | 0.688589 | 0.706043 |
| periostealreaction | 493 / 163 | 0.211656 | 0.160685 | 0.210497 | 0.233529 | 0.161600 | 0.233527 |
| pronatorsign | 83 / 83 | 0.238662 | 0.219206 | 0.238579 | 0.240767 | 0.213970 | 0.240834 |
| bonelesion | 5 / 4 | 0.240500 | 0.240532 | 0.240500 | 0.202000 | 0.194109 | 0.202000 |
| softtissue | 72 / 61 | 0.124503 | 0.112047 | 0.124389 | 0.159866 | 0.156686 | 0.160523 |
| boneanomaly | 36 / 14 | 0.063548 | 0.037277 | 0.063551 | 0.059663 | 0.026434 | 0.059663 |

E/F AP50 and AP75; all three AP metrics for all 48 class/condition combinations are in [per-class CSV](per_class_AP.csv).

| Class | E AP50 | E AP75 | F AP50 | F AP75 |
| --- | --- | --- | --- | --- |
| fracture | 0.854545 | 0.467626 | 0.916618 | 0.525911 |
| metal | 0.641599 | 0.385000 | 0.905953 | 0.610374 |
| text | 0.966628 | 0.809844 | 0.989615 | 0.830542 |
| periostealreaction | 0.434657 | 0.074143 | 0.585385 | 0.140240 |
| pronatorsign | 0.474409 | 0.100097 | 0.548654 | 0.097071 |
| bonelesion | 0.371467 | 0.197139 | 0.413182 | 0.195000 |
| softtissue | 0.316226 | 0.133219 | 0.339522 | 0.110636 |
| boneanomaly | 0.083814 | 0.006960 | 0.197735 | 0.016244 |

### Verified equal-confidence ordering limitation

F protects six non-candidate classes, yet tiny AP changes occur for some of them. The [saved-output trace](ap_tie_diagnostic.json) and [diagnostic script](check_ap_ties.py) verify that every per-image prediction row and native TP flag for these classes is identical to D. The pinned `ap_per_class` sorts the concatenated population using `np.argsort(-conf)` without stable tie-breaking. F's global row reordering changes permutations within identical-confidence groups, including TP/FP ordering, while sorted confidences remain bitwise identical. For example, soft-tissue AP50-95 changes by +0.000656790 despite identical soft-tissue outputs/TP flags. This is an evaluator ordering effect, not a detected soft-tissue improvement.

The frozen evaluator, predictions, historical results and main reported AP/intervals are preserved. No alternate/corrected AP is calculated and no metric policy is silently changed. Interpret F's tiny overall point gain with this additional limitation; patient bootstrap does not remove the tie-order sensitivity.

## 5. Patient-paired uncertainty and required comparisons

Use 500 replicates, seed 314159, resampling 914 patients with replacement and retaining all views/examinations. Identical indices apply to all six methods. AP is recomputed within each sample. All 2,000 historical A/B/C/D AP replicates reproduce exactly. Eight draws omit bone-lesion support: overall eight-class mAP and bone-lesion AP therefore have **492 valid replicates**; other classes have 500. Missing-class AP stays undefined, never zero; overall is never averaged over fewer classes. Intervals are paired percentile 95%, conditional on presence where needed.

| Contrast | mAP50-95 delta [95% interval], pp | Fracture AP50-95 delta [95% interval], pp |
| --- | --- | --- |
| E-D | -4.8816 [-5.8201, -3.7134] | -3.6026 [-4.3497, -2.8262] |
| F-D | +0.0142 [-0.0215, +0.0264] | +0.0345 [+0.0054, +0.0684] |
| E-B | -0.4410 [-4.3444, +1.6943] | -1.5590 [-2.2471, -0.8218] |
| F-C | +0.3934 [-3.1437, +2.4984] | -0.7519 [-1.3837, -0.2473] |

E-D shows a clear overall loss. F-D's +0.0142 pp overall point change is uncertain. Under the same inference methods, E-B and F-C do not establish an overall training-condition improvement; both fracture AP50-95 intervals are below zero. F's slight within-rarity fracture gain does not erase the lower fracture AP relative to C.

Interactions are calculated inside each replicate, using the same four sets of patient indices; no interval endpoints are subtracted:

| Interaction | mAP50-95 delta [95% interval], pp | Fracture AP50-95 delta [95% interval], pp |
| --- | --- | --- |
| (E-D)-(B-A) | -0.8092 [-1.6129, +0.0396] | -0.7887 [-1.4589, -0.1293] |
| (F-D)-(C-A) | +0.0252 [-0.0176, +0.0445] | +0.0184 [-0.0284, +0.0622] |

Both overall interaction intervals include zero. The SAHI fracture interaction is negative in this fixed-run comparison. These paired differences describe the measured runs and do not establish a general causal training-by-inference interaction.

Fracture preservation remains an explicit outcome:

| Effect versus D | Fracture AP50 delta [95%], pp | Fracture AP75 delta [95%], pp | Fracture AP50-95 delta [95%], pp |
| --- | --- | --- | --- |
| E-D | -6.1653 [-7.1666, -5.3066] | -5.8050 [-7.8574, -4.0214] | -3.6026 [-4.3497, -2.8262] |
| F-D | +0.0420 [+0.0020, +0.0741] | +0.0235 [-0.0137, +0.2294] | +0.0345 [+0.0054, +0.0684] |

All AP50/AP75/AP50-95 point differences and per-class paired intervals, with valid replicate counts, are in [paired AP contrasts](paired_AP_contrasts.csv) and [results JSON](results.json). Nominal intervals do not account for multiplicity, training-seed variation, validation-driven development/selection or external generalization. Bone lesion has only 5 objects from 4 patients; other uncommon classes also have limited support.

## 6. Operating counts, recovery and unmatched predictions

Primary operating point: confidence .25 / IoU .50. AP and these operating metrics use their preserved separate matching definitions. FP means annotation-relative unmatched prediction; clinical adjudication is unavailable.

| ID | TP | FP | FN | Precision, % | Recall, % |
| --- | --- | --- | --- | --- | --- |
| A | 6467 | 884 | 737 | 87.974 | 89.770 |
| B | 6599 | 3209 | 605 | 67.282 | 91.602 |
| C | 6505 | 1055 | 699 | 86.045 | 90.297 |
| D | 6518 | 1016 | 686 | 86.514 | 90.478 |
| E | 6647 | 3648 | 557 | 64.565 | 92.268 |
| F | 6552 | 1152 | 652 | 85.047 | 90.949 |

Recovery/loss counts refer to the same annotated targets. Introduced/removed unmatched predictions use the original separate one-to-one spatial correspondence; these need not equal the net FP difference individually.

| Contrast | Net TP | Net FP | Recovered GT | Lost GT | Introduced unmatched | Removed unmatched |
| --- | --- | --- | --- | --- | --- | --- |
| E-D | 129 | 2632 | 150 | 21 | 2666 | 34 |
| F-D | 34 | 136 | 34 | 0 | 136 | 0 |
| E-B | 48 | 439 | 160 | 112 | 1566 | 1127 |
| F-C | 47 | 97 | 166 | 119 | 528 | 431 |

E-D gains 129 TP but adds 2,632 FP. F-D gains 34 TP with 136 extra FP and no lost targets at this operating point. F's recall gain is real in the deterministic annotation matching, but it comes with lower precision. It does not imply AP or clinical benefit. Fracture-specific count changes:

| Contrast | Fracture net TP | Fracture net FP | Recovered fracture GT | Lost fracture GT |
| --- | --- | --- | --- | --- |
| E-D | 72 | 1341 | 80 | 8 |
| F-D | 21 | 39 | 21 | 0 |
| E-B | -15 | 93 | 60 | 75 |
| F-C | -4 | 1 | 72 | 76 |

Paired overall operating-point uncertainty uses the same 500 patient draws (recall/precision differences in pp):

| Contrast | Net TP 95% interval | Net FP 95% interval | Recall delta 95%, pp | Precision delta 95%, pp |
| --- | --- | --- | --- | --- |
| E-D | [102.0, 159.5] | [2345.3, 2928.1] | [+1.449, +2.164] | [-22.896, -21.050] |
| F-D | [22.0, 48.0] | [111.0, 166.5] | [+0.306, +0.657] | [-1.739, -1.212] |
| E-B | [17.0, 81.0] | [305.5, 561.1] | [+0.231, +1.133] | [-3.475, -1.850] |
| F-C | [14.0, 80.0] | [29.0, 165.6] | [+0.202, +1.100] | [-1.758, -0.219] |

All confidence .05/.25/.50 by IoU .50/.75 points are retained: [class points](operating_points.csv), [overall points](overall_operating_points.csv), [overall JSON](overall_operating_points.json) and `results.json`. The latter also contains paired operating intervals by class, recovery/loss at both IoUs and unmatched-prediction correspondence. F and D have exactly identical confidence-.50 outputs and operating counts at both IoUs; no threshold was tuned.

## 7. Inference cost and runtime boundaries

RTX 4050 Laptop GPU 6 GiB; driver 560.94; Python 3.12.5; torch 2.14.1+cu126 / CUDA 12.6; torchvision 0.29.1+cu126; ultralytics 8.4.177; SAHI 0.12.8; numpy 2.3.5; OpenCV 5.0.0.93. Pinned package/GPU checks pass. See preflight for remaining versions and measured free memory.

| ID | Calls / image | Mean / median / p95 E2E, ms | Mean GPU, ms | Peak allocated / reserved, GiB |
| --- | --- | --- | --- | --- |
| A | 1.0000 | 62.33 / 61.77 / 77.48 | 33.73 | 0.0773 / 0.1113 |
| B | 5.0000 | 238.13 / 237.73 / 300.10 | 160.62 | 0.0773 / 0.1113 |
| C | 1.4161 | 60.84 / 55.45 / 132.55 | 34.97 | 0.1200 / 0.1504 |
| D | 1.0000 | 59.74 / 58.87 / 74.93 | 31.05 | 0.0773 / 0.1113 |
| E | 5.0000 | 221.14 / 216.72 / 278.53 | 147.29 | 0.0773 / 0.1113 |
| F | 1.3935 | 67.04 / 58.86 / 139.79 | 38.81 | 0.0773 / 0.1113 |

E's validation loop takes **684.712 s**; F's takes **215.834 s**. These include per-image exports and checks. E/F mean inference throughput from summed E2E times is 4.522/14.917 images/s. Model loading/provenance checks and 3 synthetic warmups per stage are recorded separately. Each pilot also includes 3 exact-output replays outside the 24-image population timing; full runs require no extra inference for pilot comparison.

E2E covers decode through final output before export and most offline verification. F's immediate D-global equality check is included; its fusion replay and export are outside measured inference latency but inside full loop wall time. F's contemporaneous decode+required-global reference is **51.920 ms**, giving **1.2912x** for the frozen <=2x gate, without another model call. The reference excludes the subsequent equality check. Peak memory is PyTorch allocated/reserved memory, excluding driver/display/other-process use.

A/B/D timing comes from their preserved accuracy runs; C comes from its preserved standalone V2 timing run. The historical loop boundaries also include different metric/export work. E and F run sequentially in new sessions without heavy bootstrap/analysis concurrently. Background load, thermal state and clock behavior are not experimentally matched. **Do not claim controlled speedups from these session differences.** Call counts are directly audited; latency values are descriptive.

## 8. Frozen scientific gates and interpretation

Reapply the original pure `assess_variant` function to F versus D, without changing its limits or selecting an algorithm:

| Frozen F-versus-D criterion | Pass |
| --- | --- |
| overall map50 95 positive | True |
| overall map50 95 interval positive | False |
| overall map50 nondecreasing | True |
| fracture ap points preserved | True |
| fracture interval tolerance | True |
| all class fp ratio | False |
| fracture fp ratio | False |
| mean latency ratio | True |

F's all-class FP ratio is **1.133858** (+13.386%), and fracture FP ratio is **1.106849** (+10.685%); both exceed 1.10. Overall mAP50-95's lower interval bound is below zero. Fracture AP point/interval preservation and the contemporaneous latency bound pass. **Primary success=False; secondary trade-off success=False.** Positive fracture AP or recovered-target counts do not override the failed criteria.

Measured conclusion: SAHI Context substantially harms overall/fracture AP with rarity-only training despite greater recall and high inference cost. Frozen CG-ALR V2 is technically valid and bounded, but its tiny uncertain overall AP gain, ordering-sensitive small class differences and excess unmatched predictions do not demonstrate a supported improvement under the original criteria. E-B/F-C and the interactions do not justify a new training or inference recommendation here. No final method is selected.

Unverified hypotheses: crop context or duplicate/ranking effects may contribute to SAHI's losses, and evidence gating/confidence updates may explain parts of F's trade-off. This phase does not isolate those mechanisms or optimize them. Preserving these negative findings is part of the experiment, not a reason to add another run.

## 9. Preservation, verification and reproducibility

The independent verifier recomputes all six full AP results and primary counts, reconciles recovery/FP partitions, replays every F bundle, checks all 500 sampled memberships/contrasts/interactions/intervals and recomputes representative bootstrap AP replicates 0/1/249/499 across all six methods. The original complete **223-test** suite passes before GPU execution; inference source/config identities remain identical to that tested commit. The additional real-data AP tie diagnostic separately checks all 3,088 images and makes no evaluation change.

Before/after receipts preserve **699 prior private files** byte-for-byte, inventory SHA-256 `79558d9c8adf81982d5fe7bf7a57f2b34f644ae7268b6d16dcace8ba71abbeab`, and **121,973 raw/prepared stat entries**, canonical SHA-256 `84601ecf4c83b72f9ed8542936ef6621ffce619cb0f748eeb58567736a427b8a`. Git checks preserve earlier sources, configs and public reports. Raw/prepared preservation here uses stat metadata, not a new read/hash of held-out pixel/label contents. Validation image/label bytes are independently hash-checked against frozen preparation receipts. See [preservation](preservation.json) and [verification](verification.json).

Private predictions, patient/image linkage, coordinates, bundles, weights and logs remain under ignored `outputs/rarity_inference_comparison/study_01/` and `outputs/rarity_inference_checks/`. Public evidence contains aggregate values and artifact hashes only. [Six-condition CSV](six_condition_matrix.csv) and `results.json` identify original sources for A/B/C/D and new E/F output hashes. The pre-inference contract records execution commands as an audit trail; existing successful attempts must be reused rather than overwritten or needlessly rerun. The [publication checks](publication_checks.json) reconcile public values and privacy-safe schemas with saved verified evidence.

Limitations: one training seed per condition, validation-selected epochs 22/26, unpaired stochastic augmented views between training conditions, prior validation method development, conditional rare-class intervals, equal-confidence AP ordering and uncontrolled timing sessions. Dataset-level localization does not demonstrate clinical safety, diagnostic validity or external hospital performance. The student writes the final thesis text and controls the next authorization.

**Phase 4E stops here.** No held-out evaluation, new training, method optimization, final protocol/algorithm selection or Phase 5A synthesis follows automatically.
