# Phase 5A — Scientific analysis of the six validation conditions

24 October 2026. Retrospective research evidence for the student's thesis work; not final thesis prose or independent confirmation. **No condition meets all historical criteria for a supported adaptive improvement.** Rarity-only training produces mixed class changes and lower fracture AP; SAHI Context loses overall/fracture AP; V2 recovers annotated objects with excess unmatched predictions and negligible, uncertain overall AP changes.

## Scope and sources

A/B/C use the standard epoch-22 checkpoint with Standard/SAHI Context/CG-ALR V2 inference. D/E/F use the rarity-only epoch-26 checkpoint with the same three methods. The population is the same 3,088 validation images, 914 patients and 7,204 objects. All views/examinations stay grouped by patient. Official AP is the original-coordinate pinned evaluator, macro-averaged over all eight classes. Operating counts separately use confidence .25 / IoU .50 and the original stable, confidence-ordered one-to-one matcher.

Sources: [verified A–F results](../rarity_inference_comparison/results.json), [original report](../rarity_inference_comparison/RARITY_INFERENCE_COMPARISON_REPORT.md), and [source manifest](source_manifest.json). All six operating count sets were recalculated from saved predictions/verified validation labels. No model, checkpoint, radiograph or held-out label/prediction is loaded by the analysis. No V3 rule is implemented or evaluated.

## Overall behavior and cost

| ID | mAP50 | mAP75 | mAP50–95 | TP / FP / FN | Precision / recall, % | Calls/image | Recorded ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.599005 | 0.302277 | 0.333869 | 6467 / 884 / 737 | 87.97 / 89.77 | 1.0000 | 62.33 |
| B | 0.525768 | 0.269773 | 0.293145 | 6599 / 3209 / 605 | 67.28 / 91.60 | 5.0000 | 238.13 |
| C | 0.598619 | 0.302361 | 0.333759 | 6505 / 1055 / 699 | 86.04 / 90.30 | 1.4161 | 60.84 |
| D | 0.611845 | 0.315562 | 0.337551 | 6518 / 1016 / 686 | 86.51 / 90.48 | 1.0000 | 59.74 |
| E | 0.517918 | 0.271754 | 0.288736 | 6647 / 3648 / 557 | 64.57 / 92.27 | 5.0000 | 221.14 |
| F | 0.612083 | 0.315752 | 0.337693 | 6552 / 1152 / 652 | 85.05 / 90.95 | 1.3935 | 67.04 |

Calls are directly audited. Latency is descriptive across different sessions, not a controlled speed comparison. C/F each permit at most three calls per image. Their contemporaneous required-global references give mean end-to-end ratios about 1.322/1.291; these are the historical engineering cost controls. B/E use five calls, with substantially larger recorded latency. E2E excludes loading/warmups and export; historical outer-loop boundaries differ.

## Training, inference and interactions

| Contrast | Overall mAP50–95 delta [95%], pp | Fracture AP50–95 delta [95%], pp |
| --- | --- | --- |
| D-A | +0.3682 [-3.1573, +2.4812] | -0.7703 [-1.4017, -0.2476] |
| B-A | -4.0724 [-4.8339, -3.1124] | -2.8139 [-3.3767, -2.2659] |
| C-A | -0.0110 [-0.0354, +0.0087] | +0.0161 [-0.0084, +0.0483] |
| E-D | -4.8816 [-5.8201, -3.7134] | -3.6026 [-4.3497, -2.8262] |
| F-D | +0.0142 [-0.0215, +0.0264] | +0.0345 [+0.0054, +0.0684] |
| E-B | -0.4410 [-4.3444, +1.6943] | -1.5590 [-2.2471, -0.8218] |
| F-C | +0.3934 [-3.1437, +2.4984] | -0.7519 [-1.3837, -0.2473] |
| (E-D)-(B-A) | -0.8092 [-1.6129, +0.0396] | -0.7887 [-1.4589, -0.1293] |
| (F-D)-(C-A) | +0.0252 [-0.0176, +0.0445] | +0.0184 [-0.0284, +0.0622] |

These are fraction differences multiplied by 100 (percentage points). The [complete contrast table](paired_AP_contrasts.csv) includes all three AP metrics for all eight classes and overall, for all nine requested contrasts. Interactions reuse the original within-replicate four-condition differences; interval endpoints are never subtracted.

**Training effect, D−A:** overall AP50/AP75/AP50–95 changes are +1.2840/+1.3284/+0.3682 pp. The overall primary interval includes zero; fracture AP50–95 falls by 0.7703 pp with a negative nominal interval. Periosteal-reaction AP50–95 increases by 2.1873 pp [0.7978, 3.6061]. This is a measured validation benefit for that finding, with no demonstrated overall gain and no fracture preservation. Text/device and finding contributions are separated in [arithmetic components](map_components.csv): each class retains its original 1/8 weight. Text is an image annotation target; metal/device detections can have clinical relevance and are not treated as meaningless nuisance errors. No alternative subset mAP replaces the primary metric.

**Inference effects:** B−A and E−D show substantial overall and fracture AP losses despite increased recall. C−A is a tiny negative overall change; F−D is a tiny positive overall change with an interval crossing zero. Fracture point changes for V2 are small and do not establish clinical improvement. At .25/.50, C recovers 38 objects, loses none and adds 171 unmatched predictions; F recovers 34, loses none and adds 136. Precision falls for both. SAHI recovery/loss is 148/16 for B−A and 150/21 for E−D; the net counts alone conceal lost objects. [Recovery by class and patient support](recovery.csv) retains those identities as aggregate partitions.

**Cross-training and interaction effects:** E−B and F−C do not establish overall improvement; both fracture AP50–95 intervals are negative. Both overall interaction intervals include zero. The negative SAHI fracture interaction describes these fixed runs; it is not a general causal interaction between training and inference. A single training realization cannot isolate sampler effects from stochastic training variation.

## All eight classes

### AP50

| Class | Objects / patients | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 2808 / 538 | 0.921175 | 0.867145 | 0.921327 | 0.916198 | 0.854545 | 0.916618 |
| metal | 96 / 24 | 0.890409 | 0.636297 | 0.890475 | 0.905640 | 0.641599 | 0.905953 |
| text | 3611 / 913 | 0.988744 | 0.972154 | 0.988758 | 0.989619 | 0.966628 | 0.989615 |
| periostealreaction | 493 / 163 | 0.565996 | 0.447199 | 0.563268 | 0.586160 | 0.434657 | 0.585385 |
| pronatorsign | 83 / 83 | 0.571072 | 0.538580 | 0.570684 | 0.548437 | 0.474409 | 0.548654 |
| bonelesion | 5 / 4 | 0.415000 | 0.402895 | 0.415000 | 0.413182 | 0.371467 | 0.413182 |
| softtissue | 72 / 61 | 0.296036 | 0.257745 | 0.295759 | 0.337790 | 0.316226 | 0.339522 |
| boneanomaly | 36 / 14 | 0.143610 | 0.084129 | 0.143685 | 0.197735 | 0.083814 | 0.197735 |
### AP75

| Class | Objects / patients | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fracture | 2808 / 538 | 0.534596 | 0.511200 | 0.534976 | 0.525676 | 0.467626 | 0.525911 |
| metal | 96 / 24 | 0.559162 | 0.383416 | 0.559166 | 0.610374 | 0.385000 | 0.610374 |
| text | 3611 / 913 | 0.823867 | 0.814272 | 0.823963 | 0.830508 | 0.809844 | 0.830542 |
| periostealreaction | 493 / 163 | 0.099578 | 0.067358 | 0.099809 | 0.139762 | 0.074143 | 0.140240 |
| pronatorsign | 83 / 83 | 0.130848 | 0.119074 | 0.130835 | 0.097052 | 0.100097 | 0.097071 |
| bonelesion | 5 / 4 | 0.195000 | 0.198774 | 0.195000 | 0.195000 | 0.197139 | 0.195000 |
| softtissue | 72 / 61 | 0.056896 | 0.054425 | 0.056868 | 0.109874 | 0.133219 | 0.110636 |
| boneanomaly | 36 / 14 | 0.018274 | 0.009668 | 0.018271 | 0.016244 | 0.006960 | 0.016244 |
### AP50–95

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

Fracture AP50 is lower for D than A by 0.4977 pp; AP75 is lower by 0.8919 pp. F's small within-rarity gain does not repair the fracture deficit relative to A/C. Rarity-only periosteal recall rises from 298/493 to 333/493, while FP rises from 295 to 328. Pronator-sign and soft-tissue recall gains also carry larger unmatched-prediction costs:

| Class | A TP / FP / FN | D TP / FP / FN |
| --- | --- | --- |
| pronatorsign | 46 / 35 / 37 | 55 / 75 / 28 |
| bonelesion | 1 / 0 / 4 | 2 / 1 / 3 |
| softtissue | 17 / 9 / 55 | 27 / 60 / 45 |
| boneanomaly | 4 / 3 / 32 | 8 / 12 / 28 |

Bone lesion has five objects from four patients: one versus two operating TPs does not establish generalization or a dependable training benefit. Bone anomaly has 36 objects/14 patients; AP50 rises while AP75/AP50–95 fall, so a loose-overlap gain cannot be called uniform localization improvement. Soft tissue has 72 objects/61 patients; its AP50–95 training interval still includes zero. Metal has 96 objects/24 patients. These supports limit interpretation even when a point difference is large.

## Evaluation sensitivity and uncertainty

Reuse the original 500 paired patient-bootstrap draws, seed 314159. Bone-lesion and overall AP use 492 valid draws because eight omit that class; all other class AP intervals use 500. Undefined AP is never replaced by zero and overall never averages fewer than eight classes. These are conditional, nominal percentile intervals, not training-seed uncertainty, multiplicity-adjusted inference or clinical significance tests. No bootstrap or AP calculation is repeated unnecessarily.

The [verified tie trace](../rarity_inference_comparison/ap_tie_diagnostic.json) establishes default unstable sorting of equal-confidence predictions in the pinned evaluator. For F−D, six non-target classes retain identical predictions and native TP flags, yet their aggregate contribution is +0.009929 pp of the reported +0.014210 pp overall difference (about 69.9%). Soft tissue alone contributes +0.008210 pp. These contributions are ordering effects, not detected improvements. They are an arithmetic explanation of historical metrics, not a replacement score or correction. C has the same documented sensitivity. Existing traces suffice; no second tie convention or alternate AP experiment is introduced here.

Other limits: one seed per training condition; different augmentation realizations and native training trajectories; validation-based checkpoint selection; repeated validation use for method design and interpretation; rare-class/missing-class uncertainty; annotation-relative unmatched predictions without clinical adjudication; imperfect/unlabeled findings; and uncontrolled cross-session timing. No independent held-out or external-hospital evidence exists. Patient bootstrap does not remove these limitations.

## Conclusion and next decision

The negative/mixed findings are valid research outcomes. Training changed the class trade-off; five-call tiling increased detections but harmed precision/ranking; bounded V2 confidence confirmation mostly promoted unmatched rows. The [mechanism report](CG_ALR_V2_DIAGNOSTICS.md) separates coverage, matching, confidence and geometry. The [V3 feasibility report](V3_FEASIBILITY_AND_RECOMMENDATIONS.md) recommends only a constrained design discussion; it selects no final method and authorizes no further experiment. Formal institutional thesis-scope alignment remains pending.


## Reproduction and verification

The additive `src/wrist_analysis/` package reads sealed validation evidence only. `python -m wrist_analysis analyze --attempt NEW_NAME` creates a new private analysis; existing attempts refuse replacement. Publishing (`python -m wrist_analysis publish --attempt NEW_NAME`) and report creation (`python -m wrist_analysis.reports`) likewise refuse existing destinations. Preserve this completed publication rather than regenerating it in place.

For an intentional independent repeat check of this saved analysis, use `python -m wrist_analysis.verify --attempt analysis_02 --receipt outputs/phase_5a_analysis/audit_repeat_01/verification.json`. This writes a new private receipt and preserves the public verification record. The verification record checks historical fingerprints, all six count totals with a separate matcher, promotion/recovery/ROI partitions, nested oracle bounds, average-rank feature-separation arithmetic, public privacy and report links. The full test suite and final aggregate hashes are recorded separately in `verification.json` and `automated_tests.json`.
