"""Render concise technical research reports from the checked aggregates."""

import csv
from .evidence import ROOT, PUBLIC, CLASSES, read


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(str(value) for value in row)+' |' for row in rows]])


def csv_rows(name):
    with (PUBLIC/name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def create_reports():
    result = read('docs/evidence/rarity_inference_comparison/results.json')
    diagnostic = read('docs/evidence/phase_5a_analysis/v2_mechanism.json')
    errors = read('docs/evidence/phase_5a_analysis/error_categories.json')
    summary = csv_rows('six_condition_summary.csv'); contrasts = csv_rows('paired_AP_contrasts.csv')
    by_error = {(row['experiment'], row['class']): row for row in errors}
    def interval(contrast, name):
        row = next(r for r in contrasts if r['contrast'] == contrast and r['class'] == name and r['metric'] == 'ap50_95')
        return f"{float(row['point_delta'])*100:+.4f} [{float(row['lower'])*100:+.4f}, {float(row['upper'])*100:+.4f}]"
    matrix = table(['ID', 'mAP50', 'mAP75', 'mAP50–95', 'TP / FP / FN', 'Precision / recall, %', 'Calls/image', 'Recorded ms'],
        [[r['experiment'], *[f"{float(r[k]):.6f}" for k in ('map50', 'map75', 'map50_95')],
          ' / '.join(r[k] for k in ('tp', 'fp', 'fn')), f"{float(r['precision'])*100:.2f} / {float(r['recall'])*100:.2f}",
          f"{float(r['mean_model_calls']):.4f}", f"{float(r['recorded_mean_ms']):.2f}"] for r in summary])
    contrast_table = table(['Contrast', 'Overall mAP50–95 delta [95%], pp', 'Fracture AP50–95 delta [95%], pp'],
        [[key, interval(key, 'overall'), interval(key, 'fracture')] for key in dict.fromkeys(r['contrast'] for r in contrasts)])
    ap_tables = []
    for metric in ('ap50', 'ap75', 'ap50_95'):
        ap_tables.append('### '+{'ap50': 'AP50', 'ap75': 'AP75', 'ap50_95': 'AP50–95'}[metric]+'\n\n'+table(
            ['Class', 'Objects / patients', *list('ABCDEF')],
            [[name, f"{by_error['A', name]['objects']} / {by_error['A', name]['patients']}",
              *[f"{result['conditions'][c]['AP'][name][metric]:.6f}" for c in 'ABCDEF']] for name in CLASSES]))
    rare_counts = table(['Class', 'A TP / FP / FN', 'D TP / FP / FN'],
        [[name, *[' / '.join(str(by_error[c, name][k]) for k in ('tp', 'fp', 'fn')) for c in 'AD']]
         for name in ('pronatorsign', 'bonelesion', 'softtissue', 'boneanomaly')])
    synthesis = f"""# Phase 5A — Scientific analysis of the six validation conditions

24 October 2026. Retrospective research evidence for the student's thesis work; not final thesis prose or independent confirmation. **No condition meets all historical criteria for a supported adaptive improvement.** Rarity-only training produces mixed class changes and lower fracture AP; SAHI Context loses overall/fracture AP; V2 recovers annotated objects with excess unmatched predictions and negligible, uncertain overall AP changes.

## Scope and sources

A/B/C use the standard epoch-22 checkpoint with Standard/SAHI Context/CG-ALR V2 inference. D/E/F use the rarity-only epoch-26 checkpoint with the same three methods. The population is the same 3,088 validation images, 914 patients and 7,204 objects. All views/examinations stay grouped by patient. Official AP is the original-coordinate pinned evaluator, macro-averaged over all eight classes. Operating counts separately use confidence .25 / IoU .50 and the original stable, confidence-ordered one-to-one matcher.

Sources: [verified A–F results](../rarity_inference_comparison/results.json), [original report](../rarity_inference_comparison/RARITY_INFERENCE_COMPARISON_REPORT.md), and [source manifest](source_manifest.json). All six operating count sets were recalculated from saved predictions/verified validation labels. No model, checkpoint, radiograph or held-out label/prediction is loaded by the analysis. No V3 rule is implemented or evaluated.

## Overall behavior and cost

{matrix}

Calls are directly audited. Latency is descriptive across different sessions, not a controlled speed comparison. C/F each permit at most three calls per image. Their contemporaneous required-global references give mean end-to-end ratios about 1.322/1.291; these are the historical engineering cost controls. B/E use five calls, with substantially larger recorded latency. E2E excludes loading/warmups and export; historical outer-loop boundaries differ.

## Training, inference and interactions

{contrast_table}

These are fraction differences multiplied by 100 (percentage points). The [complete contrast table](paired_AP_contrasts.csv) includes all three AP metrics for all eight classes and overall, for all nine requested contrasts. Interactions reuse the original within-replicate four-condition differences; interval endpoints are never subtracted.

**Training effect, D−A:** overall AP50/AP75/AP50–95 changes are +1.2840/+1.3284/+0.3682 pp. The overall primary interval includes zero; fracture AP50–95 falls by 0.7703 pp with a negative nominal interval. Periosteal-reaction AP50–95 increases by 2.1873 pp [0.7978, 3.6061]. This is a measured validation benefit for that finding, with no demonstrated overall gain and no fracture preservation. Text/device and finding contributions are separated in [arithmetic components](map_components.csv): each class retains its original 1/8 weight. Text is an image annotation target; metal/device detections can have clinical relevance and are not treated as meaningless nuisance errors. No alternative subset mAP replaces the primary metric.

**Inference effects:** B−A and E−D show substantial overall and fracture AP losses despite increased recall. C−A is a tiny negative overall change; F−D is a tiny positive overall change with an interval crossing zero. Fracture point changes for V2 are small and do not establish clinical improvement. At .25/.50, C recovers 38 objects, loses none and adds 171 unmatched predictions; F recovers 34, loses none and adds 136. Precision falls for both. SAHI recovery/loss is 148/16 for B−A and 150/21 for E−D; the net counts alone conceal lost objects. [Recovery by class and patient support](recovery.csv) retains those identities as aggregate partitions.

**Cross-training and interaction effects:** E−B and F−C do not establish overall improvement; both fracture AP50–95 intervals are negative. Both overall interaction intervals include zero. The negative SAHI fracture interaction describes these fixed runs; it is not a general causal interaction between training and inference. A single training realization cannot isolate sampler effects from stochastic training variation.

## All eight classes

{chr(10).join(ap_tables)}

Fracture AP50 is lower for D than A by 0.4977 pp; AP75 is lower by 0.8919 pp. F's small within-rarity gain does not repair the fracture deficit relative to A/C. Rarity-only periosteal recall rises from 298/493 to 333/493, while FP rises from 295 to 328. Pronator-sign and soft-tissue recall gains also carry larger unmatched-prediction costs:

{rare_counts}

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
"""
    gt_table = table(['Model / class', 'Objects / patients', 'TP', 'FN', 'Low confidence', 'Localization', 'Classification', 'No residual candidate'],
        [[c+' / '+name, f"{by_error[c, name]['objects']} / {by_error[c, name]['patients']}", by_error[c, name]['tp'], by_error[c, name]['fn'],
          *[by_error[c, name]['gt_errors'].get(k, 0) for k in ('low_confidence', 'localization', 'classification', 'no_residual_candidate')]]
         for c in 'AD' for name in ('fracture', 'periostealreaction')])
    fp_table = table(['ID', 'Background/unannotated', 'Localization', 'Duplicate', 'Classification'],
        [[c, *[sum(r['fp_errors'].get(k, 0) for r in errors if r['experiment'] == c)
                for k in ('background_or_unannotated', 'localization', 'duplicate', 'classification')]] for c in 'ABCDEF'])
    funnel_keys = [('Eligible anchors', 'candidates'), ('Anchors below .15 fixed-cap recovery floor', 'candidates_below_fixed_cap_recovery_floor'),
        ('Local calls', 'local_calls'), ('Retained local evidence', 'retained_evidence'), ('Accepted protected matches', 'protected_matches'),
        ('Accepted mutable matches / score changes', 'mutable_matches'), ('Unassigned retained evidence', 'unmatched_evidence'),
        ('Rejected matching pairs', 'rejected_pairs'), ('Mutable pairs blocked by protected owner', 'mutable_pair_blocked_by_protected_owner'),
        ('ROIs without retained evidence', 'rois_without_retained_evidence'), ('ROIs without output update', 'rois_without_score_update'),
        ('ROI images without output update', 'roi_images_without_score_update'), ('ROIs contributing a recovery', 'rois_contributing_recovery')]
    funnel = table(['Stage', 'C (standard)', 'F (rarity-only)'], [[label, *[diagnostic[c]['counts'].get(key, 0) for c in 'CF']] for label, key in funnel_keys])
    stage_order = [('No eligible same-class global IoU≥.50 anchor', 'no_eligible_global_iou05_anchor'),
        ('Eligible anchor outside selected ROI', 'eligible_anchor_outside_selected_roi'),
        ('Covered anchor without mutable match', 'covered_without_mutable_match'),
        ('Mutable match without operating recovery', 'matched_without_operating_recovery'), ('Recovered', 'recovered')]
    stage_lookup = {(c, r['class'], r['stage']): (r['events'], r['patients']) for c in 'CF' for r in diagnostic[c]['object_stage_support']}
    stage_table = table(['Stage (objects / independent patients)', 'C fracture', 'C periosteal', 'F fracture', 'F periosteal'],
        [[label, *[' / '.join(map(str, stage_lookup.get((c, name, key), (0, 0))))
                   for c in 'CF' for name in ('fracture', 'periostealreaction')]] for label, key in stage_order])
    oracle_keys = [('All retained weak rows ≥.001', 'all_retained_weak_global'), ('Eligible .05–<.25 anchors', 'eligible_global_band'),
        ('Eligible anchors inside selected ROIs', 'selected_roi_covered'), ('Also reachable under fixed +.10 score cap', 'fixed_cap_reachable'),
        ('Unassigned saved local evidence (separate graph)', 'unassigned_saved_local_evidence')]
    oracle = table(['Oracle capacity (objects)', 'C fracture', 'C periosteal', 'F fracture', 'F periosteal'],
        [[label, *[diagnostic[c]['oracle_capacities'][name][key] for c in 'CF' for name in ('fracture', 'periostealreaction')]] for label, key in oracle_keys])
    feature_rows = []
    for c in 'CF':
        for item in diagnostic[c]['promotion_feature_separation']:
            if item['class'] == 'overall' and item['feature'] in ('global_confidence', 'local_confidence', 'global_local_iou', 'confident_context'):
                quant = lambda group: f"{item[group]['median']:.3f} [{item[group]['p25']:.3f}, {item[group]['p75']:.3f}]"
                feature_rows.append([c, item['feature'], quant('TP'), quant('FP'), f"{item['probability_TP_feature_exceeds_FP']:.3f}"])
    features = table(['Condition', 'Feature', 'Recovered TP median [IQR]', 'Promoted FP median [IQR]', 'P(TP feature > FP)'], feature_rows)
    geometry = table(['Condition', 'Accepted / vetoed', 'Improved / worsened / equal', 'Changes above .25', 'Geometry-only recoveries / losses'],
        [[c, f"{diagnostic[c]['counts']['geometry_accepted']} / {diagnostic[c]['counts']['geometry_collision_veto']}",
          ' / '.join(str(sum(r['events'] for r in diagnostic[c]['geometry'] if r['status'] == status)) for status in ('improved', 'worsened', 'equal')),
          sum(r['events'] for r in diagnostic[c]['geometry'] if r['above_operating_confidence']), '0 / 0'] for c in 'CF'])
    mechanism = f"""# CG-ALR V2 — saved-execution diagnostics

24 October 2026. Analyze the unchanged standard C and rarity-only F execution bundles. Public evidence is [v2_mechanism.json](v2_mechanism.json), [error categories](error_categories.json), [strata](error_strata.csv) and [recovery](recovery.csv); exact sources/hashes are in the [manifest](source_manifest.json). Detailed linked events stay private. No local inference or V3 replay is performed.

## 1. Detection errors before refinement

Use the frozen residual priority: operating TP, wrong-class IoU≥.50, same-class low-confidence IoU≥.50, same-class operating-confidence IoU≥.10, then no residual candidate. Residual assignments consume rows/objects once. “No residual candidate” is not complete absence of all spatial predictions.

{gt_table}

Low-confidence failures dominate both target classes, but the two models make different errors. D has more fracture localization/no-residual misses and fewer periosteal low-confidence misses than A. Classification is rare in these two classes. Other classes' counts and patient supports are retained in the synthesis/error tables; V2 cannot repair their classification or coverage because only classes 0 and 3 are mutable.

Annotation-relative unmatched prediction categories:

{fp_table}

Background/unannotated and localization categories dominate; SAHI adds many duplicates as well. A duplicate here is an extra operating row with same-class annotated IoU≥.50 after one-to-one matching, not necessarily a pair above the inference NMS threshold .70. None is a clinically adjudicated false alarm. Category priority is operational and is not a causal error taxonomy.

At ideal 768 scale, periosteal <16-pixel targets have 61 objects/45 patients: A detects 12, D 24. For 16–<32 pixels, support is 212/110 with 119/137 TPs. Fracture 32–<64-pixel support is 1,633/488 with 1,394 TPs for both; 64–<128 has 1,133/350 with 1,039/1,029. Smaller targets are harder in these data, but density, overlap, aspect and border strata are confounded descriptive associations. [All strata](error_strata.csv) report objects/patients/TP/FN; zero-support strata are absent, not observed zero-recall groups. No subgroup AP with ambiguous FP attribution is invented.

## 2. Frozen pipeline and candidate generation

The algorithm receives only global predictions and image dimensions. It retains global confidence≥.001; candidates are fracture/periosteal, .05≤score<.25, short side≥1 ideal 768 pixel. Priority is .6Q+.4Z with short-side references 64/32; each represented target class gets a first head, then pooled ordering. Crops cover at least .75 of each image dimension with expanded anchor containment, preserving aspect/context. At most two selected ROIs/three calls are permitted. Coalescing requires ROI IoU≥.70 and envelope containment.

C has 827 fracture/2,046 periosteal eligible anchors; F has 740/1,518. Selected/coalesced/budget-exhausted counts are 1,285/1,512/76 and 1,215/1,000/43. There are no observed crop-geometry rejections. Typical ideal zoom is only about 1.333×, so these are broad contextual views, not high-magnification crops. Candidate confidence/size distributions and independent patient supports are in the diagnostic JSON.

{funnel}

The +.10 cap means a row originally below .15 cannot reach the .25 operating point in one update: 2,177/2,873 C candidates and 1,678/2,258 F candidates fall below this algebraic floor. This does not make them useless for AP: low-score ranking and geometry can still matter. Removing them is not an approved optimization.

## 3. ROI efficiency and local matching

Zero/one/two ROI image counts are 2,074/743/271 (C) and 2,121/719/248 (F). ROI images total 1,014/967; mean calls/image are 1.4161/1.3935. No-output-update calls are 897/1,285 (69.81%) and 843/1,215 (69.38%). Only 37/34 ROIs contribute to an operating recovery (2.88%/2.80% of local calls). These are observed productivity labels after local inference, not signals available for free beforehand.

Local filtering rejects 27,923/18,235 low-score rows, 5,229/3,453 non-target rows and 85/77 artificial-boundary rows; .70 union NMS suppresses 658/640 of 2,665/2,644 eligible rows, leaving 2,007/2,004. C/F matching consumes 1,907/1,883 retained rows. Protected high-confidence rows account for 75.93%/77.22% of accepted matches and cannot be updated. Pair matching is same-class, global/local IoU≥.50, greedy one-to-one by IoU then scores/IDs; mutable envelopes must fit the local source ROI. Counts of rejected pairs are not unique discarded objects. Protected-owner blocking counts include multiple competing pairs and are not recoverable-target counts.

Of the 100/121 unassigned retained rows, 45/43 overlap some same-class annotation at IoU≥.50, but most overlap already-detected objects. The separate one-to-one local oracle below limits baseline missed-target capacity to 7/10. It is not justified to add all unassigned evidence as new detections.

Higher seed confidence is associated with any update: for fracture-only seeds, productive/nonproductive median confidence is .148/.102 (C), .133/.106 (F); for periosteal-only seeds .177/.126 and .162/.125. Context/size associations differ by seed-class composition. Productivity includes harmful score changes; these associations neither prove causality nor provide a validated ROI-selection rule.

## 4. Confidence fusion and promotion outcomes

Matched mutable rows receive min(global+.10, local, .35), with conservative float32 quantization. The same checkpoint supplies both views, so local confidence is correlated evidence, not independent confirmation. C changes 459 scores: 250 remain below .25, 38 promoted rows recover objects, and 171 promoted rows are unmatched. F changes 429: 259 remain below .25, 34 recover objects, one claims an already-detected target, and 135 are unmatched. That reassignment makes one unchanged inherited row unmatched: **F net TP+34/FP+136 is compatible with 35 promoted TPs/135 promoted FPs.** Protected output fields do not guarantee preservation of one-to-one match ownership.

{features}

TP columns use newly recovered objects only (C n=38, F n=34); FP columns use unmatched promoted rows (171/135), excluding F's reassigned TP and inherited displaced row. The final probability gives half credit for ties; .5 means no univariate stochastic separation. It is not a trained classifier, optimized cutpoint, cross-validated discrimination estimate or significance claim. Event/image/patient support is reported by class/outcome in JSON. Groups overlap by patient and rows are not independent observations.

Local score and global/local IoU distributions overlap substantially. Within fracture, IoU separation is .467/.454 (C/F); within periosteal .537/.430. Higher IoU alone is therefore not supported as a reliable beneficial-promotion discriminator. Global score separates moderately (.703/.601 overall) but IQRs overlap; box size even changes association direction (.369/.542 for short side). Confident surrounding detections are more common for unmatched promotions, especially F, but class/context confounding prevents a causal or ready-to-deploy rule. Selective fusion is an evidence-backed problem to investigate, not an established solution.

## 5. Geometry refinement

V2 proposes a quarter-weight blend only for mutable matches with local confidence≥.50. Invalid/collision proposals retain the original box and V1 score, without rematching. Proposed changes total 82/116; accepted/veto counts and annotation-relative effects are:

{geometry}

Accepted changes occur on 64/89 images (2.07%/2.88%). Best same-class annotated IoU is used here; an unannotated box gets zero and no one-to-one TP is implied. Some changes worsen IoU; an identical maximum can also mean no nearby same-class annotation. The coordinate footprint is small and leaves most outputs untouched. Confidence-only V1 reconstruction from the frozen score updates yields the same .25/.50 object recovery/loss as V2 for both checkpoints, isolating zero operating-point gain from geometry. No alternate AP/V3 metric is calculated. Small AP effects may arise at stricter IoU or from ranking/ties, rather than broad localization recovery.

## 6. Structural recovery limits and retrospective oracles

Baseline missed target support is C fracture 345 objects/206 patients and periosteal 195/105; F fracture 355/204 and periosteal 160/97. Partition these exact objects; entries below show objects / independent patients. Stage groups can share patients, so patient counts are not additive.

{stage_table}

“Eligible anchor” here explicitly requires same-class original-box IoU≥.50 to the missed annotation, in addition to actual V2 candidate eligibility. “Covered” requires that candidate's expanded envelope inside a selected ROI. There are zero misses in the outside-selected-ROI stage: within this diagnostic definition, budget/coalescing does not reduce available adequate-anchor coverage. This says nothing about new, unobserved crop shapes or lower-IoU anchors.

Of the no-eligible-anchor fracture misses, 151/150 still have some same-class retained global row at IoU≥.50; 74/88 do not. Periosteal figures are 61/51 with and 23/26 without. The first group can include rows below .05 or otherwise ineligible/consumed rows. All retained rows are post-NMS, confidence≥.001 and max-300; absence here does not prove absence of a latent neural feature. A weak global anchor can exist while local evidence is missing or assigned elsewhere. Among covered-without-mutable-match misses, retained local evidence overlaps the annotation for only 4/8 C fracture/periosteal and 0/8 F. Those signals can fail the global/local matching or ownership rule; they cannot simply be counted as accepted matches.

Oracle capacities use the actual baseline missed targets and maximum one-to-one graph matching at annotated IoU≥.50. Original-box graphs use only unused baseline weak predictions, preserving already-operating matches. The final graph uses unassigned retained local evidence separately. They measure additional baseline objects under stated constraints, not AP, achievable recall, a proposed detector, or additive gains across rows:

{oracle}

The .05 band has 230/199 one-to-one original-box capacities, reduced to 99/83 after the existing score-cap reachability condition, before requiring actual useful local evidence. Observed recovery is 38/34. Geometry is excluded from these original-box upper bounds. A GT-centered crop would be a different oracle with unseen local outputs; no such inference or claimed performance is produced. Saved local evidence comes only from the already-selected ROIs and cannot estimate an implementable image-only fallback's coverage or FP rate.

## 7. Implications

The three central limits are unreliable promotion evidence, missing/ineligible global anchors, and low useful-update yield from broad local views. Geometry is a small, mixed-quality component, not the main demonstrated recovery lever. More aggressive fusion or unassigned-local insertion could increase unmatched predictions; more calls cannot by itself repair classification, poor training representation or correlated confident errors. These diagnostics justify examining FP control first and ROI utility second, with fallback coverage remaining a separate unresolved research question.
"""
    recommendations = """# CG-ALR V3 — feasibility and recommendations

24 October 2026. **CONDITIONAL GO for a bounded technical design only.** V3 implementation, parameter selection, saved-prediction performance trials, fresh inference and held-out access remain unapproved. This is advisory research planning, not a final method choice.

## What the evidence supports

The [scientific synthesis](PHASE_5A_ANALYSIS_REPORT.md) and [V2 diagnostics](CG_ALR_V2_DIAGNOSTICS.md) establish excess unmatched promotions, many calls without updates, and substantial misses without eligible anchors. They do not establish a V3 gain. Same-model local score/IoU confirmation alone has poor, inconsistent descriptive separation, and most geometry changes have no effect on the operating detection counts.

| Direction | Measured basis | Main value | Current evidence strength | Priority |
| --- | --- | --- | --- | --- |
| Selective confidence fusion | C +38 TP/+171 FP; F +34/+136, with overlapping local-evidence features | Control harmful promotions while retaining genuine recovery | Strong evidence of the problem; weak evidence for any particular replacement rule | 1: serious design direction |
| ROI utility and evidence ownership | About 70% of local calls produce no update; protected rows consume 76–77% of matches | Reduce avoidable local work and redundant confirmation | Strong retrospective utilization evidence; no prospective selector validated | 2: serious design direction |
| Geometry revision | 64/90 accepted changes, 29/43 improved and 17/22 worsened; no operating recovery contribution | Potential small localization correction | Limited footprint, mixed quality; no large opportunity demonstrated | Defer |
| Limited image-only fallback | 225/238 fracture misses lack an adequate eligible anchor; 7/10 unmatched-local oracle capacity overall | Address anchor absence | Structural need is clear; implementable proposal coverage and false positives are unmeasured | Conditional later feasibility, not a third implementation commitment |

## Candidate 1 — selective confidence fusion

Problem: most confidence-threshold crossings are unmatched; matching a repeat local error is not reliable confirmation. Hypothesis: evidence validation that accounts for competing/protected detections and context may distinguish corroboration from repeated error better than unconditional capped promotion. Global/local scores, overlap, box size, class, existing confident detections and ownership conflicts are available at inference. Ground-truth alignment is not.

The desired effect is fewer unmatched promotions with retained annotated recovery and fracture AP. No numerical gain is predicted. Lower scores could also reduce recall or AP; good AP is not guaranteed by better .25 precision. Simply tightening IoU is not supported by the observed .43–.54 within-class separation probabilities; global-confidence and context distributions overlap and can be class-confounded. Stronger local scores show inconsistent separation across training/classes. Do not choose a threshold from these plots or fit a validation classifier.

Compute impact could be small if only fusion checks change; a requirement for a second new local view would consume additional budget and correlated evidence may remain unreliable. Complexity is low to moderate for explicit deterministic checks, higher for learned calibration, which would require new approved training/calibration data and is outside this phase. Main risks are suppressing subtle findings, benefiting one target at another's expense, protected-TP reassignment, and validation overfitting. Existing bundles support descriptions of those failures; they cannot confirm a replacement policy. Eventually required: a separately approved, preregistered common-rule comparison with actual final original-coordinate outputs, paired patient uncertainty and fixed FP/AP gates.

## Candidate 2 — ROI utility and matching efficiency

Problem: 897/843 calls produce no final change; only 37/34 ROIs contribute to recovery, and most accepted local matches are protected. Hypothesis: inference-available seed/context/ownership signals might avoid redundant protected confirmation while preserving the limited candidate opportunities. Candidate confidence, ideal size/zoom, class, expanded-envelope coverage, overlap with confident global detections and number of competing anchors are available before local inference. Whether a future crop returns useful evidence is not known in advance.

The plausible value is efficiency with preserved AP/recall/FP, not a claimed recall increase: the existing selected ROIs already cover all adequate eligible anchors counted by the diagnostic upper bounds. Changing greedy evidence ownership may redistribute evidence rather than create it, and protected matches may appropriately prevent duplicate or incorrect confirmation. Productive ROI means any update, often harmful; it is not a utility target to maximize blindly.

Call count might fall, while CPU planning overhead could rise modestly. Complexity is moderate because coverage, one-to-one assignment and deterministic coalescing must remain auditable. Risks are omitting rare useful views, moving work to confidently wrong regions, and reducing apparently unproductive calls that affect low-confidence AP ranking. Existing evidence supports utilization associations and oracle savings ceilings only, not a prospective selector or speedup. Eventually required: an approved same-checkpoint test of a frozen selector with contemporaneous standard timing and actual output preservation/recovery accounting.

## Candidate 3 — geometry changes

Problem: some accepted blends worsen annotated IoU, but few rows/images are affected and no .25/.50 recovery is attributable to geometry. A hypothesis such as more reliable edge/coordinate consensus would require inference-available local/global geometry and possibly image evidence. Ground-truth-best-IoU selection is only an oracle and cannot be deployed. AP75 could change, with uncertain AP50/recall/FP effects; no gain is predicted.

Changing blend arithmetic is easy, validating it is harder because confident rows, collision limits and ranking must stay protected. CPU arithmetic cost is small; additional model views are not free. Risks include anatomical/context errors, worse boxes and instability for large/overlapping targets. Saved proposals establish frequency and direction of errors, but not a reliable replacement. Defer this candidate until an approved geometry-specific hypothesis has enough affected examples and a controlled localization evaluation; a blend-weight search is not justified here.

## Candidate 4 — bounded fallback coverage

Problem: a large fraction of missed fracture targets has no adequate eligible anchor. Hypothesis: a class-agnostic, image-only proposal derived from radiograph foreground extent or coarse anatomical position could cover some anchor-free regions. Such a mechanism must use image pixels/dimensions and optional global outputs only; it may not use annotation centers, error maps or patient-linked labels. No proposal code, image study, threshold or crop rule is supplied here.

An image-only proposal is conceptually implementable; its sensitivity to acquisition orientation, text/metal, casts, growth plates and normal anatomy is unknown. Foreground coverage does not identify pathology. Saved bundles cover only existing ROIs; 7/10 additional missed-target capacities from unused local evidence do not characterize new image-only crops, and GT-centered oracle coverage is not feasible inference performance. Expected recall/AP/FP direction and numerical gain are unverified. Extra crops would usually add compute and unmatched rows; any future fallback must replace a local slot under the same budget, not silently increase calls. Complexity and validation risk are higher than explicit fusion checks. A separately approved proposal-coverage feasibility study using training/development data, followed by frozen actual inference, would be needed before a performance claim. Keep this as a later question rather than a current implementation recommendation.

## Proposed research framework, requiring student approval

- **Development checkpoint:** prefer the unchanged standard epoch-22 EMA (SHA-256 `31b9b54c3c6cdf5cbd8110b312691e06db3e40def6072eeea5e8cfd687bc2f8b`) for an initial bounded V3 design. It has higher fracture AP50/AP75/AP50–95 than the rarity-only checkpoint and is the established baseline reference. This avoids simultaneously changing training and inference and keeps the fracture-preservation question clear. It is a development preference, not final inference selection or proof that standard training is generally superior. The rarity epoch-26 checkpoint remains a later common-rule robustness comparison if separately approved; its overall numerical lead does not establish fracture preservation.
- **Main target:** suppress unjustified promotions, then assess ROI utility. Preserve global initialization/pixels, original-coordinate restoration, target-class semantics, deterministic membership, per-view numerical checks, shared metric implementation, fixed checkpoint, and auditable before/after identities. Keep protected global fields and no-deletion behavior unless a separate approved design explicitly revisits them. Evidence ownership and confidence validation may be proposed for revision; no revised rule is frozen here.
- **Budget:** retain at most two local views plus one global call (maximum three per image). A possible fallback uses an existing local slot. The historical success gate remains mean complete-path latency ≤2× contemporaneous standard inference; maximum calls and mean latency are different constraints. Measure loading/warmup separately, sequential actual calls, full pipeline latency, and matching runtime boundaries. Do not infer controlled speed from historical sessions or measured memory alone.
- **Scientific gates:** require positive official eight-class mAP50–95 change with a paired 95% lower bound above zero; nondecreasing overall mAP50; fracture AP50/AP75/AP50–95 point preservation and the historical −.005 absolute lower-interval tolerance; overall and fracture FP count ratios ≤1.10 at .25/.50; and the ≤2× cost gate. The −.005 and 10% values are historical engineering criteria, not clinically established noninferiority/safety margins. They are not relaxed. Any proposed replacement criterion requires justification and student approval before implementation.
- **Comparison:** fixed same checkpoint and patient population, Standard versus frozen V2 versus one approved V3 rule, shared original-coordinate outputs/evaluator and unchanged confidence/IoU reporting. Report all-class and class-specific AP50/AP75/AP50–95, TP/FP/FN, precision/recall, recovery/loss, geometry effects, calls/ROI productivity and latency. No architecture or training change should accompany the first inference study. Preserve failed outcomes rather than selecting the largest mAP number.
- **Uncertainty and tie policy:** reuse paired patient clusters/all views; report effective replicate counts and undefined classes. Before any future confirmatory study, explicitly approve and lock a deterministic tie convention for all compared methods, keeping historical official A–F results labeled and unchanged. Recompute contemporaneous comparator metrics under that future common policy only after approval; do not silently splice a corrected V3 score into historical A–F. Record sensitivity separately if warranted, and avoid significance claims from ordering-scale changes.
- **Overfitting control:** the validation set has already been inspected repeatedly; subdividing it now does not make it independent. Preregister the scientific hypothesis, limited design choices and stopping rules before any V3 performance run; avoid threshold searches and post-hoc success criteria. Patient-grouped cross-fitting may help exploratory robustness but cannot undo prior validation reuse. Independent additional development evidence would be stronger and requires approval. Keep the frozen test set untouched until a separately approved final protocol; never use it to iterate on V3. Supervisor/institutional scope approval remains pending.

## Decision

The evidence supports a bounded design discussion about selective fusion and ROI utility. It does not support a promise of improvement, a parameter search, an immediate fallback implementation or another training run. If no inference-available evidence-validation hypothesis can be articulated without fitting the repeatedly used validation outcomes, stop rather than extending experiments indefinitely.

**CONDITIONAL GO for designing CG-ALR V3; no authorization to implement, run or select it.**
"""
    for name, content in [('PHASE_5A_ANALYSIS_REPORT.md', synthesis), ('CG_ALR_V2_DIAGNOSTICS.md', mechanism),
                          ('V3_FEASIBILITY_AND_RECOMMENDATIONS.md', recommendations)]:
        with (PUBLIC/name).open('x', encoding='utf-8') as stream:
            stream.write(content)
    print('Created the three aggregate-backed Phase 5A Markdown reports', flush=True)


if __name__ == '__main__':
    create_reports()
