"""Compact aggregate-backed V3 result and scientific assessment documents."""

from wrist_eval.common import CLASSES, sha
from .analysis import METHODS, PAIRS
from .protocol import LOCAL, PUBLIC, read, require, frozen


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def outcome(data):
    passed = [v['strict_success_both_tracks'] for v in data['scientific_gates'].values()]
    if all(passed):
        return 'SUCCESS'
    tradeoffs = []
    for condition, method, base, old in (('standard', 'V3_standard', 'A', 'C'), ('rarity', 'V3_rarity', 'D', 'F')):
        new_point = data['operating_points'][method]['conf0.25_iou0.5']
        old_point = data['operating_points'][old]['conf0.25_iou0.5']
        recovery = data['changes'][method+'-'+base]['recovery']['0.5']['classes']
        tradeoffs.append(new_point['total_fp'] < old_point['total_fp'] and sum(v['recovered'] for v in recovery.values()) > 0)
    return 'PARTIAL IMPROVEMENT' if any(passed) or all(tradeoffs) else 'NO DEMONSTRATED IMPROVEMENT'


def generate():
    frozen(); require(read(LOCAL/'verification.json')['passed'], 'Verify before reporting')
    data = read(PUBLIC/'results.json'); verdict = outcome(data)
    native, stable = data['AP']['historical'], data['AP']['deterministic']
    ap_rows = [[track, method, *[f"{values[method]['overall'][k]:.8f}" for k in ('map50', 'map75', 'map50_95')],
                *[f"{values[method]['fracture'][k]:.8f}" for k in ('ap50', 'ap75', 'ap50_95')]]
               for track, values in data['AP'].items() for method in METHODS]
    ap_table = table(['Track', 'Method', 'mAP50', 'mAP75', 'mAP50–95', 'Fracture AP50', 'Fracture AP75', 'Fracture AP50–95'], ap_rows)
    intervals = []
    for track, values in data['AP'].items():
        for pair, (other, base) in PAIRS.items():
            for name, metric in (('overall', 'map50_95'), ('fracture', 'ap50_95')):
                point = (values[other][name][metric]-values[base][name][metric])*100
                ci = data['AP_intervals'][track][pair][name][metric]
                intervals.append([track, pair, name, f'{point:+.5f}', f"[{ci['lower']*100:+.5f}, {ci['upper']*100:+.5f}]", ci['replicates_present']])
    ci_table = table(['Track', 'Contrast', 'Metric', 'Delta, pp', 'Paired 95%, pp', 'Valid replicates'], intervals)
    point_rows = []
    for method in METHODS:
        point = data['operating_points'][method]['conf0.25_iou0.5']; tp, fp, fn = [point['total_'+k] for k in ('tp', 'fp', 'fn')]
        point_rows.append([method, tp, fp, fn, f'{100*tp/(tp+fp):.3f}', f'{100*tp/(tp+fn):.3f}',
                           point['classes']['fracture']['tp'], point['classes']['fracture']['fp']])
    point_table = table(['Method', 'TP', 'FP', 'FN', 'Precision, %', 'Recall, %', 'Fracture TP', 'Fracture FP'], point_rows)
    error_rows = read(PUBLIC/'error_categories.json')
    error_table = table(['Method', 'Background/unannotated FP', 'Localization FP', 'Duplicate FP', 'Classification FP'],
        [[method, *[sum(v['fp_errors'].get(k, 0) for v in error_rows if v['method'] == method)
                   for k in ('background_or_unannotated', 'localization', 'duplicate', 'classification')]] for method in METHODS])
    recovery_rows = []
    for pair in PAIRS:
        values = data['changes'][pair]['recovery']['0.5']['classes']
        recovery_rows.append([pair, sum(v['recovered'] for v in values.values()), sum(v['lost'] for v in values.values()),
                              values['fracture']['recovered'], values['fracture']['lost']])
    recovery_table = table(['Contrast', 'Recovered objects', 'Lost objects', 'Recovered fracture', 'Lost fracture'], recovery_rows)
    runtime_rows = []
    for method, value in data['runtime'].items():
        t = value['timing']; hist = value['roi_count_images']
        runtime_rows.append([method, value['usage']['calls'], f"{t['forward_passes']['mean']:.5f}",
            *[f"{t['e2e_seconds'][k]*1000:.3f}" for k in ('mean', 'median', 'p95')],
            f"{t['paired_global_e2e_seconds']['mean']*1000:.3f}", f"{value['mean_paired_latency_ratio']:.4f}",
            f"{t['allocated_peak_bytes']['max']/2**20:.2f} / {t['reserved_peak_bytes']['max']/2**20:.2f}",
            ' / '.join(str(hist[str(i)]) for i in range(3))])
    runtime_table = table(['Method', 'Calls', 'Mean calls', 'Mean ms', 'Median ms', 'p95 ms', 'Paired global ms', 'Ratio', 'Peak alloc/res MiB', '0/1/2 ROI images'], runtime_rows)
    promotion_rows, roi_rows = [], []
    for method, value in data['promotion_and_ROI'].items():
        counts = value['outcomes']; roi = value['roi_productivity']
        promotion_rows.append([method, *[counts.get(k, 0) for k in ('recovered_TP', 'reassigned_TP', 'promoted_FP', 'below_threshold', 'inherited_TP_to_FP', 'inherited_FP_to_TP')]])
        roi_rows.append([method, *[roi.get(k, 0) for k in ('local_calls', 'rois_with_update', 'rois_with_recovery', 'rois_with_promoted_FP', 'rois_both_recovery_and_FP', 'rois_without_update')]])
    promotion_table = table(['Method', 'Promoted recovered TP', 'Reassigned TP', 'Promoted FP', 'Update below .25', 'Inherited TP→FP', 'Inherited FP→TP'], promotion_rows)
    roi_table = table(['Method', 'Local calls', 'ROIs with update', 'ROIs with recovery', 'ROIs with promoted FP', 'ROIs with both', 'ROIs without update'], roi_rows)
    gate_rows = []
    for condition, result in data['scientific_gates'].items():
        for track in ('historical', 'deterministic'):
            for name, passed in result[track]['checks'].items():
                gate_rows.append([condition, track, name, 'PASS' if passed else 'FAIL'])
    gate_table = table(['Checkpoint', 'Track', 'Frozen criterion', 'Result'], gate_rows)
    attribution_rows = []
    for condition, cells in data['component_attribution'].items():
        for name, cell in cells.items():
            attribution_rows.append([condition, name, f"{cell['deterministic_AP']['overall']['map50_95']:.8f}",
                                    f"{cell['deterministic_AP']['fracture']['ap50_95']:.8f}", cell['tp'], cell['fp'], cell['fn']])
    attribution_table = table(['Checkpoint', 'Predefined cell', 'Stable mAP50–95', 'Stable fracture AP50–95', 'TP', 'FP', 'FN'], attribution_rows)
    result_text = f"""# CG-ALR V3 — experimental results

One frozen V3.0 algorithm, two predeclared checkpoints, 3,088 validation images/914 patients/7,204 objects. These are exploratory validation findings, not independent confirmation. Numerical configuration and both design copies were committed/pushed before V3 performance evaluation; [freeze](freeze.json), [execution manifest](execution_manifest.json) and [verification](verification.json) bind the inputs and code. Historical A–F, Phase 5A and all dataset/checkpoint files remain unchanged.

## Two AP tracks

{ap_table}

Historical uses the unchanged pinned matcher/evaluator and original final row order. Deterministic sensitivity restores original global-row order before the same native per-image matching and uses descending unchanged score, frozen image index and original global ID for AP ties. It retains the same ten IoUs and integration. This second track does not overwrite historical results. Inference has no GT input; annotation matching is post-inference evaluation only. All eight classes and both tracks are in [AP.csv](AP.csv).

## Paired uncertainty

{ci_table}

500 identical patient draws, seed 314159, all views/examinations with multiplicity. Historical A/C/D/F replicates are reused after hash/draw and representative AP checks; all new deterministic-track/V3 replicates are calculated. Bone-lesion/overall intervals use 492 draws; other classes 500. Undefined absent-class AP is not zero and overall never averages fewer than eight classes. [All class/metric contrasts](paired_AP_contrasts.csv) include effective support. Nominal percentile intervals do not account for validation reuse, seed variation or multiplicity.

## Operating detections and object recovery

Confidence .25 / IoU .50, unchanged descriptive one-to-one matcher; FP is annotation-relative unmatched prediction, not a clinical adjudication.

{point_table}

{recovery_table}

[operating_points.csv](operating_points.csv) retains confidence .05/.25/.50 at IoU .50/.75 for every class and method. Counts and source correspondences use their original definitions; recovered/lost targets must not be confused with introduced/removed unmatched predictions. Paired operating intervals are retained in [results.json](results.json). Residual GT and unmatched-prediction categories are in [error_categories.json](error_categories.json), with annotated-object/patient support:

{error_table}

## Confidence decisions and ROI utilization

{promotion_table}

{roi_table}

An updated row that becomes TP can reassign an already-detected object, displacing an inherited row. Those events are separate from new recovery. ROI update count is not a utility score: harmful promotions and mixed recovery/FP ROIs are explicitly shown. [All rejection/ownership counts](promotion_and_ROI.json) describe protected conflicts, reserved evidence, ambiguous graph degrees and sequential conflicts.

## Actual inference cost

{runtime_table}

Each image uses one required global forward and at most two local forwards. A model-level hook independently reconciles per-view call counters, including three synthetic warmups and three first-pilot-image replays recorded separately. Full population counts exclude warmups; no hidden extra pass supplies the paired control. Pilot/full predictions agree exactly, all new globals reproduce saved A/D, model state is unchanged, and no technical rerun is used to improve timing.

E2E begins before decode and ends after fusion/checks/CUDA synchronization. The paired standard prefix ends after required global prediction/restoration, before reference-array equality checking and local planning. Export, offline replay/GT/AP, loading and warmups are excluded. Per-image logs and outer loop times remain private; [runtime.csv](runtime.csv) records mean/median/p95 and allocator memory. PyTorch peaks exclude driver/display allocations. Separate historical A/C/D/F session times are not controlled speed comparisons.

## Frozen scientific gates

{gate_table}

SUCCESS requires every original primary criterion and stable-track corroboration on both checkpoints. Original secondary trade-off flags are retained in JSON for historical context but do not replace the strict primary criteria. The 10% FP allowance and −.005 fracture interval tolerance are engineering limits, not clinical safety or noninferiority margins.

## Predeclared component attribution

{attribution_table}

All cells use original global coordinates and deterministic AP. Historical-view versus V3-view legacy V1 cells isolate available views with the same frozen score/ownership rule. The next step changes arbitration with the legacy cap held fixed; the last step adds midpoint moderation. These are descriptive CPU replays, not candidate V3 configurations or post-hoc selection. They require no additional GPU inference, do not provide independent confirmation, and are not substituted for the official V2 geometry comparator.

**Overall classification: {verdict}.** See the [scientific assessment](V3_SCIENTIFIC_ASSESSMENT.md) for interpretation and remaining uncertainty.
"""
    assessments = []
    for condition, method, base, old in (('standard', 'V3_standard', 'A', 'C'), ('rarity', 'V3_rarity', 'D', 'F')):
        point = data['operating_points'][method]['conf0.25_iou0.5']; old_point = data['operating_points'][old]['conf0.25_iou0.5']
        baseline = data['operating_points'][base]['conf0.25_iou0.5']
        recovery = data['changes'][method+'-'+base]['recovery']['0.5']['classes']
        recovered = sum(v['recovered'] for v in recovery.values()); lost = sum(v['lost'] for v in recovery.values())
        native_delta = (native[method]['overall']['map50_95']-native[base]['overall']['map50_95'])*100
        stable_delta = (stable[method]['overall']['map50_95']-stable[base]['overall']['map50_95'])*100
        frac_delta = (stable[method]['fracture']['ap50_95']-stable[base]['fracture']['ap50_95'])*100
        frac_state = 'improves' if frac_delta > 0 else 'declines' if frac_delta < 0 else 'is unchanged'
        assessments.append(f"### {condition.capitalize()} checkpoint\n\nV3 mAP50–95 versus standard changes by {native_delta:+.5f} pp historically and {stable_delta:+.5f} pp under deterministic ties. Stable fracture AP50–95 {frac_state} ({frac_delta:+.5f} pp); all fracture AP50/AP75/AP50–95 and uncertainty gates are reported separately. At .25/.50 it recovers {recovered} objects and loses {lost}. FP is {point['total_fp']} versus standard {baseline['total_fp']} and V2 {old_point['total_fp']}; V3−V2 is {point['total_fp']-old_point['total_fp']:+d}. Strict success on both metric tracks: **{data['scientific_gates'][condition]['strict_success_both_tracks']}**. No checkpoint-specific rule was changed after these observations.")
    assessment = f"""# CG-ALR V3 — scientific assessment

**{verdict}** under the predeclared interpretation. This is an exploratory follow-up, not final inference selection. [Detailed results](V3_EXPERIMENTAL_RESULTS.md), [design](V3_DESIGN.md), [aggregate evidence](results.json) and [verification](verification.json) separate measurements from mechanisms and hypotheses.

## What changed

V3 reserves all local evidence compatible with original protected detections, accepts only isolated one-to-one weak/local graph edges, and uses capped equal-midpoint score transfer. Sequential overlap arbitration protects current operating rows from later overlapping updates. The ROI planner retains V2 crop/rank/diversity/coalescing rules but excludes candidates that the same V3 protected-overlap rule already makes ineligible. Geometry is frozen to original global boxes. No new rows, learned calibration, extra detector, fallback proposals, training or threshold optimization are introduced.

This directly addresses V2's ambiguous evidence ownership and correlated positive confirmation. It does not make the model scores calibrated probabilities or establish that a reserved/overlapping row is clinically correct. Unique correspondence is an observable constraint, not proof of correctness. Original-class/box/row-count preservation and exact global reproduction pass for every validation image.

{chr(10).join(assessments)}

## Recovery, harmful updates and efficiency

Promotion, reassignment and inherited TP/FP transitions are separately counted in the result report; an accepted match or changed score is not automatically useful. Compare true recovered/lost annotations as well as FP counts. Fewer false promotions can sacrifice recoveries or lower AP, so the strict AP/fracture/FP gates remain binding. The PARTIAL classification, when present, denotes only the frozen engineering trade-off definition; it is not a claim that every scientific criterion passed.

ROI utilization records actual calls, no-update calls, recovery-producing calls, FP-producing calls and their overlap. Any reduction in total calls is structural pruning, not evidence that reordering alone found previously uncovered fractures. The four predeclared replay cells isolate view availability, arbitration and midpoint effects with geometry held constant; they explain mechanisms without selecting a superior alternative. Official frozen V2 includes its original geometry and remains unchanged.

## AP ranking and measurement artifacts

V3 changes confidence rankings of existing global rows, not localization or detection coverage below the retained .001 baseline floor. Historical pinned AP may also vary because the global final order affects unstable equal-confidence sorting. The deterministic sensitivity track uses immutable image/global IDs and original-ID matching input. Six non-target classes are verified invariant across standard/V2/V3 under that track. Historical-versus-stable differences are measurement sensitivity, not extra detector gains, and cannot turn a failed strict gate into success.

## Limits and stop

The algorithm cannot create unanchored missed findings or repair confident global errors; same-model correlated false patterns can still survive its guards. Protected overlap and strict graph uniqueness can reject legitimate nearby findings. Original coordinates deliberately leave localization defects unresolved. Bone lesion still has five objects/four patients, with undefined absent-class bootstrap samples. One training seed per checkpoint, validation-selected epochs, extensive prior validation use, fixed-order GPU sessions and annotation limitations constrain inference. Paired intervals remain conditional descriptive uncertainty; excluding zero does not make this independent confirmation or clinical validation.

The original success gates are reported individually for both checkpoints and both tracks, including the contemporaneous ≤2× cost and actual three-call bound. None was relaxed. Stop after this bounded study: no V4, extra optimization cycle, new training, final method selection or held-out evaluation is authorized.

**{verdict}.**
"""
    for name, text in (('V3_EXPERIMENTAL_RESULTS.md', result_text), ('V3_SCIENTIFIC_ASSESSMENT.md', assessment)):
        with (PUBLIC/name).open('x', encoding='utf-8') as stream:
            stream.write(text)
    # Implementation facts were committed before evaluation; append only runtime verification.
    with (PUBLIC/'V3_IMPLEMENTATION_REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\n## Completed runtime verification\n\nBoth fixed pilots and full runs completed. Model hooks, global-reference equality, pilot/full identity, all saved bundle replays, original coordinates/protected fields, paired patient intervals and historical preservation pass. Results and criteria are in [the experimental report](V3_EXPERIMENTAL_RESULTS.md); [verification.json](verification.json) binds the aggregates. The algorithm/configuration remained frozen.\n')
    print('Created V3 result and scientific assessment reports: '+verdict, flush=True)


if __name__ == '__main__':
    generate()
