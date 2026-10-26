"""Short scientific reports derived only from verified final aggregates."""

import json
import re

from .execution import RUN, RESULTS, locked, require
from .protocol import read, sha, write_new
from .test_analysis import METHODS
from .test_data import records
from wrist_eval.common import CLASSES


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def ci_text(value, scale=100):
    return f"[{value['lower']*scale:+.4f}, {value['upper']*scale:+.4f}] (n={value['replicates_present']})" if value['replicates_present'] else 'undefined (n=0)'


def render(data):
    test, val = data['test'], data['validation']
    point_rows, fracture_rows, runtime_rows = [], [], []
    for method in METHODS:
        ap = test['AP'][method]; op = test['operating'][method]['overall']; frac = test['operating'][method]['fracture']
        point_rows.append([method, *[f"{ap['overall'][k]:.8f}" for k in ('map50_95', 'map50', 'map75')],
            op['tp'], op['fp'], op['fn'], f"{op.get('precision', 0)*100:.3f}", f"{op.get('recall', 0)*100:.3f}"])
        fracture_rows.append([method, *[f"{ap['fracture'][k]:.8f}" for k in ('ap50', 'ap75', 'ap50_95')],
            frac['tp'], frac['fp'], frac['fn'], f"{frac.get('precision', 0)*100:.3f}", f"{frac.get('recall', 0)*100:.3f}"])
        value = test['runtime'][method]; timing = value['timing']
        base = 'final_A' if method in METHODS[:3] else 'final_D'
        ratio = timing['e2e_seconds']['mean']/test['runtime'][base]['timing']['e2e_seconds']['mean']
        runtime_rows.append([method, *[f"{timing['e2e_seconds'][k]*1000:.2f}" for k in ('mean', 'median', 'p95')],
            value['usage']['calls'], f"{timing['forward_passes']['mean']:.4f}",
            f"{timing['allocated_peak_bytes']['max']/2**20:.2f}/{timing['reserved_peak_bytes']['max']/2**20:.2f}",
            f'{ratio:.3f}', f"{value['mean_paired_latency_ratio']:.3f}" if 'mean_paired_latency_ratio' in value else '—'])
    class_rows = [[c, test['support'][c]['objects'], test['support'][c]['patients'],
                   *[f"{test['AP'][m][c]['ap50_95']:.6f}" for m in METHODS]] for c in CLASSES]
    paired_rows = [[name, f"{classes['overall']['map50_95']*100:+.4f}",
        ci_text(test['bootstrap']['contrasts'][name]['overall']['map50_95']),
        f"{classes['fracture']['ap50_95']*100:+.4f}", ci_text(test['bootstrap']['contrasts'][name]['fracture']['ap50_95'])]
        for name, classes in test['AP_contrasts'].items()]
    recovery_rows = [[pair, sum(v['recovered'] for v in classes.values()), sum(v['lost'] for v in classes.values()),
                      classes['fracture']['recovered'], classes['fracture']['lost']] for pair, classes in test['recovery'].items()]
    gate_rows = [[method, name, 'PASS' if passed else 'FAIL'] for method, values in test['engineering_gates'].items() for name, passed in values['checks'].items()]
    results = f"""# Final held-out test results

Six fixed conditions completed on {test['images']} images, {test['patients']} patients and 7,361 annotations. `final_C` is the predefined primary proposed pipeline. Final C/F use V3; historical validation C/F used V2 and remain unchanged. [Execution freeze](execution_freeze.json), [data integrity](data_integrity.json) and [verification](verification.json) record provenance. AP is on a 0–1 scale; contrast intervals below are percentage points (pp).

## Overall and fracture performance

{table(['Condition', 'mAP50–95', 'mAP50', 'mAP75', 'TP', 'FP', 'FN', 'Precision %', 'Recall %'], point_rows)}

{table(['Condition', 'Fracture AP50', 'AP75', 'AP50–95', 'TP', 'FP', 'FN', 'Precision %', 'Recall %'], fracture_rows)}

Operating counts use confidence .25 / IoU .50. FP means unmatched to annotations, not a clinically confirmed false alarm.

## Class AP50–95 and support

{table(['Class', 'Objects', 'Patients', 'A', 'B', 'C', 'D', 'E', 'F'], class_rows)}

[AP.csv](AP.csv) contains AP50/AP75/AP50–95 for every class and both partitions. [support.csv](support.csv) also gives image support. Rare-class estimates require caution, especially the small bone-lesion cohort.

## Paired comparisons

{table(['Frozen contrast', 'Overall delta pp', '95% interval pp', 'Fracture delta pp', '95% interval pp'], paired_rows)}

All 500 patient resamples use seed 314159 and common draws across methods. Intervals are nominal percentile intervals; absent-class AP and affected eight-class means are undefined, with effective counts shown. The primary contrast is `v3_standard_training` (C−A). Secondary comparisons and interactions are unadjusted for multiplicity. [paired_AP.csv](paired_AP.csv) and [results.json](results.json) retain all class/metric intervals and operating-count uncertainty.

## Object recovery

{table(['Comparison', 'Recovered objects', 'Lost objects', 'Recovered fracture', 'Lost fracture'], recovery_rows)}

Recovery and loss refer to the same annotated objects under canonical operating-point matching. They are not counts of updated scores.

## Inference cost

{table(['Condition', 'Mean ms', 'Median ms', 'p95 ms', 'Calls', 'Mean calls', 'Peak alloc/res MiB', 'Session ratio vs standard', 'V3 paired ratio'], runtime_rows)}

Time includes decode through final output and CUDA completion. Loading, three synthetic warmups, export and offline metrics are excluded. Physical forwards are independently counted. Allocator peaks exclude driver/display memory. Separate A–F session ratios are descriptive; the V3 paired prefix uses its required global call without an extra pass. Fixed order and laptop GPU conditions limit timing comparisons. Complete private logs retain ROI/slice calls and model-state digests.

## Frozen engineering checks

{table(['Condition', 'Criterion', 'Result'], gate_rows)}

Every criterion is unchanged from Phase 5C. The FP allowances and fracture tolerance are engineering criteria, not clinical safety margins. No test pilot, repeat inference, parameter revision or new training was used. [verification.json](verification.json) covers independent counts/AP, bootstrap checks, all saved V3 replays and historical preservation.
"""
    compare_rows = [[m, f"{val['AP'][m]['overall']['map50_95']:.8f}", f"{test['AP'][m]['overall']['map50_95']:.8f}",
        f"{(test['AP'][m]['overall']['map50_95']-val['AP'][m]['overall']['map50_95'])*100:+.4f}",
        f"{val['AP'][m]['fracture']['ap50_95']:.8f}", f"{test['AP'][m]['fracture']['ap50_95']:.8f}"] for m in METHODS]
    effect_rows = [[name, f"{val['AP_contrasts'][name]['overall']['map50_95']*100:+.4f}",
        f"{test['AP_contrasts'][name]['overall']['map50_95']*100:+.4f}",
        ci_text(val['bootstrap']['contrasts'][name]['overall']['map50_95']),
        ci_text(test['bootstrap']['contrasts'][name]['overall']['map50_95'])] for name in test['AP_contrasts']]
    comparison = f"""# Validation and test comparison

Saved validation predictions were evaluated again with the exact Phase 5C content-order policy used for test. No validation inference was repeated and no historical report was overwritten. These references use V3 for final_C/final_F and therefore must not be substituted for historical V2 C/F.

{table(['Condition', 'Validation mAP50–95', 'Test mAP50–95', 'Test−val pp', 'Val fracture AP50–95', 'Test fracture AP50–95'], compare_rows)}

{table(['Contrast', 'Validation delta pp', 'Test delta pp', 'Val 95% pp', 'Test 95% pp'], effect_rows)}

Both cohorts use 500 patient-paired samples/seed314159, with cohort-specific patient draws and class-presence counts. Intervals compare methods within each partition; they are not intervals for the test-minus-validation difference. Signs and intervals show which observed patterns persist; a matching sign alone is not statistical confirmation. Different patient/class composition and sparse findings can shift scores. Do not interpret a cross-partition difference as a causal domain shift.

The main scientific interpretation is in [FINAL_SCIENTIFIC_ASSESSMENT.md](FINAL_SCIENTIFIC_ASSESSMENT.md). Neither this comparison nor the test results are used to change the fixed checkpoints, V3, thresholds or primary configuration.
"""
    primary = test['engineering_gates']['final_C']; failed = [k for k, passed in primary['checks'].items() if not passed]
    conclusion = ('The primary proposed pipeline meets all frozen engineering criteria on this held-out partition.' if primary['strict_success'] else
                  'The primary proposed pipeline does not meet all frozen engineering criteria. Failed checks: '+', '.join(failed)+'.')
    observations = []
    for name in ('training_standard', 'training_sahi', 'training_v3', 'sahi_standard_training', 'sahi_rarity_training', 'v3_standard_training', 'v3_rarity_training'):
        delta = test['AP_contrasts'][name]['overall']['map50_95']*100
        ci = test['bootstrap']['contrasts'][name]['overall']['map50_95']
        state = 'positive nominal interval' if ci['lower'] > 0 else 'negative nominal interval' if ci['upper'] < 0 else 'interval includes zero'
        observations.append(f'- {name}: overall {delta:+.4f} pp; {ci_text(ci)}; {state}.')
    tradeoffs = []
    for method, base in (('final_C', 'final_A'), ('final_F', 'final_D')):
        a, b = test['operating'][method]['overall'], test['operating'][base]['overall']
        pair = method+'-'+base; recovered = test['recovery'][pair]
        tradeoffs.append(f"{method} versus {base}: TP {a['tp']-b['tp']:+d}, FP {a['fp']-b['fp']:+d}; recovered {sum(v['recovered'] for v in recovered.values())}, lost {sum(v['lost'] for v in recovered.values())}; paired cost {test['runtime'][method]['mean_paired_latency_ratio']:.3f}×. Strict success: {test['engineering_gates'][method]['strict_success']}.")
    assessment = f"""# Final scientific assessment

{conclusion} This conclusion concerns the predefined standard-trained V3 pipeline, not whichever condition has the largest observed score. V3 remains the fixed final thesis algorithm; no new selection or optimization follows.

## Observed effects and uncertainty

{chr(10).join(observations)}

Rarity-only effects compare two fixed single-seed training runs. They do not establish an average sampler effect across training seeds. SAHI and V3 effects use the same corresponding checkpoint. Secondary intervals and interaction contrasts are descriptive and unadjusted for multiplicity.

{chr(10).join(tradeoffs)}

The full fracture AP50/AP75/AP50–95 and uncertainty are reported separately. An overall change does not by itself establish better fracture localization. FP changes are relative to incomplete dataset annotations and are not clinical error rates.

## Contribution and limits

V3 provides a deterministic, bounded local-inference method with protected evidence reservation, unique ownership, moderated score transfer and structural ROI filtering. Its implementation preserves original global boxes/classes and adds no unanchored detections. Measured AP, recovery, FP and cost determine whether that engineering contribution improves this dataset; the design alone is not evidence of better performance.

Validation references use the same final evaluator, preserving historical metrics separately. Test evaluation was performed once per condition after the code/protocol freeze. This is evidence on unseen patients from the same dataset, not external clinical validation. One training seed, rare-class support, annotation uncertainty, prior validation-based development, and fixed-order laptop GPU sessions remain limitations. Nominal bootstrap intervals do not remove those limits.

{conclusion} Stop after this study: no further training, tuning, method selection, additional test experiments or Chapter 5 figures were performed.
"""
    return {'FINAL_TEST_RESULTS.md': results, 'VALIDATION_TEST_COMPARISON.md': comparison, 'FINAL_SCIENTIFIC_ASSESSMENT.md': assessment}


def generate():
    locked(); require(read(RUN/'verification.json')['passed'], 'Verify before writing reports')
    data = read(RESULTS/'results.json')
    for name, text in render(data).items():
        with (RESULTS/name).open('x', encoding='utf-8') as stream:
            stream.write(text)
    # Only aggregate records may enter Git. Scan image identities and structured private fields.
    text = '\n'.join(p.read_text(encoding='utf-8') for p in RESULTS.iterdir() if p.suffix in ('.md', '.json', '.csv'))
    require(all(r['filestem'] not in text for split in ('test', 'validation') for r in records(split)), 'Private image identity entered publication')
    for forbidden in ('"patient_id"', '"filestem"', '"stem"', '"boxes"', '"predictions"'):
        require(forbidden not in text, 'Private structured fields entered publication')
    for path in RESULTS.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('https:', 'http:', '#')):
                require((path.parent/target).exists(), 'Broken report link')
    write_new(RESULTS/'publication_checks.json', {'passed': True, 'private_image_and_structured_record_scan_passed': True,
        'all_report_links_checked': True, 'artifacts_sha256': {p.name: sha(p) for p in sorted(RESULTS.iterdir()) if p.is_file()}})
    print('Three final reports and aggregate publication checks complete', flush=True)


if __name__ == '__main__':
    generate()
