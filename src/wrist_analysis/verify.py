"""Independent checks of additive Phase 5A aggregates and preservation."""

from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np

from .evidence import ROOT, PUBLIC, CLASSES, read, sha, require, preservation, install_guard, write_new


def independent_match(gt, pred):
    # Separate float64 IoU and greedy assignment implementation for important totals.
    gt = np.asarray(gt, np.float64); pred = np.asarray(pred, np.float64)
    target = np.full(len(gt), -1, int); proposal = np.full(len(pred), -1, int)
    for p in sorted(range(len(pred)), key=lambda i: (-pred[i, 4], i)):
        if pred[p, 4] < .25:
            continue
        candidates = []
        for g in range(len(gt)):
            if target[g] >= 0 or gt[g, 4] != pred[p, 5]:
                continue
            a, b = gt[g, :4], pred[p, :4]
            inter = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-7
            value = inter/union
            if value >= .5:
                candidates.append((value, -g))
        if candidates:
            g = -max(candidates)[1]; target[g] = p; proposal[p] = g
    return target, proposal


def run(attempt='analysis_02', receipt=None):
    install_guard()
    directory = ROOT/'outputs/phase_5a_analysis'/attempt
    receipt_path = PUBLIC/'verification.json' if receipt is None else (ROOT/receipt).resolve()
    require(receipt is None or receipt_path.is_relative_to(ROOT/'outputs/phase_5a_analysis'), 'Repeat audits must stay private')
    require(not receipt_path.exists(), 'Preserve the previous verification receipt')
    before = read((directory/'preservation_before.json').relative_to(ROOT).as_posix())
    manifest = read('docs/evidence/phase_5a_analysis/source_manifest.json')
    for name, expected in manifest['artifacts_sha256'].items():
        require(sha(ROOT/name) == expected, 'Analysis input changed: '+name)
    current = preservation()
    require(current['private'] == before['private'] and all(current['tracked'].get(name) == expected
        for name, expected in before['tracked'].items()), 'Historical scientific artifacts or tracked files changed')
    aggregate = read((directory/'aggregate.json').relative_to(ROOT).as_posix())
    official = read('docs/evidence/rarity_inference_comparison/results.json')
    require(sha(directory/'aggregate.json') == manifest['private_aggregate_sha256'], 'Private aggregate changed')
    for name, value in (('error_categories.json', aggregate['errors']), ('v2_mechanism.json', aggregate['mechanism'])):
        require(read('docs/evidence/phase_5a_analysis/'+name) == value, 'Published aggregate differs')
    for name, values in (('six_condition_summary.csv', aggregate['synthesis']['six_conditions']),
        ('paired_AP_contrasts.csv', aggregate['synthesis']['paired_AP_contrasts']),
        ('map_components.csv', aggregate['synthesis']['map_components']), ('recovery.csv', aggregate['recovery']),
        ('error_strata.csv', aggregate['strata'])):
        with (PUBLIC/name).open(encoding='utf-8', newline='') as stream:
            actual = list(csv.DictReader(stream))
        require(actual == [{k: str(v) for k, v in row.items()} for row in values], 'Published CSV differs')
    # Reconstruct only validation labels through the already-verified loader.
    from .evidence import load_inputs
    _, rows, predictions, bundles, _ = load_inputs()
    totals = {}
    matched = {}
    for condition, by_stem in predictions.items():
        tp = fp = fn = 0; matched[condition] = {}
        for stem, row in rows.items():
            g, p = independent_match(row['gt'], by_stem[stem])
            tp += int(sum(g >= 0)); fn += int(sum(g < 0))
            fp += int(sum((by_stem[stem][:, 4] >= .25) & (p < 0)))
            matched[condition][stem] = (g, p)
        target = official['conditions'][condition]['operating_points']['conf0.25_iou0.5']
        require((tp, fp, fn) == tuple(target['total_'+k] for k in ('tp', 'fp', 'fn')), 'Independent detection totals differ')
        totals[condition] = {'tp': tp, 'fp': fp, 'fn': fn}
    for condition, base in (('C', 'A'), ('F', 'D')):
        data = aggregate['mechanism'][condition]; counts = data['counts']
        roi_hist = Counter(len(v['plan']['rois']) for v in bundles[condition])
        require({str(k): v for k, v in roi_hist.items()} == data['roi_count_images'], 'ROI histogram differs')
        require(sum(k*v for k, v in roi_hist.items()) == counts['local_calls'], 'Model-call denominator differs')
        require(counts['geometry_accepted'] == (64 if condition == 'C' else 90), 'Geometry count differs')
        recovered = sum(int(((matched[base][stem][0] < 0) & (matched[condition][stem][0] >= 0)).sum()) for stem in rows)
        lost = sum(int(((matched[base][stem][0] >= 0) & (matched[condition][stem][0] < 0)).sum()) for stem in rows)
        require(recovered == counts['updated_recovered_TP'] and lost == 0, 'Recovery attribution differs')
        require(counts['updated_FP']+counts['inherited_tp_to_fp']-counts['inherited_fp_to_tp'] == totals[condition]['fp']-totals[base]['fp'], 'FP promotion partition differs')
        for name, values in data['oracle_capacities'].items():
            order = [values[k] for k in ('all_retained_weak_global', 'eligible_global_band', 'selected_roi_covered', 'fixed_cap_reachable')]
            require(order == sorted(order, reverse=True), 'Oracle capacities do not respect nested constraints')
            support = official['conditions'][base]['operating_points']['conf0.25_iou0.5']['classes'][name]['fn']
            require(sum(data['missed_target_stages'][name].values()) == support, 'Missed-object stages do not partition support')
        events = read((directory/(condition+'_private_events.json')).relative_to(ROOT).as_posix())
        # Average-rank U is independent of the pairwise comparison used by the analyzer.
        for row in data['promotion_feature_separation']:
            values = [v for v in events['updates'] if row['class'] == 'overall' or v['class'] == row['class']]
            a = [v[row['feature']] for v in values if v['outcome'] == 'recovered_TP']
            b = [v[row['feature']] for v in values if v['outcome'] == 'FP']
            combined = np.asarray(a+b); unique = sorted(set(combined)); rank = 1; rank_sum = 0.
            for value in unique:
                na, nb = a.count(value), b.count(value); size = na+nb
                rank_sum += na*(rank+(size-1)/2); rank += size
            estimate = (rank_sum-len(a)*(len(a)+1)/2)/(len(a)*len(b))
            require(abs(estimate-row['probability_TP_feature_exceeds_FP']) < 1e-12, 'Feature separation differs')
    public_text = '\n'.join(p.read_text(encoding='utf-8') for p in PUBLIC.iterdir() if p.is_file() and p.suffix in ('.md', '.csv', '.json'))
    require(all(stem not in public_text for stem in rows), 'A private image identifier entered public evidence')
    require('"patient_id"' not in public_text and '"stem"' not in public_text, 'Private record fields entered public evidence')
    for path in PUBLIC.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('http:', 'https:', '#')):
                require((path.parent/target.split('#')[0]).exists(), 'Missing report link: '+target)
    write_new(receipt_path, {'passed': True, 'independently_reproduced_operating_totals': totals,
        'paired_AP_intervals_reused_not_recomputed': True, 'independent_feature_separation_rank_check': True,
        'all_C_F_recovery_FP_and_ROI_partitions_reconciled': True, 'all_nested_oracle_bounds_reconcile': True,
        'public_identifiers_and_report_links_checked': True,
        'protected_tracked_files_unchanged': len(before['tracked']), 'protected_private_files_unchanged': len(before['private']),
        'checkpoint_protection': 'size and mtime only; checkpoint files never opened',
        'model_loads': 0, 'model_forwards': 0, 'heldout_image_label_prediction_files_opened': 0,
        'source_manifest_sha256': sha(PUBLIC/'source_manifest.json'),
        'analysis_source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT/'src/wrist_analysis').glob('*.py'))},
        'public_artifacts_sha256': {p.name: sha(p) for p in sorted(PUBLIC.iterdir()) if p.is_file()}})
    print('Independent Phase 5A counts, separation, support, privacy, links and preservation checks pass', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', default='analysis_02')
    parser.add_argument('--receipt', help='New private receipt path for an intentional repeated audit')
    args = parser.parse_args()
    run(args.attempt, args.receipt)
