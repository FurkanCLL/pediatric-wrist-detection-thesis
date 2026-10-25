"""Verify and publish aggregate V3 evidence without private linkage or coordinates."""

from collections import defaultdict
import csv
import json
import re

import numpy as np

from wrist_eval.common import CLASSES, sha, identity
from wrist_eval.analyze import truth
from wrist_eval.bootstrap import patient_groups, resample_indices
from wrist_analysis.verify import independent_match
from wrist_compare.common import install_access_guard
from .analysis import load_inputs, METHODS, PAIRS
from .metrics import ap_delta
from .protocol import ROOT, LOCAL, PUBLIC, read, write_new, frozen, preserve, require, source_identity


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def compact_points(points):
    return {key: {'confidence': v['confidence'], 'iou': v['iou'],
        **{k: v[k] for k in ('total_tp', 'total_fp', 'total_fn')},
        'classes': {name: {k: c[k] for k in ('objects', 'patients', 'images', 'tp', 'fp', 'fn', 'precision', 'recall')}
                    for name, c in v['classes'].items()}} for key, v in points.items()}


def verification():
    install_access_guard()
    cfg, _, _ = frozen()
    data = read(LOCAL/'analysis/results.json')
    _, _, rows, predictions, ids, bundles, summaries, official = load_inputs()
    require(data['freeze_sha256'] == sha(PUBLIC/'freeze.json'), 'Results belong to another freeze')
    totals = {}
    for method in METHODS:
        tp = fp = fn = 0
        for row in rows:
            pred = predictions[method][row['filestem']]
            g, p = independent_match(truth(row), pred)
            tp += int(sum(g >= 0)); fn += int(sum(g < 0)); fp += int(sum((p < 0) & (pred[:, 4] >= .25)))
        point = data['operating_points'][method]['conf0.25_iou0.5']
        require((tp, fp, fn) == tuple(point['total_'+k] for k in ('tp', 'fp', 'fn')), 'Independent V3 study counts differ')
        totals[method] = {'tp': tp, 'fp': fp, 'fn': fn}
    for condition, base, method in (('standard', 'A', 'V3_standard'), ('rarity', 'D', 'V3_rarity')):
        records = bundles[method]; summary = summaries[method]
        calls = sum(record['passes'] for record in records)
        require(calls == summary['usage']['calls'] and all(1 <= record['passes'] <= 3 for record in records), 'Physical call accounting differs')
        require(summary['physical_model_calls_including_warmups_replays'] == calls+3, 'Full physical calls/warmups differ')
        for row, record in zip(rows, records):
            original = predictions[base][row['filestem']]; pred = predictions[method][row['filestem']]
            order = ids[method][row['filestem']]
            require(np.array_equal(pred[:, :4], original[order, :4]) and np.array_equal(pred[:, 5], original[order, 5]), 'V3 geometry/class changed')
            inverse = np.argsort(order); protected = original[:, 4] >= .25
            require(np.array_equal(pred[inverse][protected], original[protected]), 'Protected output changed')
        outcome = data['promotion_and_ROI'][method]['outcomes']
        require(outcome.get('promoted_FP', 0)+outcome.get('inherited_TP_to_FP', 0)-outcome.get('inherited_FP_to_TP', 0)
                == totals[method]['fp']-totals[base]['fp'], 'Promotion/FP partition differs')
        recovery = data['changes'][method+'-'+base]['recovery']['0.5']['classes']
        require(sum(v['recovered']-v['lost'] for v in recovery.values()) == totals[method]['tp']-totals[base]['tp'], 'Recovery/TP partition differs')
    path = LOCAL/'analysis/bootstrap_replicates.jsonl'
    require(sha(path) == data['bootstrap_replicates_sha256'], 'Bootstrap output changed')
    groups, rng = patient_groups(rows), np.random.default_rng(314159)
    collected = defaultdict(list); presence = defaultdict(int)
    with path.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream]
    require(len(records) == 500, 'Bootstrap replicate count differs')
    for rep, record in enumerate(records):
        require(record['replicate'] == rep and identity(resample_indices(groups, rng)) == record['indices_sha256'], 'Patient draw identity differs')
        for track in ('historical', 'deterministic'):
            for method in METHODS:
                for name in record['AP'][track][method]:
                    presence[track, method, name] += 1
            for pair, (other, base) in PAIRS.items():
                expected = ap_delta(record['AP'][track][other], record['AP'][track][base])
                require(expected == record['contrasts'][track][pair], 'Paired replicate arithmetic differs')
                for name, values in expected.items():
                    for metric, value in values.items():
                        collected[track, pair, name, metric].append(value)
    for (track, method, name), count in presence.items():
        require(count == (492 if name in ('overall', 'bonelesion') else 500), 'Conditional class support differs')
    for (track, pair, name, metric), values in collected.items():
        recorded = data['AP_intervals'][track][pair][name][metric]
        require(recorded['replicates_present'] == len(values)
            and recorded['lower'] == float(np.quantile(values, .025))
            and recorded['upper'] == float(np.quantile(values, .975)), 'Paired interval differs')
    preservation = preserve('after')
    receipt = {'passed': True, 'independent_operating_counts': totals, 'all_6176_V3_bundle_replays_exact': True,
        'immutable_source_ids_and_protected_geometry_checked': True, 'all_500_patient_draws_and_AP_intervals_checked': True,
        'conditional_overall_and_bonelesion_replicates': 492, 'other_class_replicates': 500,
        'actual_calls_and_all_recovery_FP_partitions_reconcile': True, 'preservation': preservation,
        'source_identity': source_identity(), 'freeze_sha256': sha(PUBLIC/'freeze.json'),
        'private_results_sha256': sha(LOCAL/'analysis/results.json'), 'bootstrap_sha256': sha(path),
        'new_training': False, 'test_image_label_prediction_contents_opened': 0}
    write_new(LOCAL/'verification.json', receipt)
    print('Independent V3 counts, physical calls, protected fields, paired intervals and preservation pass', flush=True)


def export():
    frozen()
    require(read(LOCAL/'verification.json')['passed'], 'Verification must precede publication')
    data = read(LOCAL/'analysis/results.json')
    require(sha(LOCAL/'analysis/results.json') == read(LOCAL/'verification.json')['private_results_sha256'], 'Verified results changed')
    points = {name: compact_points(value) for name, value in data['operating_points'].items()}
    public = {**data, 'operating_points': points}
    write_new(PUBLIC/'results.json', public)
    write_new(PUBLIC/'verification.json', read(LOCAL/'verification.json'))
    ap_rows, contrast_rows, point_rows, runtime_rows = [], [], [], []
    for track, methods in data['AP'].items():
        for method, classes in methods.items():
            for name, metrics in classes.items():
                ap_rows.append({'track': track, 'method': method, 'class': name,
                    **{k: metrics['m'+k if name == 'overall' else k] for k in ('ap50', 'ap75', 'ap50_95')}})
        for pair, (other, base) in PAIRS.items():
            delta = ap_delta(methods[other], methods[base])
            for name, metrics in delta.items():
                for metric, value in metrics.items():
                    interval = data['AP_intervals'][track][pair][name][metric]
                    contrast_rows.append({'track': track, 'comparison': pair, 'class': name, 'metric': metric, 'delta': value, **interval})
    for method, method_points in points.items():
        for key, point in method_points.items():
            for name, value in point['classes'].items():
                point_rows.append({'method': method, 'confidence': point['confidence'], 'iou': point['iou'], 'class': name, **value})
    for method, summary in data['runtime'].items():
        timing = summary['timing']
        runtime_rows.append({'method': method, 'images': summary['images'], 'calls': summary['usage']['calls'],
            'mean_calls': timing['forward_passes']['mean'], 'mean_ms': timing['e2e_seconds']['mean']*1000,
            'median_ms': timing['e2e_seconds']['median']*1000, 'p95_ms': timing['e2e_seconds']['p95']*1000,
            'paired_global_mean_ms': timing['paired_global_e2e_seconds']['mean']*1000,
            'paired_ratio': summary['mean_paired_latency_ratio'], 'peak_allocated_mib': timing['allocated_peak_bytes']['max']/2**20,
            'peak_reserved_mib': timing['reserved_peak_bytes']['max']/2**20})
    write_csv(PUBLIC/'AP.csv', ap_rows); write_csv(PUBLIC/'paired_AP_contrasts.csv', contrast_rows)
    write_csv(PUBLIC/'operating_points.csv', point_rows); write_csv(PUBLIC/'runtime.csv', runtime_rows)
    write_new(PUBLIC/'component_attribution.json', data['component_attribution'])
    write_new(PUBLIC/'promotion_and_ROI.json', data['promotion_and_ROI'])
    # Keep error mechanisms visible without publishing per-image matches.
    prior_errors = read(ROOT/'docs/evidence/phase_5a_analysis/error_categories.json')
    error_rows = [{'method': v['experiment'], 'class': v['class'], 'objects': v['objects'], 'patients': v['patients'],
                   'gt_errors': v['gt_errors'], 'fp_errors': v['fp_errors']}
                  for v in prior_errors if v['experiment'] in ('A', 'C', 'D', 'F')]
    for method in ('V3_standard', 'V3_rarity'):
        for name, value in data['operating_points'][method]['conf0.25_iou0.5']['classes'].items():
            error_rows.append({'method': method, 'class': name, **{k: value[k] for k in ('objects', 'patients', 'gt_errors', 'fp_errors')}})
    write_new(PUBLIC/'error_categories.json', error_rows)
    write_new(PUBLIC/'execution_manifest.json', {k: v for k, v in read(LOCAL/'manifest.json').items() if k != 'pilot'})
    for condition in ('standard', 'rarity'):
        for stage in ('pilot', 'full'):
            write_new(PUBLIC/f'{stage}_{condition}.json', read(LOCAL/f'{stage}_{condition}/summary.json'))
    print('V3 aggregate evidence published without patient/image linkage or coordinates', flush=True)


def publication_check():
    # Public fields are aggregate-only; scan against the complete frozen validation identity list.
    from .protocol import metadata
    rows, _ = metadata()
    texts = []
    for path in PUBLIC.iterdir():
        if path.is_file() and path.suffix in ('.json', '.md', '.csv'):
            text = path.read_text(encoding='utf-8'); texts.append(text)
            if path.suffix == '.json':
                json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    combined = '\n'.join(texts)
    require(all(row['filestem'] not in combined for row in rows), 'Private image identity entered publication')
    require('"patient_id"' not in combined and '"stem"' not in combined and '"predictions"' not in combined,
            'Detailed private records entered publication')
    for path in PUBLIC.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('https:', 'http:', '#')):
                require((path.parent/target.split('#')[0]).exists(), 'Report link is missing')
    write_new(PUBLIC/'publication_checks.json', {'passed': True, 'all_validation_identity_strings_excluded': True,
        'aggregate_JSON_finite': True, 'report_links_checked': True, 'historical_results_and_code_preserved': True,
        'artifacts_sha256': {p.name: sha(p) for p in sorted(PUBLIC.iterdir()) if p.is_file()},
        'configuration_sha256': sha(ROOT/'configs/cg_alr_v3.json')})
    print('V3 publication privacy, finite aggregates, source hashes and links pass', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'export', 'check'))
    args = parser.parse_args()
    {'verify': verification, 'export': export, 'check': publication_check}[args.stage]()
