"""Predeclared two-track scientific analysis after both fixed V3 runs finish."""

from collections import Counter, defaultdict
import json
import time

import numpy as np

from wrist_eval.common import CLASSES, validation_records, sha, identity
from wrist_eval.analyze import truth, describe, compare, compare_false_positives
from wrist_eval.metrics import match
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_eval.adaptive import fuse_predictions as legacy_fuse
from wrist_eval.adaptive_validation import assess_variant
from wrist_compare.common import install_access_guard
from .planning import plan_rois
from .fusion import fuse_predictions
from .metrics import image_stats, historical_ap, deterministic_ap, ap_delta
from .protocol import ROOT, PUBLIC, LOCAL, read, write_new, frozen, references, require

METHODS = ('A', 'C', 'V3_standard', 'D', 'F', 'V3_rarity')
PAIRS = {'V3_standard-A': ('V3_standard', 'A'), 'V3_standard-C': ('V3_standard', 'C'),
         'V3_rarity-D': ('V3_rarity', 'D'), 'V3_rarity-F': ('V3_rarity', 'F')}


def load_inputs():
    cfg, legacy, _ = frozen()
    rows = validation_records(); predictions = {}; ids = {}; bundles = {}; summaries = {}
    official = read(ROOT/'docs/evidence/rarity_inference_comparison/results.json')
    for condition, base, old in (('standard', 'A', 'C'), ('rarity', 'D', 'F')):
        predictions[base] = references(condition)
        ids[base] = {stem: list(range(len(pred))) for stem, pred in predictions[base].items()}
        old_path = ROOT/('outputs/adaptive_refinement/full/full_01/bundles.jsonl' if old == 'C'
                         else 'outputs/rarity_inference_comparison/study_01/full_F/bundles.jsonl')
        old_expected = (official['conditions']['C']['source']['prediction_sha256'] if old == 'C' else
                        read(ROOT/'docs/evidence/rarity_inference_comparison/full_F.json')['bundles_sha256'])
        require(sha(old_path) == old_expected, 'Historical V2 bundle identity differs')
        old_records = [json.loads(line) for line in old_path.read_text(encoding='utf-8').splitlines()]
        predictions[old] = {r['stem']: np.asarray(r['variants']['V2']['pred'], np.float32).reshape(-1, 6) for r in old_records}
        ids[old] = {r['stem']: r['variants']['V2']['diagnostics']['output_global_ids'] for r in old_records}
        bundles[old] = old_records
        directory = LOCAL/f'full_{condition}'; summary = read(directory/'summary.json')
        require(summary['passed'] and summary['images'] == 3088 and summary['freeze_sha256'] == sha(PUBLIC/'freeze.json'), 'Full V3 run incomplete')
        require(sha(directory/'predictions.jsonl') == summary['prediction_sha256'] and sha(directory/'bundles.jsonl') == summary['bundles_sha256'], 'Saved V3 artifacts changed')
        records = [json.loads(line) for line in (directory/'bundles.jsonl').read_text(encoding='utf-8').splitlines()]
        exported = [json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        key = 'V3_'+condition; summaries[key] = summary; bundles[key] = records
        require([r['stem'] for r in records] == [r['filestem'] for r in rows], 'Full membership/order differs')
        predictions[key] = {}; ids[key] = {}
        require([v['stem'] for v in exported] == [r['filestem'] for r in rows], 'Prediction export membership differs')
        for row, record, item in zip(rows, records, exported):
            stem = row['filestem']; original = np.asarray(record['global'], np.float32).reshape(-1, 6)
            require(np.array_equal(original, predictions[base][stem]), 'V3 global reference differs')
            plan = plan_rois(original, (row['height'], row['width']), legacy, cfg)
            require(plan == record['plan'], 'Saved V3 ROI decisions differ from frozen replay')
            views = [{**v, 'predictions': np.asarray(v['predictions'], np.float32).reshape(-1, 6)} for v in record['local_views']]
            replay = fuse_predictions(original, views, plan, legacy, cfg)
            expected = np.asarray(record['predictions'], np.float32).reshape(-1, 6)
            require(np.array_equal(expected, np.asarray(item['pred'], np.float32).reshape(-1, 6))
                and item['global_ids'] == record['diagnostics']['output_global_ids'], 'Prediction/bundle identity join differs')
            require(np.array_equal(replay['predictions'], expected) and replay['diagnostics'] == record['diagnostics'], 'V3 fusion replay differs')
            predictions[key][stem] = expected; ids[key][stem] = record['diagnostics']['output_global_ids']
        require(all(set(p) == {r['filestem'] for r in rows} for p in (predictions[base], predictions[old], predictions[key])), 'Comparator membership differs')
    return cfg, legacy, rows, predictions, ids, bundles, summaries, official


def collect(destination, differences):
    for comparison, classes in differences.items():
        for name, metrics in classes.items():
            for key, value in metrics.items():
                destination[comparison][name][key].append(float(value))


def summarize_collected(values):
    return {pair: {name: {metric: interval(samples) for metric, samples in metrics.items()}
                   for name, metrics in classes.items()} for pair, classes in values.items()}


def gate(track_ap, points, ci, summary, base, method, legacy):
    delta = ap_delta(track_ap[method], track_ap[base]); pair = method+'-'+base
    metric_input = {'methods': {'V3': {'operating_points': points[method]}, 'standard_serial': {'operating_points': points[base]}},
        'comparisons': {'V3_minus_standard_serial': {'overall_ap_delta': delta['overall'],
             'per_class_ap_delta': {name: delta[name] for name in CLASSES}}}}
    uncertainty = {'comparisons': {'V3_minus_standard_serial': ci[pair]}}
    timing = summary['timing']
    result = assess_variant('V3', metric_input, uncertainty, timing['e2e_seconds']['mean'],
                            timing['paired_global_e2e_seconds']['mean'], legacy['phase_3c2_plan']['proposed_success'])
    result['checks']['maximum_physical_calls'] = timing['forward_passes']['max'] <= 3
    result['strict_success'] = all(result['checks'].values())
    return result


def promotions(rows, predictions, bundles, method, base):
    counts = Counter(); by_class = {name: Counter() for name in CLASSES}; roi_counts = Counter()
    for row, record in zip(rows, bundles[method]):
        stem = row['filestem']; gt = truth(row)
        old_g, old_p, _ = match(gt, predictions[base][stem]); new_g, new_p, _ = match(gt, predictions[method][stem])
        inverse = {g: i for i, g in enumerate(record['diagnostics']['output_global_ids'])}
        update_rois, useful_rois, fp_rois = set(), set(), set()
        for change in record['diagnostics']['score_changes']:
            g = change['global_id']; pi = inverse[g]; gi = new_p[pi]
            crossed = change['before'] < .25 <= change['after']
            status = ('recovered_TP' if crossed and gi >= 0 and old_g[gi] < 0 else
                      'reassigned_TP' if crossed and gi >= 0 else 'promoted_FP' if crossed else 'below_threshold')
            counts[status] += 1; by_class[CLASSES[int(predictions[base][stem][g, 5])]][status] += 1
            update_rois.add(change['roi_id'])
            if status == 'recovered_TP':
                useful_rois.add(change['roi_id'])
            elif status == 'promoted_FP':
                fp_rois.add(change['roi_id'])
        for g in np.flatnonzero(predictions[base][stem][:, 4] >= .25):
            counts['inherited_TP_to_FP'] += int(old_p[g] >= 0 and new_p[inverse[g]] < 0)
            counts['inherited_FP_to_TP'] += int(old_p[g] < 0 and new_p[inverse[g]] >= 0)
        roi_counts.update({'local_calls': len(record['plan']['rois']), 'rois_with_update': len(update_rois),
            'rois_with_recovery': len(useful_rois), 'rois_with_promoted_FP': len(fp_rois),
            'rois_both_recovery_and_FP': len(useful_rois & fp_rois),
            'rois_without_update': len(record['plan']['rois'])-len(update_rois)})
        counts.update('candidate_'+v['reason'] for v in record['plan']['candidate_evaluation'])
        counts.update('matching_'+v['reason'] for v in record['diagnostics']['candidate_matching'])
        counts.update('arbitration_'+v['reason'] for v in record['diagnostics']['arbitration'])
    return {'outcomes': dict(counts), 'classes': {c: dict(v) for c, v in by_class.items()}, 'roi_productivity': dict(roi_counts)}


def component_attribution(rows, bundles, legacy, cfg):
    result = {}
    for condition, old, method in (('standard', 'C', 'V3_standard'), ('rarity', 'F', 'V3_rarity')):
        cells = {name: [] for name in cfg['evaluation']['component_attribution']}
        historical_records = {r['stem']: r for r in bundles[old]}
        for image_index, (row, record) in enumerate(zip(rows, bundles[method])):
            original = np.asarray(record['global'], np.float32).reshape(-1, 6); old_record = historical_records[row['filestem']]
            old_views = [{**v, 'predictions': np.asarray(v['predictions'], np.float32).reshape(-1, 6)} for v in old_record['local_views']]
            views = [{**v, 'predictions': np.asarray(v['predictions'], np.float32).reshape(-1, 6)} for v in record['local_views']]
            hybrid_plan = {**record['plan'], 'candidates': record['plan']['all_candidates']}
            variants = [legacy_fuse(original, old_views, old_record['plan'], legacy, 'V1'),
                legacy_fuse(original, views, hybrid_plan, legacy, 'V1'),
                fuse_predictions(original, views, record['plan'], legacy, cfg, legacy_cap=True),
                {'predictions': np.asarray(record['predictions'], np.float32).reshape(-1, 6), 'diagnostics': record['diagnostics']}]
            for name, value in zip(cells, variants):
                cells[name].append((value['predictions'], value['diagnostics']['output_global_ids']))
        result[condition] = {}
        for name, values in cells.items():
            stats = [image_stats(truth(row), pred, ids, i, True) for i, (row, (pred, ids)) in enumerate(zip(rows, values))]
            tp = fp = fn = 0
            for row, (pred, _) in zip(rows, values):
                g, p, _ = match(truth(row), pred)
                tp += int(sum(g >= 0)); fn += int(sum(g < 0)); fp += int(sum((p < 0) & (pred[:, 4] >= .25)))
            result[condition][name] = {'deterministic_AP': deterministic_ap(stats), 'tp': tp, 'fp': fp, 'fn': fn}
    return result


def run():
    install_access_guard()
    destination = LOCAL/'analysis'; destination.mkdir(exist_ok=False)
    cfg, legacy, rows, predictions, ids, bundles, summaries, official = load_inputs()
    stats = {track: {} for track in ('historical', 'deterministic')}; aps = {track: {} for track in stats}
    for track in stats:
        for method in METHODS:
            stats[track][method] = [image_stats(truth(row), predictions[method][row['filestem']], ids[method][row['filestem']], i,
                canonical=track == 'deterministic') for i, row in enumerate(rows)]
            aps[track][method] = (historical_ap if track == 'historical' else deterministic_ap)(stats[track][method])
            if track == 'historical' and method in 'ACDF':
                require(aps[track][method] == official['conditions'][method]['AP'], 'Historical AP no longer reproduces')
        for base, other in (('A', 'C'), ('A', 'V3_standard'), ('D', 'F'), ('D', 'V3_rarity')):
            if track == 'deterministic':
                require(all(aps[track][base][CLASSES[c]] == aps[track][other][CLASSES[c]] for c in (1, 2, 4, 5, 6, 7)), 'Protected-class deterministic AP drifted')
    points = {c: official['conditions'][c]['operating_points'] for c in 'ACDF'}
    for method in ('V3_standard', 'V3_rarity'):
        points[method] = {}
        for confidence in cfg['evaluation']['operating_confidences']:
            for threshold in cfg['evaluation']['operating_ious']:
                point, detailed = describe(rows, predictions[method], confidence, threshold)
                points[method][f'conf{confidence}_iou{threshold}'] = point
                if confidence == .25 and threshold == .5:
                    write_new(destination/(method+'_primary_details.json'), detailed)
    changes = {name: {'recovery': compare(rows, predictions[base], predictions[other]),
                     'FP_correspondence': compare_false_positives(rows, predictions[base], predictions[other])}
               for name, (other, base) in PAIRS.items()}
    usage = {method: promotions(rows, predictions, bundles, method, base) for method, base in (('V3_standard', 'A'), ('V3_rarity', 'D'))}
    print('Both metric tracks, frozen comparator AP, reporting grid and promotion diagnostics pass', flush=True)
    prior_path = ROOT/'outputs/rarity_inference_comparison/study_01/bootstrap/replicates.jsonl'
    require(sha(prior_path) == official['bootstrap']['replicates_sha256'], 'Historical bootstrap identity differs')
    prior = [json.loads(line) for line in prior_path.read_text(encoding='utf-8').splitlines()]
    groups, rng = patient_groups(rows), np.random.default_rng(314159)
    collected = {track: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for track in stats}
    counts = {}
    for method in METHODS:
        values = []
        for row in rows:
            pred = predictions[method][row['filestem']]; gt = truth(row); g, p, _ = match(gt, pred)
            values.append([int(sum(g >= 0)), int(sum((p < 0) & (pred[:, 4] >= .25))), int(sum(g < 0)),
                int(sum((g >= 0) & (gt[:, 4] == 0))), int(sum((p < 0) & (pred[:, 4] >= .25) & (pred[:, 5] == 0))), int(sum((g < 0) & (gt[:, 4] == 0)))])
        counts[method] = np.asarray(values, int)
    operating_samples = defaultdict(lambda: defaultdict(list))
    started = time.perf_counter()
    with (destination/'bootstrap_replicates.jsonl').open('x', encoding='utf-8') as stream:
        for rep in range(500):
            indices = resample_indices(groups, rng)
            require(identity(indices) == prior[rep]['indices_sha256'], 'Paired patient draw differs')
            sampled = {track: {} for track in stats}
            for method in METHODS:
                if method in 'ACDF':
                    sampled['historical'][method] = prior[rep]['metrics'][method]
                    if rep in (0, 1, 249, 499):
                        require(historical_ap([stats['historical'][method][i] for i in indices]) == sampled['historical'][method], 'Historical replicate reproduction differs')
                else:
                    sampled['historical'][method] = historical_ap([stats['historical'][method][i] for i in indices])
                sampled['deterministic'][method] = deterministic_ap([stats['deterministic'][method][i] for i in indices])
            differences = {track: {name: ap_delta(sampled[track][other], sampled[track][base])
                                  for name, (other, base) in PAIRS.items()} for track in stats}
            for track in stats:
                collect(collected[track], differences[track])
            for name, (other, base) in PAIRS.items():
                a, b = counts[other][indices].sum(axis=0), counts[base][indices].sum(axis=0)
                for key, value in {'net_TP': a[0]-b[0], 'net_FP': a[1]-b[1],
                    'precision_delta': a[0]/(a[0]+a[1])-b[0]/(b[0]+b[1]),
                    'recall_delta': a[0]/(a[0]+a[2])-b[0]/(b[0]+b[2]),
                    'FP_ratio': a[1]/b[1], 'fracture_FP_ratio': a[4]/b[4]}.items():
                    operating_samples[name][key].append(float(value))
            stream.write(json.dumps({'replicate': rep, 'indices_sha256': identity(indices), 'AP': sampled, 'contrasts': differences}, allow_nan=False)+'\n')
            stream.flush()
            if rep % 25 == 0:
                print(f'V3 paired bootstrap {rep+1}/500 ({time.perf_counter()-started:.1f}s)', flush=True)
    intervals = {track: summarize_collected(collected[track]) for track in stats}
    decisions = {}
    for condition, method, base in (('standard', 'V3_standard', 'A'), ('rarity', 'V3_rarity', 'D')):
        decisions[condition] = {track: gate(aps[track], points, intervals[track], summaries[method], base, method, legacy) for track in stats}
        decisions[condition]['strict_success_both_tracks'] = all(decisions[condition][track]['strict_success'] for track in stats)
    attribution = component_attribution(rows, bundles, legacy, cfg)
    write_new(destination/'results.json', {'scope': 'Exploratory follow-up; historical A-F unchanged; one V3 configuration and both predefined checkpoints',
        'AP': aps, 'operating_points': points, 'changes': changes, 'promotion_and_ROI': usage, 'runtime': summaries,
        'AP_intervals': intervals, 'operating_intervals': {name: {k: interval(v) for k, v in metrics.items()} for name, metrics in operating_samples.items()},
        'scientific_gates': decisions, 'component_attribution': attribution,
        'bootstrap_replicates': 500, 'historical_bootstrap_reused': True, 'bootstrap_replicates_sha256': sha(destination/'bootstrap_replicates.jsonl'),
        'freeze_sha256': sha(PUBLIC/'freeze.json')})
    print('V3 scientific analysis and predeclared component attribution saved', flush=True)


if __name__ == '__main__':
    run()
