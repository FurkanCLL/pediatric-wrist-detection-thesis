"""Independent assignment, AP integration, journals and preservation checks."""

from collections import defaultdict
import csv
import json
import sys

import numpy as np

from .execution import RUN, RESULTS, PRIVATE, locked, require, access_guard
from .protocol import ROOT, read, sha, identity, write_new
from .test_analysis import inputs, point_metrics, METHODS
from .test_data import truth
from .metrics import canonical_predictions, contrasts
from wrist_eval.common import CLASSES
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_eval.metrics import ious


def independent_operating(gt, pred):
    # Keep the frozen float32 IoU primitive; independently assign canonical operating rows.
    matrix = ious(gt[:, :4], pred[:, :4]); target = np.full(len(gt), -1, int); proposal = np.full(len(pred), -1, int)
    for p in range(len(pred)):
        if pred[p, 4] < .25:
            continue
        eligible = [g for g in range(len(gt)) if target[g] < 0 and gt[g, 4] == pred[p, 5] and matrix[g, p] >= .5]
        if eligible:
            g = max(eligible, key=lambda g: (matrix[g, p], -g)); target[g] = p; proposal[p] = g
    return target, proposal


def independent_flags(gt, pred):
    # Assignment is implemented separately, retaining the pinned float32 IoU primitive/thresholds.
    import torch
    from ultralytics.utils.metrics import box_iou
    matrix = box_iou(torch.tensor(gt[:, :4]), torch.tensor(pred[:, :4])).numpy()
    correct = np.zeros((len(pred), 10), bool)
    for column, threshold in enumerate(torch.linspace(.5, .95, 10).numpy()):
        used = set()
        for p in range(len(pred)):
            eligible = [g for g in range(len(gt)) if g not in used and gt[g, 4] == pred[p, 5] and matrix[g, p] >= threshold]
            if eligible:
                g = max(eligible, key=lambda g: (matrix[g, p], -g))
                used.add(g); correct[p, column] = True
    return correct


def independent_ap(stats):
    rows = []
    targets = np.concatenate([s['target_cls'] for s in stats])
    for occurrence, stat in enumerate(stats):
        for i in range(len(stat['conf'])):
            rows.append(((-float(stat['conf'][i]), int(stat['image_index'][i]), int(stat['global_ids'][i]), occurrence),
                         int(stat['pred_cls'][i]), stat['tp'][i]))
    rows.sort(key=lambda v: v[0]); result = {}; all_ap = []
    for c in sorted(set(map(int, targets))):
        correct = np.asarray([v[2] for v in rows if v[1] == c], float).reshape(-1, 10)
        aps = []
        for j in range(10):
            tp = correct[:, j].cumsum(); fp = (1-correct[:, j]).cumsum()
            recall = tp/(int(sum(targets == c))+1e-16); precision = tp/(tp+fp+1e-16)
            x = np.r_[0., recall, recall[-1] if len(recall) else 1., 1.]
            y = np.r_[1., precision, 0., 0.]
            for i in range(len(y)-2, -1, -1):
                y[i] = max(y[i], y[i+1])
            aps.append(float(np.trapezoid(np.interp(np.linspace(0, 1, 101), x, y), np.linspace(0, 1, 101))) if len(correct) else 0.)
        result[CLASSES[c]] = {'ap50': aps[0], 'ap75': aps[5], 'ap50_95': float(np.mean(aps))}; all_ap.append(aps)
    if len(result) == 8:
        arr = np.asarray(all_ap)
        result['overall'] = {'map50': float(arr[:, 0].mean()), 'map75': float(arr[:, 5].mean()), 'map50_95': float(arr.mean())}
    return result


def same_ap(a, b):
    require(set(a) == set(b), 'Independent AP class support differs')
    require(all(abs(a[c][k]-b[c][k]) < 1e-12 for c in a for k in a[c]), 'Independent AP differs')


def verify_split(split, cfg):
    data = read(RUN/('analysis_'+split)/'results.json'); rows, predictions, _ = inputs(split)
    stats, counts, ap, operating = point_metrics(rows, predictions)
    totals = {}
    for method in METHODS:
        tp = fp = fn = 0
        for i, (row, pred) in enumerate(zip(rows, predictions[method])):
            gt = truth(row); pred = canonical_predictions(pred, row['width'], row['height'])
            require(np.array_equal(independent_flags(gt, pred), stats[method][i]['tp']), 'Independent native assignment differs')
            g, p = independent_operating(gt, pred)
            tp += int(sum(g >= 0)); fp += int(sum((p < 0) & (pred[:, 4] >= .25))); fn += int(sum(g < 0))
        require((tp, fp, fn) == tuple(data['operating'][method]['overall'][k] for k in ('tp', 'fp', 'fn')), 'Independent operating totals differ')
        same_ap(independent_ap(stats[method]), data['AP'][method]); totals[method] = {'tp': tp, 'fp': fp, 'fn': fn}
    collected = {track: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for track in ('contrasts', 'operating_contrasts')}
    groups = patient_groups(rows); rng = np.random.default_rng(314159); draw_ids = []
    path = RUN/('analysis_'+split)/'bootstrap.jsonl'
    require(sha(path) == data['bootstrap']['journal_sha256'], 'Bootstrap journal hash differs')
    n = 0
    for rep, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        item = json.loads(line); indices = resample_indices(groups, rng); draw_ids.append(identity(indices)); n += 1
        require(item['replicate'] == rep and item['indices_sha256'] == draw_ids[-1], 'Paired patient draws differ')
        if rep in (0, 249, 499):
            for method in METHODS:
                same_ap(independent_ap([stats[method][i] for i in indices]), item['AP'][method])
        for track, key in (('contrasts', 'AP'), ('operating_contrasts', 'operating')):
            for name, classes in contrasts(item[key], cfg['comparisons']).items():
                for c, metrics in classes.items():
                    for metric, value in metrics.items():
                        collected[track][name][c][metric].append(value)
    require(n == 500 and identity(draw_ids) == data['bootstrap']['draw_sequence_sha256'], 'Bootstrap sequence differs')
    for track, comparisons in collected.items():
        for name, classes in comparisons.items():
            for c, metrics in classes.items():
                for metric, values in metrics.items():
                    require(interval(values) == data['bootstrap'][track][name][c][metric], 'Independent interval arithmetic differs')
    return {'passed': True, 'independent_operating_totals': totals, 'every_native_assignment_checked': True,
            'independent_all_point_AP_and_three_paired_AP_replicates_per_method': True,
            'all_500_draws_and_all_contrast_intervals_checked': True, 'results_sha256': sha(RUN/('analysis_'+split)/'results.json')}


def verify():
    cfg, _ = locked(); sys.addaudithook(access_guard)
    result = {split: verify_split(split, cfg) for split in ('test', 'validation')}
    # Recheck all saved V3 decisions and output identities without model calls.
    from wrist_v3.planning import plan_rois
    from wrist_v3.fusion import fuse_predictions
    legacy, settings = read(ROOT/'configs/adaptive_refinement_proposed.json'), read(ROOT/'configs/cg_alr_v3.json')
    rows, predictions, runtime = inputs('test'); replays = 0
    for method, base in (('final_C', 'final_A'), ('final_F', 'final_D')):
        path = RUN/method/'bundles.jsonl'
        require(sha(path) == runtime[method]['bundles_sha256'], 'Saved V3 bundles changed')
        bundles = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require([b['stem'] for b in bundles] == [r['filestem'] for r in rows], 'V3 bundle membership differs')
        for row, original, output, bundle in zip(rows, predictions[base], predictions[method], bundles):
            require(np.array_equal(original, np.asarray(bundle['global'], np.float32).reshape(-1, 6)), 'Global reference mismatch')
            plan = plan_rois(original, (row['height'], row['width']), legacy, settings)
            require(plan == bundle['plan'], 'ROI replay differs')
            views = [{**v, 'predictions': np.asarray(v['predictions'], np.float32).reshape(-1, 6)} for v in bundle['local_views']]
            fused = fuse_predictions(original, views, plan, legacy, settings)
            require(np.array_equal(output, fused['predictions']) and fused['diagnostics'] == bundle['diagnostics'], 'Fusion replay differs')
            replays += 1
    for method, summary in runtime.items():
        items = [json.loads(line) for line in (RUN/method/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        calls = sum(v['timing']['forward_passes'] for v in items)
        require(calls == summary['usage']['calls'] and calls+3 == summary['physical_calls_including_warmups'], 'Actual call totals differ')
        for key, value in summary['timing'].items():
            from wrist_eval.metrics import distribution
            actual = distribution([v['timing'][key] for v in items if v['timing'][key] is not None])
            require(actual == value, 'Published runtime distribution differs')
    before = read(PRIVATE/'historical_before.json')
    require(all(sha(ROOT/name) == digest for name, digest in before.items()), 'Historical tracked artifact changed')
    require(all(sha(ROOT/v['path']) == v['sha256'] for v in cfg['checkpoints'].values()), 'Checkpoint bytes changed')
    receipt = {'passed': True, 'splits': result, 'V3_replays': replays, 'physical_call_and_runtime_checks': True,
        'historical_tracked_files_preserved': len(before), 'best_checkpoints_unchanged': True,
        'execution_freeze_sha256': sha(RESULTS/'execution_freeze.json'), 'no_new_validation_inference': True,
        'no_test_pilot_replay_or_parameter_revision': True}
    write_new(RUN/'verification.json', receipt)
    print('Independent final-test and validation metrics, journals, calls, V3 replays and preservation pass', flush=True)


def export():
    cfg, _ = locked(); receipt = read(RUN/'verification.json'); require(receipt['passed'], 'Verify before publication')
    data = {split: read(RUN/('analysis_'+split)/'results.json') for split in ('test', 'validation')}
    for split in data:
        require(sha(RUN/('analysis_'+split)/'results.json') == receipt['splits'][split]['results_sha256'], 'Verified analysis changed')
    write_new(RESULTS/'results.json', data); write_new(RESULTS/'verification.json', receipt)
    prepared = read(RUN/'data_receipt.json')
    write_new(RESULTS/'data_integrity.json', {k: v for k, v in prepared.items() if k != 'prepared_files'})
    tables = defaultdict(list)
    for split, values in data.items():
        for method in METHODS:
            for c, metrics in values['AP'][method].items():
                tables['AP'].append({'split': split, 'method': method, 'class': c, **metrics})
            for c, counts in values['operating'][method].items():
                tables['operating'].append({'split': split, 'method': method, 'class': c, **counts})
        for name, classes in values['AP_contrasts'].items():
            for c, metrics in classes.items():
                for metric, point in metrics.items():
                    tables['paired_AP'].append({'split': split, 'contrast': name, 'class': c, 'metric': metric,
                        'delta': point, **values['bootstrap']['contrasts'][name][c][metric]})
        for pair, classes in values['recovery'].items():
            for c, counts in classes.items():
                tables['recovery'].append({'split': split, 'contrast': pair, 'class': c, **counts})
        for c, support in values['support'].items():
            tables['support'].append({'split': split, 'class': c, **support})
    for method, value in data['test']['runtime'].items():
        t = value['timing']
        tables['runtime'].append({'method': method, 'calls': value['usage']['calls'], 'mean_calls': t['forward_passes']['mean'],
            'mean_ms': t['e2e_seconds']['mean']*1000, 'median_ms': t['e2e_seconds']['median']*1000,
            'p95_ms': t['e2e_seconds']['p95']*1000, 'peak_allocated_mib': t['allocated_peak_bytes']['max']/2**20,
            'peak_reserved_mib': t['reserved_peak_bytes']['max']/2**20,
            'paired_global_ratio': value.get('mean_paired_latency_ratio')})
    for name, values in tables.items():
        fields = list(dict.fromkeys(k for row in values for k in row))
        with (RESULTS/(name+'.csv')).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    print('Aggregate final results exported; detailed predictions remain private', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('verify', 'export'))
    args = parser.parse_args(); verify() if args.stage == 'verify' else export()
