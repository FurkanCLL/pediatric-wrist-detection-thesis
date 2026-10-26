"""Common final metrics, fixed paired contrasts and saved-validation references."""

from collections import defaultdict
import json
import sys
import time

import numpy as np

from .execution import RUN, RESULTS, locked, require, access_guard
from .protocol import ROOT, read, sha, identity, write_new
from .test_data import records, truth
from .metrics import final_stats, canonical_predictions, operating_counts, count_summary, contrasts
from wrist_eval.common import CLASSES
from wrist_eval.metrics import match
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_v3.metrics import deterministic_ap

METHODS = ['final_'+v for v in 'ABCDEF']


def load_predictions(path, rows, expected):
    require(sha(path) == expected, 'Prediction artifact hash differs')
    items = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    require([v['stem'] for v in items] == [r['filestem'] for r in rows], 'Prediction membership/order differs')
    return [np.asarray(v['pred'], np.float32).reshape(-1, 6) for v in items]


def inputs(split):
    rows = records(split); predictions = {}; summaries = {}
    if split == 'test':
        for method in METHODS:
            summary = read(RUN/method/'summary.json')
            require(summary['passed'] and summary['images'] == 3178, 'Every final condition must complete before evaluation')
            predictions[method] = load_predictions(RUN/method/'predictions.jsonl', rows, summary['prediction_sha256'])
            summaries[method] = summary
    else:
        old = read(ROOT/'docs/evidence/rarity_inference_comparison/results.json')['conditions']
        paths = {'final_A': 'outputs/error_analysis_sahi/full/standard_serial/predictions.jsonl',
                 'final_B': 'outputs/error_analysis_sahi/full/sahi_context/predictions.jsonl',
                 'final_D': 'outputs/rarity_comparison/study_01/full/predictions.jsonl',
                 'final_E': 'outputs/rarity_inference_comparison/study_01/full_E/predictions.jsonl'}
        for method, path in paths.items():
            predictions[method] = load_predictions(ROOT/path, rows, old[method[-1]]['source']['prediction_sha256'])
        for method, name in (('final_C', 'standard'), ('final_F', 'rarity')):
            summary = read(ROOT/f'docs/evidence/V3/full_{name}.json')
            predictions[method] = load_predictions(ROOT/f'outputs/V3/study_01/full_{name}/predictions.jsonl', rows, summary['prediction_sha256'])
    return rows, predictions, summaries


def point_metrics(rows, predictions):
    stats, counts, ap, operating = {}, {}, {}, {}
    for method in METHODS:
        stats[method] = []; per_image = []
        for i, (row, pred) in enumerate(zip(rows, predictions[method])):
            gt = truth(row)
            stats[method].append(final_stats(gt, pred, row['width'], row['height'], i))
            values = operating_counts(gt, pred, row['width'], row['height'])
            per_image.append([[values[c][k] for k in ('tp', 'fp', 'fn')] for c in CLASSES])
        counts[method] = np.asarray(per_image, int)
        ap[method] = deterministic_ap(stats[method]); operating[method] = count_summary(counts[method].sum(axis=0))
    return stats, counts, ap, operating


def bootstrap(rows, stats, counts, coefficients, path):
    # Same numerical operations as the frozen helper, with an inspectable replicate journal.
    groups = patient_groups(rows); rng = np.random.default_rng(314159)
    samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    operating = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    draws = []; began = time.perf_counter()
    with path.open('x', encoding='utf-8') as stream:
        for rep in range(500):
            indices = resample_indices(groups, rng); draws.append(identity(indices))
            ap = {m: deterministic_ap([v[i] for i in indices]) for m, v in stats.items()}
            op = {m: count_summary(v[indices].sum(axis=0)) for m, v in counts.items()}
            differences, op_differences = contrasts(ap, coefficients), contrasts(op, coefficients)
            for target, values in ((samples, differences), (operating, op_differences)):
                for name, classes in values.items():
                    for c, metrics in classes.items():
                        for metric, value in metrics.items():
                            target[name][c][metric].append(value)
            stream.write(json.dumps({'replicate': rep, 'indices_sha256': draws[-1], 'AP': ap, 'operating': op}, allow_nan=False)+'\n'); stream.flush()
            if rep % 25 == 0:
                print(f'{path.parent.name} bootstrap {rep+1}/500 ({time.perf_counter()-began:.1f}s)', flush=True)
    def summarize(values):
        return {name: {c: {metric: interval(v) for metric, v in metrics.items()} for c, metrics in classes.items()}
                for name, classes in values.items()}
    return {'replicates': 500, 'seed': 314159, 'draw_sequence_sha256': identity(draws),
            'contrasts': summarize(samples), 'operating_contrasts': summarize(operating), 'journal_sha256': sha(path)}


def recoveries(rows, predictions):
    result = {}
    for other, base in (('B', 'A'), ('C', 'A'), ('C', 'B'), ('E', 'D'), ('F', 'D'), ('F', 'E')):
        counts = {c: {'both': 0, 'recovered': 0, 'lost': 0, 'neither': 0} for c in CLASSES}
        for row, a, b in zip(rows, predictions['final_'+base], predictions['final_'+other]):
            gt = truth(row)
            a = canonical_predictions(a, row['width'], row['height']); b = canonical_predictions(b, row['width'], row['height'])
            ga, _, _ = match(gt, a); gb, _, _ = match(gt, b)
            for i, box in enumerate(gt):
                label = 'both' if ga[i] >= 0 and gb[i] >= 0 else 'recovered' if gb[i] >= 0 else 'lost' if ga[i] >= 0 else 'neither'
                counts[CLASSES[int(box[4])]][label] += 1
        result[f'final_{other}-final_{base}'] = counts
    return result


def gates(ap, operating, ci, runtime, cfg):
    limit = cfg['engineering_gates']; result = {}
    for method, base, pair in (('final_C', 'final_A', 'v3_standard_training'), ('final_F', 'final_D', 'v3_rarity_training')):
        a, b = ap[method], ap[base]; op, bp = operating[method], operating[base]
        check = {'overall_map50_95_positive': a['overall']['map50_95'] > b['overall']['map50_95'],
            'overall_lower95_positive': ci['contrasts'][pair]['overall']['map50_95']['lower'] > 0,
            'map50_nondecreasing': a['overall']['map50'] >= b['overall']['map50'],
            'fracture_AP_points_preserved': all(a['fracture'][k] >= b['fracture'][k] for k in ('ap50', 'ap75', 'ap50_95')),
            'fracture_lower95_tolerance': all(ci['contrasts'][pair]['fracture'][k]['lower'] >= limit['fracture_lower95_tolerance'] for k in ('ap50', 'ap75', 'ap50_95')),
            'FP_limit': op['overall']['fp'] <= limit['FP_ratio_max']*bp['overall']['fp'],
            'fracture_FP_limit': op['fracture']['fp'] <= limit['fracture_FP_ratio_max']*bp['fracture']['fp'],
            'paired_latency_limit': runtime[method]['mean_paired_latency_ratio'] <= limit['V3_mean_paired_latency_ratio_max'],
            'maximum_calls': runtime[method]['timing']['forward_passes']['max'] <= limit['maximum_V3_calls']}
        result[method] = {'checks': check, 'strict_success': all(check.values())}
    return result


def analyze():
    cfg, _ = locked(); sys.addaudithook(access_guard)
    for method in METHODS:
        require(read(RUN/method/'summary.json')['passed'], 'Complete all six conditions first')
    for split in ('test', 'validation'):
        destination = RUN/('analysis_'+split); destination.mkdir(exist_ok=False)
        rows, predictions, runtime = inputs(split)
        stats, counts, ap, operating = point_metrics(rows, predictions)
        print(f'{split}: common point metrics computed for all six conditions', flush=True)
        ci = bootstrap(rows, stats, counts, cfg['comparisons'], destination/'bootstrap.jsonl')
        support = {c: {'objects': sum(sum(b['class_id'] == i for b in r['boxes']) for r in rows),
            'patients': len({r['patient_id'] for r in rows if any(b['class_id'] == i for b in r['boxes'])}),
            'images': sum(any(b['class_id'] == i for b in r['boxes']) for r in rows)} for i, c in enumerate(CLASSES)}
        result = {'split': split, 'images': len(rows), 'patients': len({r['patient_id'] for r in rows}), 'support': support,
            'AP': ap, 'operating': operating, 'AP_contrasts': contrasts(ap, cfg['comparisons']),
            'operating_differences': contrasts(operating, cfg['comparisons']), 'bootstrap': ci,
            'recovery': recoveries(rows, predictions), 'runtime': runtime}
        if split == 'test':
            result['engineering_gates'] = gates(ap, operating, ci, runtime, cfg)
        write_new(destination/'results.json', result)
        print(f'{split} analysis complete', flush=True)


if __name__ == '__main__':
    analyze()
