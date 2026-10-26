"""A common content-based row order for all six final prediction sets."""

from collections import defaultdict
import numpy as np

from wrist_eval.common import CLASSES
from wrist_eval.metrics import validate_predictions, match
from wrist_eval.bootstrap import patient_groups, resample_indices, interval
from wrist_v3.metrics import image_stats, deterministic_ap


def canonical_predictions(predictions, width, height):
    pred = validate_predictions(predictions, width, height)
    if len(pred) > 300:
        raise ValueError('Final predictions exceed the frozen image-level cap')
    # Scores are never rounded or perturbed. Identical rows are interchangeable duplicate occurrences.
    order = sorted(range(len(pred)), key=lambda i: (-float(pred[i, 4]), int(pred[i, 5]),
                                                   *map(float, pred[i, :4])))
    return pred[order]


def final_stats(gt, predictions, width, height, image_index):
    gt = np.asarray(gt, np.float32).reshape(-1, 5)
    if not np.isfinite(gt).all() or (len(gt) and (np.any(gt[:, 2:4] <= gt[:, :2])
            or np.any(gt[:, :4] < 0) or np.any(gt[:, [0, 2]] > width) or np.any(gt[:, [1, 3]] > height)
            or np.any(gt[:, 4] < 0) or np.any(gt[:, 4] >= 8) or np.any(gt[:, 4] != np.floor(gt[:, 4])))):
        raise ValueError('Invalid evaluation annotations')
    pred = canonical_predictions(predictions, width, height)
    return image_stats(gt, pred, list(range(len(pred))), image_index, canonical=True)


def operating_counts(gt, predictions, width, height):
    gt = np.asarray(gt, np.float32).reshape(-1, 5)
    pred = canonical_predictions(predictions, width, height)
    g, p, _ = match(gt, pred, threshold=.50, confidence=.25)
    result = {}
    for c, name in enumerate(CLASSES):
        tp = int(np.sum((g >= 0) & (gt[:, 4] == c)))
        fn = int(np.sum((g < 0) & (gt[:, 4] == c)))
        fp = int(np.sum((p < 0) & (pred[:, 4] >= .25) & (pred[:, 5] == c)))
        result[name] = {'tp': tp, 'fp': fp, 'fn': fn}
    return result


def contrasts(values, coefficients):
    # A class/overall metric is undefined if any contributing method lacks that class.
    result = {}
    for name, weights in coefficients.items():
        classes = set.intersection(*(set(values[method]) for method in weights))
        result[name] = {c: {metric: float(sum(weight*values[method][c][metric] for method, weight in weights.items()))
                           for metric in sorted(set.intersection(*(set(values[m][c]) for m in weights)))}
                        for c in sorted(classes)}
    return result


def count_summary(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (8, 3) or np.any(counts < 0):
        raise ValueError('Expected eight TP/FP/FN triples')
    result = {}
    for name, values in zip([*CLASSES, 'overall'], [*counts, counts.sum(axis=0)]):
        tp, fp, fn = map(int, values)
        result[name] = {'tp': tp, 'fp': fp, 'fn': fn}
        if tp+fp:
            result[name]['precision'] = tp/(tp+fp)
        if tp+fn:
            result[name]['recall'] = tp/(tp+fn)
    return result


def paired_bootstrap(rows, stats, coefficients, replicates=500, seed=314159, counts=None):
    if replicates != 500 or seed != 314159:
        raise ValueError('The final bootstrap is fixed at 500 replicates and seed 314159')
    groups = patient_groups(rows)
    if not groups or any(len(v) != len(rows) for v in stats.values()):
        raise ValueError('All methods must contain the same complete image population')
    rng = np.random.default_rng(seed)
    collected = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    operating = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if counts is not None and (set(counts) != set(stats) or any(np.asarray(v).shape != (len(rows), 8, 3) for v in counts.values())):
        raise ValueError('Operating counts must follow the same methods, images and classes')
    draw_ids = []
    from .protocol import identity
    for _ in range(replicates):
        indices = resample_indices(groups, rng)
        draw_ids.append(identity(indices))
        values = {method: deterministic_ap([method_stats[i] for i in indices]) for method, method_stats in stats.items()}
        for name, classes in contrasts(values, coefficients).items():
            for c, metrics in classes.items():
                for metric, value in metrics.items():
                    collected[name][c][metric].append(value)
        if counts is not None:
            values = {method: count_summary(np.asarray(v)[indices].sum(axis=0)) for method, v in counts.items()}
            for name, classes in contrasts(values, coefficients).items():
                for c, metrics in classes.items():
                    for metric, value in metrics.items():
                        operating[name][c][metric].append(value)
    def summarize(values):
        return {name: {c: {metric: interval(samples) for metric, samples in metrics.items()}
                       for c, metrics in classes.items()} for name, classes in values.items()}
    return {'replicates': replicates, 'seed': seed, 'draw_sequence_sha256': identity(draw_ids),
            'contrasts': summarize(collected), 'operating_contrasts': summarize(operating)}
