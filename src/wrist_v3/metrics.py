"""Historical AP and a separate immutable-source-ID tie sensitivity track."""

import numpy as np

from wrist_eval.common import CLASSES
from wrist_eval.metrics import native_stats, ap_summary


def compact_ap(value):
    result = {name: {k: value['classes'][name][k] for k in ('ap50', 'ap75', 'ap50_95')}
              for name in CLASSES if name in value['classes']}
    if len(result) == len(CLASSES):
        result['overall'] = {k: value[k] for k in ('map50', 'map75', 'map50_95')}
    return result


def image_stats(gt, predictions, global_ids, image_index, canonical=False):
    ids = np.asarray(global_ids, int)
    if len(ids) != len(predictions) or sorted(ids.tolist()) != list(range(len(ids))):
        raise ValueError('Missing or duplicate original global prediction IDs')
    order = np.argsort(ids, kind='stable') if canonical else np.arange(len(ids))
    value = native_stats(gt, predictions[order])
    return {**value, 'global_ids': ids[order], 'image_index': np.full(len(ids), image_index, int)}


def historical_ap(stats):
    return compact_ap(ap_summary(stats))


def deterministic_ap(stats):
    from ultralytics.utils.metrics import compute_ap
    # Sort original identities after unmodified score; sample occurrence only separates replicas.
    tp = np.concatenate([s['tp'] for s in stats])
    confidence = np.concatenate([s['conf'] for s in stats])
    classes = np.concatenate([s['pred_cls'] for s in stats])
    targets = np.concatenate([s['target_cls'] for s in stats])
    images = np.concatenate([s['image_index'] for s in stats])
    ids = np.concatenate([s['global_ids'] for s in stats])
    occurrences = np.concatenate([np.full(len(s['conf']), i, int) for i, s in enumerate(stats)])
    order = np.lexsort((occurrences, ids, images, -confidence))
    tp, classes = tp[order], classes[order]
    result = {}; all_ap = []
    for class_id, count in zip(*np.unique(targets.astype(int), return_counts=True)):
        selected = classes == class_id
        correct = tp[selected].cumsum(axis=0); incorrect = (1-tp[selected]).cumsum(axis=0)
        recall = correct/(count+1e-16); precision = correct/(correct+incorrect+1e-16)
        ap = np.array([compute_ap(recall[:, j], precision[:, j])[0] if selected.any() else 0. for j in range(10)])
        result[CLASSES[class_id]] = {'ap50': float(ap[0]), 'ap75': float(ap[5]), 'ap50_95': float(ap.mean())}
        all_ap.append(ap)
    if len(result) == len(CLASSES):
        ap = np.asarray(all_ap)
        result['overall'] = {'map50': float(ap[:, 0].mean()), 'map75': float(ap[:, 5].mean()), 'map50_95': float(ap.mean())}
    return result


def ap_delta(other, base):
    return {name: {key: value-base[name][key] for key, value in metrics.items()}
            for name, metrics in other.items() if name in base}
