"""Small diagnostic calculations; never choose or alter inference parameters."""

from collections import defaultdict

import numpy as np


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0}
    return {'n': len(values), 'mean': float(values.mean()),
        **{name: float(np.quantile(values, q)) for name, q in
           (('min', 0), ('p05', .05), ('p25', .25), ('median', .5), ('p75', .75), ('p95', .95), ('max', 1))}}


def superiority(positive, negative):
    # Probability that a TP feature exceeds an FP feature, with half credit for ties.
    # This is descriptive separation, not a fitted classifier or a chosen decision rule.
    a, b = np.asarray(positive, float), np.asarray(negative, float)
    if not len(a) or not len(b):
        return None
    return float(((a[:, None] > b).sum()+.5*(a[:, None] == b).sum())/(len(a)*len(b)))


def cardinality(edges):
    # Maximum one-to-one graph matching gives an oracle capacity, never predictions/AP.
    assigned = {}
    def augment(target, seen):
        for proposal in sorted(edges[target]):
            if proposal in seen:
                continue
            seen.add(proposal)
            if proposal not in assigned or augment(assigned[proposal], seen):
                assigned[proposal] = target
                return True
        return False
    return sum(augment(target, set()) for target in sorted(edges))


def oracle_capacity(matrix, gt_ids, pred_ids):
    return cardinality({int(g): [int(p) for p in pred_ids if matrix[g, p] >= .5] for g in gt_ids})


def contained(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def feature_groups(items, group_keys, feature_names):
    groups = defaultdict(list)
    for item in items:
        groups[tuple(item[k] for k in group_keys)].append(item)
    result = []
    for key, values in sorted(groups.items()):
        result.append({**dict(zip(group_keys, key)), 'events': len(values),
            'images': len({v['stem'] for v in values}), 'patients': len({v['patient_id'] for v in values}),
            'features': {name: summarize([v[name] for v in values]) for name in feature_names}})
    return result
