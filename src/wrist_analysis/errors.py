"""Error categories and object-level recovery using unchanged saved predictions."""

from collections import Counter, defaultdict

import numpy as np

from wrist_eval.analyze import error_types, geometry
from wrist_eval.metrics import match
from .evidence import CLASSES, require


def errors(result, rows, predictions):
    details, aggregates, strata, recovery = {}, [], [], []
    for condition, by_stem in predictions.items():
        groups = {name: {'gt_errors': Counter(), 'fp_errors': Counter(), 'patients': set(), 'images': set(),
                         'objects': 0, 'tp': 0, 'fp': 0} for name in CLASSES}
        bins = defaultdict(lambda: {'objects': 0, 'tp': 0, 'patients': set()})
        details[condition] = {}
        for stem, row in rows.items():
            gt, pred = row['gt'], by_stem[stem]
            g, p, matrix, reasons, fp_reasons = error_types(gt, pred)
            geometries = []
            for gi, box in enumerate(gt):
                name = CLASSES[int(box[4])]; group = groups[name]
                group['objects'] += 1; group['tp'] += int(g[gi] >= 0)
                group['patients'].add(row['patient_id']); group['images'].add(stem)
                group['gt_errors'][str(reasons[gi])] += 1
                geo = geometry(row, gt, gi); geometries.append(geo)
                for dimension in ('size', 'multiple_same_class', 'overlap_nontext', 'aspect', 'border'):
                    value = bins[name, dimension, geo[dimension]]
                    value['objects'] += 1; value['tp'] += int(g[gi] >= 0); value['patients'].add(row['patient_id'])
            for pi, box in enumerate(pred):
                if box[4] >= .25 and p[pi] < 0:
                    group = groups[CLASSES[int(box[5])]]
                    group['fp'] += 1; group['fp_errors'][str(fp_reasons[pi])] += 1
            details[condition][stem] = {'g': g, 'p': p, 'matrix': matrix, 'geometry': geometries}
        for name, group in groups.items():
            expected = result['conditions'][condition]['operating_points']['conf0.25_iou0.5']['classes'][name]
            require(all(group[k] == expected[k] for k in ('objects', 'tp', 'fp')), 'Independent operating counts differ')
            require(len(group['patients']) == expected['patients'], 'Patient support differs')
            require(sum(group['gt_errors'].values()) == group['objects'], 'GT error categories do not partition objects')
            require(sum(group['fp_errors'].values()) == group['fp'], 'FP categories do not partition unmatched predictions')
            aggregates.append({'experiment': condition, 'class': name, 'objects': group['objects'], 'patients': len(group['patients']),
                'images': len(group['images']), 'tp': group['tp'], 'fp': group['fp'], 'fn': group['objects']-group['tp'],
                'gt_errors': dict(group['gt_errors']), 'fp_errors': dict(group['fp_errors'])})
        for (name, dimension, value), group in sorted(bins.items()):
            strata.append({'experiment': condition, 'class': name, 'dimension': dimension, 'stratum': value,
                'objects': group['objects'], 'patients': len(group['patients']), 'tp': group['tp'], 'fn': group['objects']-group['tp']})
    for other, base in (('D', 'A'), ('B', 'A'), ('C', 'A'), ('E', 'D'), ('F', 'D'), ('E', 'B'), ('F', 'C')):
        for class_id, name in enumerate(CLASSES):
            counts = Counter(); support = defaultdict(set)
            for stem, row in rows.items():
                a, b = details[base][stem]['g'], details[other][stem]['g']
                for gi in np.flatnonzero(row['gt'][:, 4] == class_id):
                    status = 'both' if a[gi] >= 0 and b[gi] >= 0 else 'recovered' if b[gi] >= 0 else 'lost' if a[gi] >= 0 else 'neither'
                    counts[status] += 1; support[status].add(row['patient_id'])
            recovery.append({'contrast': other+'-'+base, 'class': name, **{k: counts[k] for k in ('both', 'recovered', 'lost', 'neither')},
                **{k+'_patients': len(support[k]) for k in ('both', 'recovered', 'lost', 'neither')}})
    return {'errors': aggregates, 'strata': strata, 'recovery': recovery}, details
