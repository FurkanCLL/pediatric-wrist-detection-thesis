"""Deterministic illustrative selections; these cases do not estimate performance."""

import numpy as np

from .data import CLASSES, crop_bounds, require


def conversion_cases(rows):
    chosen, used_patients = [], set()
    for class_id, count in ((5, 6), (3, 14)):
        eligible = [row for row in rows if row['split'] == 'train' and any(b['class_id'] == class_id for b in row['boxes'])]
        def area(row):
            return min((b['xyxy'][2]-b['xyxy'][0])*(b['xyxy'][3]-b['xyxy'][1])/(row['width']*row['height'])
                for b in row['boxes'] if b['class_id'] == class_id)
        ranked = sorted(eligible, key=lambda row: (area(row), row['source_image_sha256']))
        # Spread cases across annotation sizes, with one image per patient for visual variety.
        for quantile in np.linspace(.05, .95, count):
            target = round(float(quantile)*(len(ranked)-1))
            index = next(i for i in sorted(range(len(ranked)), key=lambda i: (abs(i-target), i))
                if ranked[i]['patient_id'] not in used_patients)
            row = ranked[index]
            used_patients.add(row['patient_id'])
            boxes = [b['xyxy'] for b in row['boxes'] if b['class_id'] in (3, 5)]
            chosen.append({'stem': row['filestem'], 'split': 'train',
                'reason': f"{CLASSES[class_id]} polygon; size quantile {float(quantile):.2f}",
                'crop_xyxy': crop_bounds(boxes, row['width'], row['height']), 'focus_class_id': class_id})
    require(len(chosen) == len({v['stem'] for v in chosen}) == 20, 'Need twenty different training examples')
    return chosen


def qualitative_cases(rows, predictions, details):
    ranked = sorted((row for row in rows if row['split'] == 'val'), key=lambda row: row['source_image_sha256'])
    summaries = {}
    for row in ranked:
        stem = row['filestem']
        record = {}
        for c in 'DEF':
            pred = predictions[c][stem]
            gt_map = np.asarray(details[c][stem]['g_to_p'])
            pred_map = np.asarray(details[c][stem]['p_to_g'])
            require(len(gt_map) == len(row['boxes']) and len(pred_map) == len(pred), 'Saved descriptive matching differs')
            visible = (pred[:, 4] >= .25) & (pred[:, 5] != 2)
            record[c] = {'count': int(visible.sum()), 'fp': int((visible & (pred_map < 0)).sum()), 'gt': gt_map}
        summaries[stem] = record
    used_patients, selected = set(), []
    for category in ('extra_sahi', 'adaptive_recovery', 'shared_detection'):
        eligible = []
        for row in ranked:
            if row['patient_id'] in used_patients:
                continue
            s = summaries[row['filestem']]
            nontext = [i for i, b in enumerate(row['boxes']) if b['class_id'] != 2]
            if not nontext or len(nontext) > 4 or max(s[c]['count'] for c in 'DEF') > 8:
                continue
            recovered = [i for i in nontext if s['D']['gt'][i] < 0 <= s['F']['gt'][i]]
            focus = None
            if category == 'extra_sahi':
                keep = s['E']['fp'] >= s['D']['fp']+2 and 1 <= s['D']['count'] <= 3
            elif category == 'adaptive_recovery':
                # Prefer a real periosteal recovery; fracture is a fallback if none meet display constraints.
                recovered.sort(key=lambda i: (row['boxes'][i]['class_id'] != 3, i))
                keep = bool(recovered) and s['F']['count'] <= 5
                focus = recovered[0] if recovered else None
            else:
                keep = max(s[c]['count'] for c in 'DEF') <= 3 and all(s[c]['gt'][i] >= 0 for c in 'DEF' for i in nontext)
                keep = keep and any(row['boxes'][i]['class_id'] == 0 for i in nontext)
            if keep:
                eligible.append((row, focus))
        if category == 'adaptive_recovery':
            eligible.sort(key=lambda item: (item[0]['boxes'][item[1]]['class_id'] != 3, item[0]['source_image_sha256']))
        require(eligible, 'No genuine saved example meets '+category)
        row, focus = eligible[0]
        used_patients.add(row['patient_id'])
        if focus is not None:
            boxes = [row['boxes'][focus]['xyxy']]
            crop = crop_bounds(boxes, row['width'], row['height'], minimum_fraction=.32, padding=1.)
        else:
            boxes = [b['xyxy'] for b in row['boxes'] if b['class_id'] != 2]
            boxes += [v[:4] for c in 'DEF' for v in predictions[c][row['filestem']] if v[4] >= .25 and v[5] != 2]
            crop = crop_bounds(boxes, row['width'], row['height'], minimum_fraction=.5, padding=.10)
        selected.append({'stem': row['filestem'], 'split': 'val', 'category': category,
            'focus_gt_index': focus, 'crop_xyxy': crop, 'eligible_images': len(eligible),
            'reason': {'extra_sahi': 'SAHI has at least two extra annotation-relative unmatched non-text predictions',
                'adaptive_recovery': 'A saved target is matched by F but not D at confidence .25 / IoU .50',
                'shared_detection': 'All three saved methods match the visible annotated targets'}[category]})
    return selected
