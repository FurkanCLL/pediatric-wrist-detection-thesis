"""Reuse original six-condition estimates and patient-paired intervals."""

from .evidence import CLASSES, read, require

CONTRASTS = {'D-A': {'D': 1, 'A': -1}, 'B-A': {'B': 1, 'A': -1}, 'C-A': {'C': 1, 'A': -1},
    'E-D': {'E': 1, 'D': -1}, 'F-D': {'F': 1, 'D': -1}, 'E-B': {'E': 1, 'B': -1}, 'F-C': {'F': 1, 'C': -1},
    '(E-D)-(B-A)': {'E': 1, 'D': -1, 'B': -1, 'A': 1},
    '(F-D)-(C-A)': {'F': 1, 'D': -1, 'C': -1, 'A': 1}}


def synthesis(result):
    old_sahi = read('docs/evidence/error_analysis_sahi/patient_bootstrap.json')
    old_adaptive = read('docs/evidence/adaptive_refinement_validation/patient_bootstrap.json')
    old_training = read('docs/evidence/rarity_comparison/patient_bootstrap.json')
    intervals = {**result['bootstrap']['contrasts'],
        'D-A': old_training['intervals']['D_minus_A'],
        'B-A': old_sahi['paired_ap_differences']['sahi_context'],
        'C-A': old_adaptive['comparisons']['V2_minus_standard_serial']}
    summary, contrasts = [], []
    for condition, value in result['conditions'].items():
        point = value['operating_points']['conf0.25_iou0.5']
        tp, fp, fn = (point['total_'+key] for key in ('tp', 'fp', 'fn'))
        summary.append({'experiment': condition, **value['AP']['overall'], 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp), 'recall': tp/(tp+fn),
            'mean_model_calls': value['source']['mean_passes'],
            'recorded_mean_ms': value['source']['timing']['e2e_seconds']['mean']*1000})
    for contrast, coefficients in CONTRASTS.items():
        for name in (*CLASSES, 'overall'):
            for metric in ('ap50', 'ap75', 'ap50_95'):
                key = 'm'+metric if name == 'overall' else metric
                delta = sum(weight*result['conditions'][c]['AP'][name][key] for c, weight in coefficients.items())
                interval = intervals[contrast][name][key]
                require(interval['replicates_present'] == (492 if name in ('bonelesion', 'overall') else 500), 'Bootstrap support differs')
                contrasts.append({'contrast': contrast, 'class': name, 'metric': metric, 'point_delta': delta,
                    'lower': interval['lower'], 'upper': interval['upper'], 'valid_replicates': interval['replicates_present']})
    # Attribution uses arithmetic components, not a replacement class-subset mAP.
    components = []
    for contrast, coefficients in CONTRASTS.items():
        for name in CLASSES:
            delta = sum(weight*result['conditions'][c]['AP'][name]['ap50_95'] for c, weight in coefficients.items())
            components.append({'contrast': contrast, 'class': name, 'class_ap50_95_delta': delta,
                'contribution_to_official_eight_class_map_delta': delta/8,
                'group': 'text_annotation' if name == 'text' else 'metal_or_device' if name == 'metal' else 'other_finding'})
    return {'six_conditions': summary, 'paired_AP_contrasts': contrasts, 'map_components': components,
        'bootstrap_recomputed': False, 'bootstrap_replicates_reused': 500}
