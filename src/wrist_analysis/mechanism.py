"""Retrospective CG-ALR V2 accounting from sealed execution records."""

from collections import Counter, defaultdict

import numpy as np

from wrist_eval.metrics import ious, match
from .calculations import contained, feature_groups, oracle_capacity, summarize, superiority
from .evidence import CLASSES, require

FEATURES = ('global_confidence', 'local_confidence', 'global_local_iou', 'short_768', 'relative_area', 'score_delta', 'confident_context')


def mechanism(condition, bundles, rows, predictions, details):
    base = 'A' if condition == 'C' else 'D'
    counts = Counter(); candidate_classes = Counter(); local_reasons = Counter(); selection = Counter()
    roi_hist = Counter(); stages = defaultdict(Counter); upper = defaultdict(Counter)
    candidate_events, update_events, roi_events, geometry_events, object_events = [], [], [], [], []
    for bundle in bundles:
        stem = bundle['stem']; row = rows[stem]; gt = row['gt']
        original, output = predictions[base][stem], predictions[condition][stem]
        before, after = details[base][stem], details[condition][stem]
        plan = bundle['plan']; diag = bundle['variants']['V2']['diagnostics']
        rois = {roi['id']: roi for roi in plan['rois']}
        candidates = {c['global_id']: c for c in plan['candidates']}
        evidence = diag['evidence']; matches = {p['global_id']: p for p in diag['matches']}
        owners = {p['evidence_id']: p for p in diag['matches']}
        ids = diag['output_global_ids']; out_index = {g: i for i, g in enumerate(ids)}
        require(len(ids) == len(original) == len(output) and sorted(ids) == list(range(len(original))), 'Output identity differs')
        require(np.array_equal(output[:, 5], original[ids, 5]), 'Global classes changed')
        roi_hist[len(rois)] += 1
        counts['images'] += 1; counts['local_calls'] += len(rois)
        counts['candidates'] += len(candidates); counts['retained_evidence'] += len(evidence)
        local_reasons.update(v['reason'] for v in diag['local_decisions'])
        selection.update(v['action'] for v in plan['selection_decisions'])
        candidate_classes.update(CLASSES[v['class_id']] for v in candidates.values())
        counts['candidates_below_fixed_cap_recovery_floor'] += sum(c['confidence'] < .15 for c in candidates.values())
        decision = {v['global_id']: v['action'] for v in plan['selection_decisions']}
        for c in candidates.values():
            candidate_events.append({'stem': stem, 'patient_id': row['patient_id'], 'condition': condition,
                'class': CLASSES[c['class_id']], 'confidence': c['confidence'], 'short_768': c['short_768'],
                'score': c['score'], 'selection': decision.get(c['global_id'], c['geometry_rejection'])})
        counts['pair_candidates'] += len(diag['pairs'])
        counts['rejected_pairs'] += sum(not p['accepted'] for p in diag['pairs'])
        counts['mutable_matches'] += sum(p['mutable'] for p in diag['matches'])
        counts['protected_matches'] += sum(not p['mutable'] for p in diag['matches'])
        counts['unmatched_evidence'] += len(evidence)-len(diag['matches'])
        counts['mutable_pair_blocked_by_protected_owner'] += sum(p['mutable'] and not p['accepted']
            and p['evidence_id'] in owners and not owners[p['evidence_id']]['mutable'] for p in diag['pairs'])
        for ei, item in enumerate(evidence):
            same = gt[:, 4] == item['row'][5]
            aligns = bool(ious([item['row'][:4]], gt[same, :4]).max(initial=0) >= .5)
            counts['retained_evidence_aligned_to_annotation'] += int(aligns)
            counts['unassigned_evidence_aligned_to_annotation'] += int(aligns and ei not in owners)
        changes_by_roi = defaultdict(list); recovery_by_roi = defaultdict(list)
        for change in diag['score_changes']:
            g = change['global_id']; p = matches[g]; box = original[g]; class_id = int(box[5])
            roi = rois[p['roi_id']]; w, h = box[2]-box[0], box[3]-box[1]
            centers = (original[:, :2]+original[:, 2:4])/2
            context = ((original[:, 4] >= .25) & (centers[:, 0] >= roi['box'][0]) & (centers[:, 0] <= roi['box'][2])
                & (centers[:, 1] >= roi['box'][1]) & (centers[:, 1] <= roi['box'][3]))
            pi = out_index[g]; gi = int(after['p'][pi])
            crossed = change['before'] < .25 <= change['after']
            # A promoted row may claim an already-detected target and displace an old TP.
            outcome = ('recovered_TP' if crossed and gi >= 0 and before['g'][gi] < 0 else
                       'reassigned_TP' if crossed and gi >= 0 else 'FP' if crossed else 'below_threshold')
            event = {'condition': condition, 'stem': stem, 'patient_id': row['patient_id'], 'class': CLASSES[class_id],
                'outcome': outcome, 'global_confidence': float(box[4]), 'local_confidence': p['local_confidence'],
                'global_local_iou': p['iou'], 'short_768': min(float(w), float(h))*768/max(row['width'], row['height']),
                'relative_area': float(w*h)/(row['width']*row['height']), 'score_delta': change['after']-change['before'],
                'confident_context': int(context.sum())}
            update_events.append(event); changes_by_roi[p['roi_id']].append(g)
            counts['score_updates'] += 1; counts['updated_'+outcome] += 1
            if crossed and gi >= 0 and before['g'][gi] < 0:
                recovery_by_roi[p['roi_id']].append(gi)
        for g in np.flatnonzero(original[:, 4] >= .25):
            was_tp, now_tp = before['p'][g] >= 0, after['p'][out_index[g]] >= 0
            counts['inherited_tp_to_fp'] += int(was_tp and not now_tp)
            counts['inherited_fp_to_tp'] += int(not was_tp and now_tp)
        for roi_id, roi in rois.items():
            retained = [v for v in evidence if v['roi_id'] == roi_id]
            mutable = [v for v in diag['matches'] if v['roi_id'] == roi_id and v['mutable']]
            seeds = [candidates[g] for g in roi['seed_ids']]
            recovered = recovery_by_roi[roi_id]
            values = {'condition': condition, 'stem': stem, 'patient_id': row['patient_id'], 'roi_id': roi_id,
                'retained_evidence': len(retained), 'mutable_matches': len(mutable), 'updates': len(changes_by_roi[roi_id]),
                'recovered_targets': len(recovered), 'productive': bool(changes_by_roi[roi_id]),
                'seed_class': '+'.join(str(c) for c in sorted({v['class_id'] for v in seeds})),
                'max_seed_confidence': max(v['confidence'] for v in seeds),
                'min_seed_short_768': min(v['short_768'] for v in seeds), 'max_seed_score': max(v['score'] for v in seeds),
                'ideal_zoom': roi['ideal_zoom']}
            centers = (original[:, :2]+original[:, 2:4])/2
            values['confident_context'] = int(((original[:, 4] >= .25) & (centers[:, 0] >= roi['box'][0])
                & (centers[:, 0] <= roi['box'][2]) & (centers[:, 1] >= roi['box'][1]) & (centers[:, 1] <= roi['box'][3])).sum())
            roi_events.append(values)
            counts['rois_without_retained_evidence'] += int(not retained)
            counts['rois_without_score_update'] += int(not changes_by_roi[roi_id])
            counts['rois_contributing_recovery'] += int(bool(recovered))
        counts['roi_images_without_score_update'] += int(bool(rois) and not any(changes_by_roi.values()))
        for geo in diag['geometry']:
            counts['geometry_'+geo['reason']] += 1
            if geo['reason'] != 'accepted':
                continue
            g = geo['global_id']; class_id = int(original[g, 5]); same = gt[:, 4] == class_id
            a = float(ious([geo['before']], gt[same, :4]).max(initial=0))
            b = float(ious([geo['after']], gt[same, :4]).max(initial=0))
            geometry_events.append({'condition': condition, 'stem': stem, 'patient_id': row['patient_id'],
                'class': CLASSES[class_id], 'before_iou': a, 'after_iou': b, 'iou_delta': b-a,
                'status': 'improved' if b > a else 'worsened' if b < a else 'equal',
                'above_operating_confidence': bool(output[out_index[g], 4] >= .25)})
        # Attribute operating-point effects to geometry using the already-frozen V1 rule.
        # This reconstructs confidence-only V1 rows; it is not a V3 experiment or AP estimate.
        confidence_only = output.copy(); confidence_only[:, :4] = original[ids, :4]
        v1_g, _, _ = match(gt, confidence_only)
        counts['geometry_only_recovered_targets'] += int(((v1_g < 0) & (after['g'] >= 0)).sum())
        counts['geometry_only_lost_targets'] += int(((v1_g >= 0) & (after['g'] < 0)).sum())
        local_rows = np.asarray([v['row'] for v in evidence], np.float32).reshape(-1, 6)
        local_matrix = ious(gt[:, :4], local_rows[:, :4])
        for class_id in (0, 3):
            name = CLASSES[class_id]
            missed = np.flatnonzero((gt[:, 4] == class_id) & (before['g'] < 0))
            available = np.flatnonzero((original[:, 5] == class_id) & (before['p'] < 0) & (original[:, 4] < .25))
            eligible = [g for g in available if g in candidates]
            covered = [g for g in eligible if any(contained(r['box'], candidates[g]['envelope']) for r in rois.values())]
            cap_reachable = [g for g in covered if float(original[g, 4])+.1 >= .25]
            for label, proposal_ids in (('all_retained_weak_global', available), ('eligible_global_band', eligible),
                                        ('selected_roi_covered', covered), ('fixed_cap_reachable', cap_reachable)):
                upper[name][label] += oracle_capacity(before['matrix'], missed, proposal_ids)
            unused_local = [e for e, v in enumerate(evidence) if e not in owners and int(v['row'][5]) == class_id]
            upper[name]['unassigned_saved_local_evidence'] += oracle_capacity(local_matrix, missed, unused_local)
            for gi in missed:
                adequate = [g for g in candidates if int(original[g, 5]) == class_id and before['matrix'][gi, g] >= .5]
                within = [g for g in adequate if any(contained(r['box'], candidates[g]['envelope']) for r in rois.values())]
                matched = [g for g in within if g in matches and matches[g]['mutable']]
                stage = ('recovered' if after['g'][gi] >= 0 else 'no_eligible_global_iou05_anchor' if not adequate else
                    'eligible_anchor_outside_selected_roi' if not within else 'covered_without_mutable_match' if not matched else
                    'matched_without_operating_recovery')
                stages[name][stage] += 1
                any_global = bool(np.any((original[:, 5] == class_id) & (before['matrix'][gi] >= .5)))
                local_gt = bool(np.any((local_rows[:, 5] == class_id) & (local_matrix[gi] >= .5)))
                object_events.append({'condition': condition, 'stem': stem, 'patient_id': row['patient_id'], 'class': name,
                    'stage': stage, 'any_retained_global_iou05': any_global, 'saved_retained_local_iou05': local_gt,
                    'gt_fully_inside_selected_roi': any(contained(r['box'], gt[gi, :4]) for r in rois.values())})
    require(sum(roi_hist.values()) == 3088 and sum(n*count for n, count in roi_hist.items()) == counts['local_calls'], 'ROI accounting differs')
    require(sum(counts['updated_'+key] for key in ('recovered_TP', 'reassigned_TP', 'FP', 'below_threshold')) == counts['score_updates'], 'Update outcomes do not partition')
    result = {'condition': condition, 'counts': dict(counts), 'roi_count_images': dict(roi_hist),
        'candidate_classes': dict(candidate_classes), 'local_decisions': dict(local_reasons), 'selection_decisions': dict(selection),
        'missed_target_stages': {k: dict(v) for k, v in stages.items()},
        'oracle_capacities': {k: dict(v) for k, v in upper.items()},
        'candidate_features': feature_groups(candidate_events, ('class', 'selection'), ('confidence', 'short_768', 'score')),
        'update_features': feature_groups(update_events, ('class', 'outcome'), FEATURES),
        'roi_features': feature_groups(roi_events, ('productive', 'seed_class'), ('max_seed_confidence', 'min_seed_short_768', 'max_seed_score', 'ideal_zoom', 'confident_context')),
        'geometry': feature_groups(geometry_events, ('class', 'status', 'above_operating_confidence'), ('iou_delta',)),
        'object_stage_support': feature_groups(object_events, ('class', 'stage'),
            ('any_retained_global_iou05', 'saved_retained_local_iou05', 'gt_fully_inside_selected_roi')),
        'promotion_feature_separation': []}
    for name in ('overall', 'fracture', 'periostealreaction'):
        selected = [v for v in update_events if name == 'overall' or v['class'] == name]
        tp, fp = [v for v in selected if v['outcome'] == 'recovered_TP'], [v for v in selected if v['outcome'] == 'FP']
        for feature in FEATURES:
            a, b = [v[feature] for v in tp], [v[feature] for v in fp]
            result['promotion_feature_separation'].append({'class': name, 'feature': feature,
                'TP': summarize(a), 'FP': summarize(b), 'probability_TP_feature_exceeds_FP': superiority(a, b)})
    return result, {'candidates': candidate_events, 'updates': update_events, 'rois': roi_events,
                    'geometry': geometry_events, 'objects': object_events}
