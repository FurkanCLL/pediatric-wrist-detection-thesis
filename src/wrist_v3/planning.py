"""Prediction-only candidate filtering and bounded contextual ROI selection."""

from wrist_eval.adaptive import plan_rois as legacy_plan, contains, overlap, checked_rows


def evaluate_candidates(global_rows, image_hw, legacy, settings):
    height, width = map(int, image_hw)
    global_rows = checked_rows(global_rows, width, height)
    original = legacy_plan(global_rows, image_hw, legacy)
    protected = [i for i, row in enumerate(global_rows)
                 if int(row[5]) in legacy['candidates']['class_ids'] and row[4] >= settings['protected_confidence']]
    evaluation, eligible = [], []
    for candidate in original['candidates']:
        g = candidate['global_id']; row = global_rows[g]
        conflicts = [p for p in protected if global_rows[p, 5] == row[5]
                     and overlap(global_rows[p, :4], row[:4]) >= settings['ownership_iou']]
        reason = 'protected_conflict' if conflicts else candidate['geometry_rejection'] or 'eligible'
        evaluation.append({'global_id': g, 'reason': reason, 'protected_ids': conflicts})
        if reason == 'eligible':
            eligible.append(candidate)
    return original, eligible, evaluation


def plan_rois(global_rows, image_hw, legacy, settings):
    # Keep V2 crop geometry and ranking, removing only candidates forbidden by V3 protection.
    original, eligible, evaluation = evaluate_candidates(global_rows, image_hw, legacy, settings)
    key = lambda c: (-c['score'], -c['confidence'], c['short_768'], c['global_id'])
    ordered = sorted(eligible, key=key)
    heads = [next((c for c in ordered if c['class_id'] == class_id), None)
             for class_id in legacy['candidates']['class_ids']]
    heads = [c for c in heads if c is not None]
    head_ids = {c['global_id'] for c in heads}
    sequence = sorted(heads, key=key)+[c for c in ordered if c['global_id'] not in head_ids]
    rois, decisions = [], []
    for candidate in sequence:
        shared = [(overlap(roi['box'], candidate['crop']), roi['id']) for roi in rois
                  if contains(roi['box'], candidate['envelope'])]
        shared = sorted((value for value in shared if value[0] >= legacy['selection']['roi_overlap_iou_min']),
                        key=lambda value: (-value[0], value[1]))
        if shared:
            roi_id = shared[0][1]; action = 'coalesced'
            rois[roi_id]['seed_ids'].append(candidate['global_id'])
        elif len(rois) < settings['maximum_local_views']:
            roi_id = len(rois); action = 'selected'; box = candidate['crop']
            rois.append({'id': roi_id, 'box': box, 'seed_ids': [candidate['global_id']],
                         'ideal_zoom': max(image_hw)/max(box[2]-box[0], box[3]-box[1])})
        else:
            roi_id, action = None, 'budget_exhausted'
        decisions.append({'global_id': candidate['global_id'], 'action': action, 'roi_id': roi_id})
    return {'image_hw': list(map(int, image_hw)), 'candidates': eligible,
        'all_candidates': original['candidates'], 'candidate_evaluation': evaluation,
        'excluded': original['excluded'], 'rois': rois, 'selection_decisions': decisions}
