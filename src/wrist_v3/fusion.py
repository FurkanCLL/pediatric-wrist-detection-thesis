"""Protected evidence ownership and conservative two-view score arbitration."""

from collections import defaultdict
import numpy as np

from wrist_eval.adaptive import checked_rows, local_evidence, contains, overlap


def match_evidence(global_rows, evidence, plan, legacy, settings):
    protected = [i for i, row in enumerate(global_rows) if int(row[5]) in legacy['candidates']['class_ids']
                 and row[4] >= settings['protected_confidence']]
    reserved = {}
    for e, item in enumerate(evidence):
        owners = [g for g in protected if global_rows[g, 5] == item['row'][5]
                  and overlap(global_rows[g, :4], item['row'][:4]) >= settings['ownership_iou']]
        if owners:
            reserved[e] = owners
    roi_by_id = {roi['id']: roi for roi in plan['rois']}
    by_global, by_local, pairs = defaultdict(list), defaultdict(list), {}
    for candidate in plan['candidates']:
        g = candidate['global_id']; row = global_rows[g]
        for e, item in enumerate(evidence):
            if e in reserved or row[5] != item['row'][5]:
                continue
            if not contains(roi_by_id[item['roi_id']]['box'], candidate['envelope']):
                continue
            value = overlap(row[:4], item['row'][:4])
            if value >= settings['ownership_iou']:
                by_global[g].append(e); by_local[e].append(g)
                pairs[g, e] = {'global_id': g, 'evidence_id': e, 'iou': value,
                    'roi_id': item['roi_id'], 'local_id': item['local_id'], 'local_confidence': item['row'][4]}
    matches, decisions = [], []
    for candidate in plan['candidates']:
        g = candidate['global_id']; neighbors = by_global[g]
        covered = any(contains(roi['box'], candidate['envelope']) for roi in plan['rois'])
        reason = ('no_selected_roi_coverage' if not covered else 'no_unreserved_compatible_evidence' if not neighbors else
                  'ambiguous_global_degree' if len(neighbors) != 1 else
                  'ambiguous_local_degree' if len(by_local[neighbors[0]]) != 1 else 'unique_pair')
        decisions.append({'global_id': g, 'reason': reason, 'evidence_ids': neighbors,
                          'local_degrees': [len(by_local[e]) for e in neighbors]})
        if reason == 'unique_pair':
            matches.append(pairs[g, neighbors[0]])
    return matches, {'reserved_evidence': [{'evidence_id': e, 'protected_ids': owners} for e, owners in reserved.items()],
                     'candidate_matching': decisions, 'eligible_graph_edges': len(pairs)}


def confidence_update(original, local, settings, legacy_cap=False):
    # Quantize down so float32 cannot exceed the documented mathematical cap.
    confirmation = float(local) if legacy_cap else ((1-settings['local_score_weight'])*float(original)
                                                    +settings['local_score_weight']*float(local))
    cap = min(float(original)+settings['maximum_positive_delta'], confirmation, settings['maximum_updated_confidence'])
    value = np.float32(cap)
    if float(value) > cap:
        value = np.nextafter(value, np.float32(-np.inf))
    return value


def construct_output(original, output):
    if not np.array_equal(original[:, :4], output[:, :4]) or not np.array_equal(original[:, 5], output[:, 5]):
        raise AssertionError('V3 changed coordinates or class IDs')
    ids = list(range(len(output)))
    if not np.array_equal(original, output):
        ids.sort(key=lambda i: (-float(output[i, 4]), i))
    return output[ids], ids


def fuse_predictions(global_predictions, local_views, plan, legacy, settings, legacy_cap=False):
    height, width = plan['image_hw']
    original = checked_rows(global_predictions, width, height); output = original.copy()
    evidence, local_decisions = local_evidence(local_views, plan, legacy)
    matches, matching = match_evidence(original, evidence, plan, legacy, settings)
    updates, arbitration = [], []
    for pair in sorted(matches, key=lambda p: (-float(original[p['global_id'], 4]), p['global_id'])):
        g = pair['global_id']; row = original[g]
        conflicts = [j for j, other in enumerate(output) if j != g and other[5] == row[5]
                     and other[4] >= settings['protected_confidence']
                     and overlap(row[:4], other[:4]) >= settings['ownership_iou']]
        if conflicts:
            arbitration.append({'global_id': g, 'reason': 'operating_conflict', 'conflicting_ids': conflicts})
            continue
        value = confidence_update(row[4], pair['local_confidence'], settings, legacy_cap)
        if value < row[4]:
            raise AssertionError('V3 reduced an original score')
        output[g, 4] = value
        if value != row[4]:
            updates.append({'global_id': g, 'before': float(row[4]), 'after': float(value), **pair})
        arbitration.append({'global_id': g, 'reason': 'updated' if value != row[4] else 'quantized_unchanged'})
    mutable = {c['global_id'] for c in plan['candidates']}
    protected = [i for i in range(len(original)) if i not in mutable]
    if not np.array_equal(original[protected], output[protected]):
        raise AssertionError('Protected global prediction changed')
    result, ids = construct_output(original, output)
    checked_rows(result, width, height)
    if not np.array_equal(result[result[:, 4] >= .5], original[original[:, 4] >= .5]):
        raise AssertionError('V3 changed confidence-.50 output')
    return {'predictions': result, 'diagnostics': {'output_global_ids': ids, 'evidence': evidence,
        'local_decisions': local_decisions, 'matches': matches, **matching, 'score_changes': updates,
        'arbitration': arbitration, 'geometry': 'unchanged'}}
