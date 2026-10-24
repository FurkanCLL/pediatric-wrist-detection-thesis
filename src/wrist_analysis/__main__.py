"""Run additive Phase 5A diagnostics without loading a model."""

import argparse
import csv
import sys

from . import evidence
from .evidence import ROOT, PUBLIC, SOURCES, read, write_new, source, require, preservation, install_guard
from .errors import errors
from .mechanism import mechanism
from .synthesis import synthesis


def analyze(attempt):
    require(attempt and all(c.islower() or c.isdigit() or c == '_' for c in attempt), 'Use a simple new attempt name')
    directory = ROOT/'outputs/phase_5a_analysis'/attempt
    require(not directory.exists(), 'Preserve existing analysis attempts')
    install_guard()
    before = preservation()
    write_new(directory/'preservation_before.json', before)
    result, rows, predictions, bundles, label_check = evidence.load_inputs()
    print('Provenance and 3,088 validation label identities checked; no image/model access', flush=True)
    summary = synthesis(result)
    error_summary, details = errors(result, rows, predictions)
    print('All six operating-point counts independently reproduced; bootstrap intervals reused', flush=True)
    diagnostic = {}
    for condition in ('C', 'F'):
        aggregate, private = mechanism(condition, bundles[condition], rows, predictions, details)
        diagnostic[condition] = aggregate
        write_new(directory/(condition+'_private_events.json'), private)
        print(condition+' saved candidate/ROI/evidence/fusion/geometry analysis completed', flush=True)
    require(diagnostic['C']['counts']['updated_recovered_TP'] == 38 and diagnostic['C']['counts']['updated_FP'] == 171,
        'C promotion evidence differs from historical findings')
    require(diagnostic['F']['counts']['updated_recovered_TP'] == 34 and diagnostic['F']['counts']['updated_FP']
            +diagnostic['F']['counts']['inherited_tp_to_fp']-diagnostic['F']['counts']['inherited_fp_to_tp'] == 136,
        'F promotion evidence differs from historical findings')
    write_new(directory/'aggregate.json', {'synthesis': summary, **error_summary, 'mechanism': diagnostic})
    after = preservation()
    require(after == before, 'Prior tracked or private experimental artifacts changed')
    require('torch' not in sys.modules, 'The analysis must not import the model runtime')
    write_new(directory/'inputs.json', {'artifacts_sha256': dict(sorted(SOURCES.items())), **label_check,
        'analysis_source_sha256': {p.relative_to(ROOT).as_posix(): evidence.sha(p) for p in sorted((ROOT/'src/wrist_analysis').glob('*.py'))},
        'historical_tracked_files_unchanged': len(before['tracked']), 'historical_private_files_unchanged': len(before['private']),
        'checkpoint_preservation': 'size and mtime only; no checkpoint opened',
        'model_loads': 0, 'model_forwards': 0, 'heldout_image_label_prediction_files_opened': 0,
        'bootstrap_recomputation': False, 'parameters_searched': 0})
    print('Phase 5A aggregates saved privately; historical preservation checks pass', flush=True)


def csv_new(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def publish(attempt):
    # Publish aggregates only; private rows contain image and patient linkage.
    directory = ROOT/'outputs/phase_5a_analysis'/attempt
    aggregate = read((directory/'aggregate.json').relative_to(ROOT).as_posix())
    inputs = read((directory/'inputs.json').relative_to(ROOT).as_posix())
    require(not PUBLIC.exists(), 'Preserve an existing published analysis')
    PUBLIC.mkdir(parents=True)
    csv_new(PUBLIC/'six_condition_summary.csv', aggregate['synthesis']['six_conditions'])
    csv_new(PUBLIC/'paired_AP_contrasts.csv', aggregate['synthesis']['paired_AP_contrasts'])
    csv_new(PUBLIC/'map_components.csv', aggregate['synthesis']['map_components'])
    csv_new(PUBLIC/'recovery.csv', aggregate['recovery'])
    csv_new(PUBLIC/'error_strata.csv', aggregate['strata'])
    write_new(PUBLIC/'error_categories.json', aggregate['errors'])
    write_new(PUBLIC/'v2_mechanism.json', aggregate['mechanism'])
    for key in ('artifacts_sha256',):
        inputs[key] = dict(sorted(inputs[key].items()))
    inputs['scope'] = 'Retrospective validation diagnostics; no V3 implementation or performance experiment'
    inputs['private_aggregate_sha256'] = evidence.sha(directory/'aggregate.json')
    inputs['reproduction'] = 'python -m wrist_analysis analyze --attempt NEW_NAME; python -m wrist_analysis publish --attempt NEW_NAME (publication refuses an existing directory)'
    write_new(PUBLIC/'source_manifest.json', inputs)
    print('Published aggregate tables and diagnostic evidence only', flush=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=('analyze', 'publish'))
parser.add_argument('--attempt', default='analysis_01')
args = parser.parse_args()
if args.stage == 'analyze':
    analyze(args.attempt)
else:
    publish(args.attempt)
