"""Build the requested static figure set from existing files only."""

from pathlib import Path
import sys

from .data import ROOT, WORK, FIGURES, EVIDENCE, read, write_json, write_csv, sha, require, aggregates, records, saved_predictions
from .select import conversion_cases, qualitative_cases
from .verify import capture_preservation
from . import render


def update_ap_charts():
    # Read published aggregates only; no image, prediction or model access is needed.
    result, counts = aggregates()
    manifest_path = FIGURES/'manifests/source_manifest.json'
    manifest = read(manifest_path)
    for path, expected in manifest['source_files_sha256'].items():
        require(sha(ROOT/path) == expected, 'Existing figure source changed: '+path)
    for entry in manifest['figures']:
        for path, expected in entry['file_sha256'].items():
            require(sha(ROOT/path) == expected, 'Existing figure export changed: '+path)
    updates = render.charts(result, counts, ap_only=True)
    by_name = {entry['name']: entry for entry in updates}
    entries = [by_name.pop(entry['name'], entry) for entry in manifest['figures']]
    entries.extend(by_name.values())
    for entry in updates:
        entry['file_sha256'] = {path: sha(ROOT/path) for path in entry['files']}
    manifest['figures'] = entries
    manifest['renderer_source_sha256'] = {path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(manifest_path, manifest)
    write_csv(FIGURES/'manifests/figures.csv', [{'filename': path, 'description': entry['description'],
        'experiments': ';'.join(entry['experiments']), 'source_image_id': entry.get('image_id', ''),
        'data_sources': ';'.join(entry['sources']), 'sha256': entry['file_sha256'][path]}
        for entry in entries for path in entry['files']])
    # The old review cannot attest to newly exported charts; keep it as history.
    receipt_path = FIGURES/'manifests/verification.json'
    receipt = read(receipt_path)
    receipt['passed'] = False
    receipt['ap_chart_update'] = {'status': 'pending verification and visual review',
        'figures': [entry['name'] for entry in updates]}
    write_json(receipt_path, receipt)
    require('torch' not in sys.modules, 'The figure builder must not import the model runtime')
    print('Exported class-wise A-C AP50 and overall A-F mAP50 only; verify and review before publication', flush=True)


def build():
    capture_preservation()
    result, counts = aggregates()
    rows = records(); by_stem = {row['filestem']: row for row in rows}
    predictions, details, prediction_paths = saved_predictions(rows, result)
    plan_path = WORK/'selection.json'
    if plan_path.exists():
        plan = read(plan_path)
    else:
        plan = {'annotations': conversion_cases(rows), 'qualitative': qualitative_cases(rows, predictions, details)}
        write_json(plan_path, plan)
    entries, private = render.charts(result, counts), []
    for group, cases, renderer in (('annotations', plan['annotations'], render.conversion),
                                     ('qualitative', plan['qualitative'], render.qualitative)):
        for index, case in enumerate(cases, 1):
            row = by_stem[case['stem']]
            require(row['split'] == case['split'], 'Selected figure split differs')
            args = (row, case, predictions, index) if group == 'qualitative' else (row, case, index)
            public, source = renderer(*args)
            entries.append(public)
            private.append({'figure': public['files'][0], 'source_image_id': public['image_id'], **source})
            print(f"Rendered {group} {index}/{len(cases)}", flush=True)
    sources = {path for entry in entries for path in entry['sources']}
    sources.update(['docs/evidence/rarity_inference_comparison/six_condition_matrix.csv',
        'docs/evidence/rarity_inference_comparison/per_class_AP.csv',
        'outputs/rarity_comparison/study_01/analysis/D_details.json',
        'outputs/rarity_inference_comparison/study_01/analysis/E_details.json',
        'outputs/rarity_inference_comparison/study_01/analysis/F_details.json',
        'outputs/dataset_preparation/freeze.json', 'docs/evidence/dataset_audit/source_release.json'])
    for path in sources:
        require((ROOT/path).is_file(), 'Missing recorded source: '+path)
    for entry in entries:
        entry['file_sha256'] = {path: sha(ROOT/path) for path in entry['files']}
    manifest = {'created': '2026-10-20', 'scope': 'Presentation of existing evidence only; no training, inference, evaluation study or parameter search',
        'figures': entries, 'source_files_sha256': {path: sha(ROOT/path) for path in sorted(sources)},
        'renderer_source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        'rendering': {'matplotlib': render.matplotlib.__version__, 'png_dpi': 300, 'chart_svg_exports': True,
            'conversion_images': 'existing lossless 8-bit prepared training PNGs, no contrast adjustment',
            'qualitative_images': 'existing validation PNGs and preserved rarity D/E/F predictions',
            'redaction': 'all annotated text rectangles expanded and blanked in export copies; original pixels/labels unchanged',
            'qualitative_filter': 'confidence >= .25; text class omitted in all panels; same post-hoc display crop for D/E/F'},
        'limitations': ['Only periostealreaction and bonelesion have source polygons; other classes are source rectangles',
            'Cases are selected after annotation/prediction inspection for illustration, not a representative performance sample',
            'Latency uses D/E/F recorded means from separate sessions; no controlled speedup is claimed',
            'AP bars are point estimates; paired uncertainty and equal-confidence ordering caveats remain in the Phase 4E report'],
        'private_exact_source_index': 'outputs/thesis_figures/build_01/private_sources.json',
        'model_loads': 0, 'model_forwards': 0, 'heldout_image_label_prediction_files_opened': 0}
    write_json(WORK/'private_sources.json', private)
    write_json(FIGURES/'manifests/source_manifest.json', manifest)
    write_csv(FIGURES/'manifests/figures.csv', [{'filename': path, 'description': entry['description'],
        'experiments': ';'.join(entry['experiments']), 'source_image_id': entry.get('image_id', ''),
        'data_sources': ';'.join(entry['sources']), 'sha256': entry['file_sha256'][path]} for entry in entries for path in entry['files']])
    conversions = [entry for entry in entries if entry['name'].startswith('annotation_conversion_')]
    write_csv(FIGURES/'manifests/annotation_conversion_candidates.csv', [{'filename': entry['files'][0],
        'source_image_id': entry['image_id'], 'classes': ';'.join(entry['classes']), 'annotations_shown': entry['annotations_shown'],
        'source_target_annotations': entry['source_target_annotations'], 'source_annotation_sha256': entry['source_annotation_sha256'],
        'notes': 'Training image; genuine source polygons only; detail crop; annotated text redacted'} for entry in conversions])
    qualitative = [entry for entry in entries if entry['name'].startswith('qualitative_')]
    write_csv(FIGURES/'manifests/qualitative_examples.csv', [{'filename': entry['files'][0], 'source_image_id': entry['image_id'],
        'experiments': 'D;E;F', 'training': 'rarity-only', 'confidence': .25, 'selection_matching_iou': .5,
        'reason': entry['description'], 'notes': 'Same display crop across methods; text masked and not overlaid'} for entry in qualitative])
    require('torch' not in sys.modules, 'The figure builder must not import the model runtime')
    print(f"Created {len(entries)} figures; saved exact private source mapping and public hash-based manifests", flush=True)


if __name__ == '__main__':
    build()
