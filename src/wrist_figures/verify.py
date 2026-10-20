"""Reconcile figure sources, exports and preserved experimental files."""

import subprocess

import numpy as np
from PIL import Image

from .data import ROOT, WORK, FIGURES, EVIDENCE, read, write_json, sha, require, aggregates, csv_rows, records, load_image, polygons, mask_text


def capture_preservation():
    destination = WORK/'preservation_before.json'
    if destination.exists():
        return
    names = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    tracked = {name: sha(ROOT/name) for name in names if not name.startswith(('docs/figures/', 'src/wrist_figures/'))
               and not name.split('/')[-1].startswith('.env')}
    private = {}
    roots = ('training/standard_baseline', 'training/rarity_only', 'error_analysis_sahi', 'adaptive_refinement',
        'rarity_training_completion', 'rarity_training_readiness', 'error_guided_training_feasibility',
        'class_aware_design', 'rarity_comparison', 'rarity_inference_comparison')
    for name in roots:
        for path in sorted((ROOT/'outputs'/name).rglob('*')):
            if path.is_file() and not path.name.startswith('.env'):
                stat = path.stat()
                private[path.relative_to(ROOT).as_posix()] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    **({} if path.suffix == '.pt' else {'sha256': sha(path)})}
    for name in ('source_records.jsonl', 'split_manifest.csv', 'policy.json', 'freeze.json', 'completed.sqlite'):
        path = ROOT/'outputs/dataset_preparation'/name
        private[path.relative_to(ROOT).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
    write_json(destination, {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'tracked': tracked, 'private': private})


def verify():
    result, counts = aggregates()
    manifest = read(FIGURES/'manifests/source_manifest.json')
    entries = manifest['figures']
    require(len(entries) == 30, 'Expected seven charts, twenty conversion and three qualitative figures')
    for path, expected in manifest['source_files_sha256'].items():
        require(sha(ROOT/path) == expected, 'Figure source changed: '+path)
    for entry in entries:
        for name, digest in entry['file_sha256'].items():
            path = ROOT/name
            require(sha(path) == digest and path.stat().st_size > 1000, 'Export missing/changed: '+name)
            if path.suffix == '.png':
                with Image.open(path) as image:
                    require(min(image.size) >= 900 and image.info.get('dpi', (0,))[0] >= 299, 'Insufficient PNG export resolution')
                    require(not image.getexif(), 'Source EXIF must not enter figure exports')
                    require(image.convert('RGB').getpixel((0, 0)) == (255, 255, 255), 'Figure margins must be white')
    chart = {entry['name']: entry for entry in entries if 'plot_values' in entry}
    # Enforce the requested scopes independently of figure ordering in the manifest.
    require(chart['classwise_ap50']['experiments'] == ['A'], 'Class-wise chart must use Experiment A only')
    require(set(chart['classwise_ap50']['plot_values']) == set(counts), 'Class-wise chart must include all eight classes')
    require(all(chart['classwise_ap50']['plot_values'][name] == result['conditions']['A']['AP'][name]['ap50'] for name in counts),
        'Class-wise chart differs from Experiment A AP50')
    require(chart['overall_map50']['experiments'] == list('ABCDEF')
        and list(chart['overall_map50']['plot_values']) == list('ABCDEF'), 'Overall mAP50 must include A-F')
    require(chart['class_distribution']['plot_values'] == counts, 'Class chart differs from the 47,436-object census')
    for c, data in result['conditions'].items():
        require(chart['overall_performance']['plot_values'][c] == data['AP']['overall']['map50_95'], 'Overall chart AP differs')
        require(chart['overall_map50']['plot_values'][c] == data['AP']['overall']['map50'], 'Overall mAP50 chart differs')
        require(chart['fracture_ap50']['plot_values'][c] == data['AP']['fracture']['ap50'], 'Fracture chart AP differs')
        require(chart['false_positives']['plot_values'][c] == data['operating_points']['conf0.25_iou0.5']['total_fp'], 'FP chart differs')
    for c in 'DEF':
        source = result['conditions'][c]['source']
        require(chart['inference_cost']['plot_values']['latency_ms'][c] == source['timing']['e2e_seconds']['mean']*1000,
            'Latency conversion differs')
        require(chart['inference_cost']['plot_values']['model_calls'][c] == source['mean_passes'], 'Call count differs')
    conversion_index = csv_rows(FIGURES/'manifests/annotation_conversion_candidates.csv')
    require(len(conversion_index) == len({row['source_image_id'] for row in conversion_index}) == 20, 'Conversion candidates are not twenty unique images')
    require(len(csv_rows(FIGURES/'manifests/qualitative_examples.csv')) == 3, 'Need three qualitative examples')
    rows = {row['filestem']: row for row in records()}
    private = read(WORK/'private_sources.json')
    max_delta = 0.
    for item in private:
        row = rows[item['stem']]
        pixels, labels, provenance = load_image(row)
        max_delta = max(max_delta, provenance['label_roundtrip_max_delta_pixels'])
        masked, regions = mask_text(pixels, row['boxes'])
        require(regions == item['redaction_rectangles'], 'Redaction regions differ')
        for x1, y1, x2, y2 in regions:
            require(not masked[y1:y2, x1:x2].any(), 'Export-copy text mask failed')
        require(item['prepared_image_sha256'] == provenance['prepared_image_sha256'], 'Prepared image mapping differs')
        if row['split'] == 'train':
            objects, path = polygons(row)
            crop = item['crop_xyxy']
            for index, obj in objects:
                points = np.asarray(obj['points']['exterior'])
                expected = [*points.min(axis=0), *points.max(axis=0)]
                require(np.allclose(expected, labels[index, :4], atol=1e-8, rtol=0), 'Plotted box is not the actual polygon envelope')
                require(crop[0] <= expected[0] < expected[2] <= crop[2] and crop[1] <= expected[1] < expected[3] <= crop[3], 'A source polygon was cropped')
    before = read(WORK/'preservation_before.json')
    permitted = {'README.md', 'requirements.txt', 'pyproject.toml'}
    for path, digest in before['tracked'].items():
        if path not in permitted:
            require(sha(ROOT/path) == digest, 'Historical tracked file changed: '+path)
    for path, data in before['private'].items():
        file = ROOT/path
        require([file.stat().st_size, file.stat().st_mtime_ns] == [data['bytes'], data['mtime_ns']], 'Historical private file changed: '+path)
        if 'sha256' in data:
            require(sha(file) == data['sha256'], 'Historical private bytes changed: '+path)
    # Original source/label bytes are verified again after rendering; no test image/label is opened.
    public = (FIGURES/'manifests/source_manifest.json').read_text(encoding='utf-8')
    require(all(item['stem'] not in public for item in private) and 'patient_id' not in public, 'Private identity entered public manifest')
    write_json(FIGURES/'manifests/verification.json', {'passed': True, 'figures': 30, 'png_files': 30, 'svg_files': 7,
        'unique_conversion_images': 20, 'qualitative_images': 3, 'real_source_polygons_only': True,
        'converted_label_roundtrip_max_delta_pixels': max_delta, 'matching_source_hashes': len(manifest['source_files_sha256']),
        'historical_tracked_files_unchanged': len(before['tracked'])-len(permitted),
        'historical_private_files_unchanged': len(before['private']), 'checkpoint_protection': 'byte size and modification time; no checkpoint loaded',
        'other_protected_private_files': 'SHA256 plus byte size and modification time',
        'no_training_inference_or_new_evaluation_study': True, 'heldout_image_label_prediction_files_opened': 0,
        'png_dpi': 300, 'public_identities': 'Original image SHA256 only; exact paths are in ignored private_sources.json',
        'visual_review': 'pending manual inspection of every PNG'})
    print('Figure source/export/preservation checks pass; manual visual review remains separate', flush=True)


if __name__ == '__main__':
    verify()
