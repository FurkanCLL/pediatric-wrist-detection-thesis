"""Check plotted values, exports, privacy and historical preservation."""

import argparse
import csv
import hashlib
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .data import ROOT, OUT, WORK, CLASSES, METHODS, read, sha, write_json, require, load


def verify_values(manifest, data):
    entries = {v['name'].split('/')[-1]: v for v in manifest['figures']}
    test = data['results']['test']
    for prefix in ('train', 'val'):
        values = entries[prefix+'_losses']['plotted_values']
        for key in ('box_loss', 'cls_loss', 'dfl_loss'):
            for strategy, history in data['histories'].items():
                require(values[prefix+'/'+key][strategy] == [r[prefix+'/'+key] for r in history['rows']], 'Loss curve differs')
    values = entries['native_validation_metrics']['plotted_values']
    for key in ('precision', 'recall', 'mAP50', 'mAP50-95'):
        for strategy, history in data['histories'].items():
            require(values[key][strategy] == [r['metrics/'+key+'(B)']*100 for r in history['rows']], 'Native metric curve differs')
    require(entries['test_map50_95']['plotted_values'] == {m: test['AP'][m]['overall']['map50_95']*100 for m in METHODS}, 'Test AP plot differs')
    for split in ('validation', 'test'):
        require(entries['validation_test_map50_95']['plotted_values'][split] == [data['results'][split]['AP'][m]['overall']['map50_95']*100 for m in METHODS], 'Cohort AP plot differs')
    require(entries['class_ap50_95_heatmap']['plotted_values']['AP_percent'] == [[test['AP'][m][c]['ap50_95']*100 for m in METHODS] for c in CLASSES], 'Heatmap differs')
    for key in ('ap50', 'ap75', 'ap50_95'):
        require(entries['fracture_ap']['plotted_values'][key] == [test['AP'][m]['fracture'][key]*100 for m in METHODS], 'Fracture AP differs')
    require(entries['false_positives']['plotted_values'] == {m: test['operating'][m]['overall']['fp'] for m in METHODS}, 'FP chart differs')
    for key in ('mean', 'p95', 'calls'):
        actual = [test['runtime'][m]['timing']['forward_passes']['mean'] if key == 'calls' else test['runtime'][m]['timing']['e2e_seconds'][key]*1000 for m in METHODS]
        require(entries['latency_and_calls']['plotted_values'][key] == actual, 'Cost panel differs')
    for row in entries['main_paired_differences']['plotted_values']:
        ci = test['bootstrap']['contrasts'][row['contrast']]['overall']['map50_95']
        require(row['delta_pp'] == test['AP_contrasts'][row['contrast']]['overall']['map50_95']*100
            and row['lower_pp'] == ci['lower']*100 and row['upper_pp'] == ci['upper']*100, 'Forest interval differs')
    for row in entries['rarity_class_differences']['plotted_values']:
        ci = test['bootstrap']['contrasts']['training_standard'][row['class']]['ap50_95']
        require(row['delta_pp'] == test['AP_contrasts']['training_standard'][row['class']]['ap50_95']*100
            and row['lower_pp'] == ci['lower']*100 and row['upper_pp'] == ci['upper']*100, 'Class delta interval differs')
    change = entries['recovery_fp_tradeoff']['plotted_values']
    for i, (other, base) in enumerate((('B', 'A'), ('C', 'A'), ('E', 'D'), ('F', 'D'))):
        recovery = test['recovery']['final_'+other+'-final_'+base]
        require(change['recovered'][i] == sum(v['recovered'] for v in recovery.values()) and change['lost'][i] == sum(v['lost'] for v in recovery.values()), 'Recovery bars differ')
        require(change['additional_FP'][i] == test['operating']['final_'+other]['overall']['fp']-test['operating']['final_'+base]['overall']['fp'], 'Added FP bars differ')


def verify(visual=False):
    manifest = read(OUT/'source_manifest.json'); data = load()
    require(len(manifest['figures']) == 12 and len(manifest['tables']) == 4, 'Expected figure/table set is incomplete')
    verify_values(manifest, data)
    for name, expected in manifest['sources_sha256'].items():
        require(sha(ROOT/name) == expected, 'Input artifact changed')
    # Normalized newlines keep the source identity stable across Git checkouts.
    for name, expected in manifest['plotting_source_sha256'].items():
        actual = hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode()).hexdigest()
        require(actual == expected, 'Figure code changed since rendering: '+name)
    png_count = svg_count = 0
    for entry in manifest['figures']:
        require(not entry['layout_text_outside_canvas'], 'Text extends outside the canvas: '+entry['name'])
        for name in entry['files']:
            path = OUT/name; require(path.exists(), 'Missing figure export')
            if path.suffix == '.png':
                with Image.open(path) as image:
                    require(all(abs(d-300) < 1 for d in image.info['dpi']), 'PNG is not 300 DPI')
                    require(abs(image.width-6.6*300) <= 1, 'PNG physical width differs')
                    require(image.convert('RGB').getpixel((0, 0)) == (255, 255, 255), 'PNG background differs')
                png_count += 1
            else:
                tree = ET.parse(path)
                require(tree.findall('.//{http://www.w3.org/2000/svg}text'), 'SVG contains no editable text')
                require(not tree.findall('.//{http://www.w3.org/2000/svg}image'), 'Quantitative SVG unexpectedly embeds a raster')
                svg_count += 1
    for entry in manifest['tables']:
        with (OUT/entry['files'][0]).open(encoding='utf-8-sig', newline='') as stream:
            actual = list(csv.reader(stream))
        require(actual == [entry['headers'], *[[str(v) for v in row] for row in entry['rows']]], 'Table CSV differs from recorded cells')
    # Check numerical result tables against the evidence independently of the writer.
    test = data['results']['test']; tables = {v['name']: v for v in manifest['tables']}
    for row, method in zip(tables['table_2_final_test_performance']['rows'], METHODS):
        ap = test['AP'][method]['overall']; op = test['operating'][method]['overall']
        require(row[:4] == [method[-1], *[f'{ap[k]*100:.4f}' for k in ('map50_95', 'map50', 'map75')]], 'Overall AP table differs')
        require(row[4:7] == [op[k] for k in ('tp', 'fp', 'fn')], 'Operating counts table differs')
        require(row[7:] == [f'{op[k]*100:.2f}' for k in ('precision', 'recall')], 'Operating rate table differs')
    for row, cls in zip(tables['table_3_class_AP_support']['rows'], CLASSES):
        require(row[1:3] == [test['support'][cls][k] for k in ('objects', 'patients')], 'Support table differs')
        require(row[3:] == [f"{test['AP'][m][cls]['ap50_95']*100:.2f}" for m in METHODS], 'Class AP table differs')
    for row, (name, contrast) in zip(tables['table_4_paired_comparisons']['rows'], test['AP_contrasts'].items()):
        ci = test['bootstrap']['contrasts'][name]['overall']['map50_95']
        require(row[1:] == [f"{contrast['overall']['map50_95']*100:+.4f}",
                f"[{ci['lower']*100:+.4f}, {ci['upper']*100:+.4f}]", ci['replicates_present']], 'Paired table differs')
    before = read(WORK/'historical_before.json')
    require(all(sha(ROOT/name) == expected for name, expected in before.items()), 'Historical file changed')
    text = '\n'.join(p.read_text(encoding='utf-8') for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.md', '.json', '.csv', '.svg'))
    private = read(WORK/'qualitative_sources.json')
    require(all(v['image_stem'] not in text for v in private), 'Qualitative image identifier entered public files')
    require(not any(v in text for v in ('"patient_id"', '"image_stem"', '"display_crop"', '"boxes"')), 'Private fields entered public output')
    for path in OUT.rglob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            resolved = path.parent/target
            # The receipt linked by the index is written only after these checks pass.
            require(resolved == OUT/'verification.json' or resolved.exists(), 'Broken figure/table index link: '+target)
    from .data import SOURCES
    receipt = {'passed': True, 'quantitative_figures': 12, 'public_PNG_300dpi': png_count, 'public_vector_SVG': svg_count,
        'tables': 4, 'table_formats': ['CSV', 'Markdown'], 'private_qualitative_figures': len(private),
        'figure_values_match_verified_sources': True, 'table_cells_match_manifest': True, 'result_tables_match_verified_sources': True,
        'plotting_code_hashes_checked': len(manifest['plotting_source_sha256']),
        'quantitative_SVG_contains_no_raster_images': True, 'no_text_outside_canvas': True,
        'historical_tracked_files_unchanged': len(before), 'source_hashes_checked': len(manifest['sources_sha256']),
        'no_private_radiograph_or_patient_linkage_published': True, 'no_training_or_inference_or_metric_rule_change': True,
        'visual_review': {'passed': visual, 'public_figures_reviewed': 12 if visual else 0, 'private_examples_reviewed': 2 if visual else 0,
            'scope': 'Normal thesis-width previews, font readability, clipping, scales and local redaction; qualitative publication remains uncleared'},
        'artifacts_sha256': {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'verification.json'}}
    write_json(OUT/'verification.json', receipt)
    print('Figure values, 300 DPI/vector exports, privacy and historical preservation pass; visual review:', visual, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--visual-reviewed', action='store_true')
    verify(parser.parse_args().visual_reviewed)
