"""Build a separate Chapter 5 package without changing experiments or old figures."""

import csv
import hashlib
from pathlib import Path
import subprocess
import platform

from .data import ROOT, OUT, WORK, SOURCES, require, read, write_json, sha, snapshot, install_guard, load


def index(figures, tables, qualitative):
    text = ['# Chapter 5 figures and tables', '',
        'Final A–C use the standard checkpoint; D–F use the rarity-only checkpoint. Inference is Standard/SAHI Context/CG-ALR V3 within each group. C is the predefined primary pipeline. Historical V2 figures remain unchanged.', '',
        'Quantitative exports are 300 DPI PNG plus vector SVG at 6.6 inches (167.6 mm) width. Text is designed for a normal A4 thesis page. AP figures use percentages, contrast figures use percentage points, losses retain native units, and latency uses milliseconds. Use the following text as technical caption guidance; the student writes the final thesis text.', '',
        'Native training-time validation curves are separate from the common final original-coordinate evaluator. Training losses reflect different sampled mixtures and are descriptive. No smoothing or missing measurements are added.', '']
    for entry in figures:
        text.extend([f"## {entry['section']} — {Path(entry['name']).name.replace('_', ' ')}", '',
            ' · '.join(f'[{Path(p).suffix[1:].upper()}]({p})' for p in entry['files']), '', entry['caption'], ''])
    text.extend(['## Scientific tables', '', 'CSV files use UTF-8 with BOM for Word/Excel import. Markdown has simple headers and no merged cells. AP point estimates are percentages; differences and intervals use percentage points. Rounding is for display only.', ''])
    for entry in tables:
        text.extend([f"- Section {entry['section']}: [Markdown]({entry['files'][1]}) · [CSV]({entry['files'][0]}) — {entry['caption']}"])
    text.extend(['', '## Local qualitative examples', '',
        'Section 5.4 has two local-only PNG/SVG comparisons: a genuine saved V3 fracture recovery and an annotation-relative SAHI fracture FP. They are under `outputs/chapter5_figures/build_01/qualitative/`, with a private index and patient-linked provenance outside Git. Each uses Ground truth/A/B/C, identical display crops, fracture-only overlays, confidence .25 and matching IoU .50. Annotated text is masked and image metadata is not copied. Original prediction coordinates are preserved. Selection is the first hash-ranked eligible case whose target avoids text redaction; examples are explanatory, not representative.', '',
        'The cached official release records CC BY 4.0, but complete individual privacy clearance is not established. Therefore no radiographs or qualitative image identifiers are published in this package. The local index records attribution and transformations.', '',
        '## Sources, interpretation and reproduction', '',
        '[Source manifest](source_manifest.json) records every plotted value, table value, source hash and rendering identity. [Verification](verification.json) records data, file, layout and visual checks. Run `.venv/Scripts/python.exe -m wrist_final_figures.build` with existing local evidence. This reads saved data only. It does not load weights, train, infer, rerun bootstrap estimates, or alter evaluation rules.', '',
        'A separate global mAP50/mAP75 figure was omitted because Table 2 reports these metrics. The three-panel cost figure includes p95 latency; a redundant latency-distribution figure was omitted. No learning-curve metric was unavailable: both verified logs contain all 80 epochs and the requested loss/P/R/AP fields.', '',
        'The figures retain the mixed result: SAHI increases unmatched predictions and lowers AP; rarity-only raises overall test macro AP but not every class; V3 adds modest operating recoveries within budget, with overall AP intervals spanning zero. Bootstrap intervals are nominal and do not establish clinical validation. Sparse findings and one training seed per strategy limit interpretation.', ''])
    (OUT/'FIGURE_INDEX.md').write_text('\n'.join(text), encoding='utf-8', newline='\n')


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    install_guard()
    baseline = WORK/'historical_before.json'
    if not baseline.exists():
        write_json(baseline, snapshot())
    data = load()
    from .plots import figures
    import matplotlib
    import numpy as np
    import PIL
    from .tables import tables
    from .qualitative import build as qualitative_build
    # Use LF text exports so Git round trips preserve the recorded byte hashes.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'.gitattributes').write_text('* text eol=lf\n*.png -text\n', encoding='utf-8', newline='\n')
    entries = figures(data); table_entries = tables(data); qualitative = qualitative_build()
    index(entries, table_entries, qualitative)
    code = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted((ROOT/'src/wrist_final_figures').glob('*.py'))}
    manifest = {'schema_version': 1, 'phase': '5E', 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sources_sha256': SOURCES, 'plotting_source_sha256': code, 'figures': entries, 'tables': table_entries,
        'qualitative': qualitative, 'rendering': {'dpi': 300, 'font': 'DejaVu Sans', 'base_font_points': 8.5,
            'width_inches': 6.6, 'background': 'white', 'AP_scale': 'percent', 'AP_differences': 'percentage points',
            'versions': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                         'numpy': np.__version__, 'Pillow': PIL.__version__}},
        'frozen_checkpoints': data['matrix']['checkpoints'], 'no_training_inference_or_evaluation_rule_change': True}
    write_json(OUT/'source_manifest.json', manifest)
    print(f'Created {len(entries)} public quantitative figures, {len(table_entries)} tables and {qualitative["figures"]} private qualitative figures', flush=True)


if __name__ == '__main__':
    build()
