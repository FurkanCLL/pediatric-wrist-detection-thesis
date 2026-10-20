# Thesis and result figures

This folder contains 29 figures made from existing project artifacts: six charts, 20 annotation-conversion candidates and three qualitative comparisons. Every figure has a 300 dpi PNG. The six charts also have SVG copies for vector export.

## Thesis figures

- [Class distribution](thesis/class_distribution_chart.png): dataset-wide counts for the eight project classes, 47,436 annotations in total. Counts come from the existing preparation statistics and agree with the source audit. Foreign-body objects and axis lines are excluded.
- [Conversion candidates](manifests/annotation_conversion_candidates.csv): 20 separate training-image examples, named `thesis/annotation_conversion_candidates/example_01.png` through `example_20.png`.

The original export uses polygons for **bone lesion** and **periosteal reaction**. The other six classes already have rectangular annotations. The candidates therefore show genuine polygons from those two classes, covering different sizes and images. Six selections target bone lesion and fourteen target periosteal reaction, with one source image per patient. All source polygons in each selected image are drawn; other source rectangles are not drawn in these comparison panels.

Each candidate uses the same existing 8-bit image crop twice. The left panel draws the original polygon vertices; the right draws boxes decoded from the actual saved YOLO labels. Image contrast is unchanged. Crops are for presentation only.

## Result charts

| Figure | Files |
|---|---|
| Overall A–F mAP50–95 | [PNG](results/overall_performance_map50_95.png) · [SVG](results/overall_performance_map50_95.svg) |
| AP50 for all eight classes | [PNG](results/classwise_ap50_all_classes.png) · [SVG](results/classwise_ap50_all_classes.svg) |
| Fracture AP50 | [PNG](results/fracture_ap50_comparison.png) · [SVG](results/fracture_ap50_comparison.svg) |
| Recorded latency and model calls | [PNG](results/inference_cost_comparison.png) · [SVG](results/inference_cost_comparison.svg) |
| False positives at confidence .25 / IoU .50 | [PNG](results/false_positives_comparison.png) · [SVG](results/false_positives_comparison.svg) |

These are the completed validation results on 3,088 images from 914 patients. AP uses the existing 0–1 scale. The bars show point estimates; the [Phase 4E report](../evidence/rarity_inference_comparison/RARITY_INFERENCE_COMPARISON_REPORT.md) retains the paired uncertainty and equal-confidence AP-ordering limitation. False positives mean annotation-relative unmatched predictions.

The method-cost figure uses **D/E/F**, all from the rarity-only checkpoint. It shows recorded mean end-to-end latency and actual mean model calls. Measurements came from separate sessions and do not establish controlled speed differences.

## Qualitative comparisons

Each figure shows the same validation image under Standard, SAHI Context and CG-ALR V2, using saved D/E/F predictions:

1. [Additional SAHI detections](results/qualitative_example_01.png).
2. [A target recovered by adaptive refinement at the fixed operating point](results/qualitative_example_02.png).
3. [A shared correct-detection example](results/qualitative_example_03.png).

Dashed amber boxes are annotations; solid blue boxes are predictions. The display threshold is .25 in every panel. Annotated text regions are redacted, and text-class overlays are omitted consistently. Short labels identify the predicted class and score. The crop is identical across methods and may focus on an annotated target **after inference**, solely to make the illustration readable; it is not an inference ROI.

These are deliberately selected examples, not a representative sample or additional performance evidence. The [qualitative index](manifests/qualitative_examples.csv) records selection reasons and experiment IDs.

## Sources and reproduction

- [Full source manifest](manifests/source_manifest.json): exact artifact paths, hashes, plotted values, image hashes and rendering settings.
- [File index](manifests/figures.csv).
- [Verification receipt](manifests/verification.json).

Public indexes use the original image's SHA-256 as its identifier. Exact original/prepared image and annotation filenames remain in the local, ignored `outputs/thesis_figures/build_01/private_sources.json`. They can also be resolved through `source_image_sha256` in the frozen `source_records.jsonl`. Patient IDs and original filenames are not printed on the figures or published in their indexes.

With the local data and saved results present, run `python -m wrist_figures`, then `python -m wrist_figures.verify`. This reads existing artifacts and renders figures. It does not load a model, run training/inference, modify the experiments or open held-out image/label/prediction files.

Image source: Eszter Nagy, Michael Janisch, Franko Hrzic, Erich Sorantin and Sebastian Tschauner, [GRAZPEDWRI-DX v2](https://doi.org/10.6084/m9.figshare.14825193.v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution follows the project's cached official release metadata. These exports add crops, text redaction and annotation/prediction overlays to the project's prepared images. Original dataset files remain unchanged.
