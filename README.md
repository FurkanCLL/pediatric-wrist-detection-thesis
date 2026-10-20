# Pediatric Wrist Detection Thesis

This repository contains the code and results for my RTU bachelor thesis on object detection in pediatric wrist X-rays. The dataset is GRAZPEDWRI-DX. The project compares standard YOLOv8s training with patient-aware rarity sampling, then tests three inference methods on each model.

## Current status

Both models have finished training, and the six validation experiments are complete:

| Training | Standard inference | SAHI Context | CG-ALR V2 |
|---|---|---|---|
| Standard | A | B | C |
| Rarity-only | D | E | F |

SAHI Context reduced overall and fracture AP in these runs. CG-ALR V2 recovered some targets but added false positives and did not meet the predefined improvement criteria. The [comparison report](docs/evidence/rarity_inference_comparison/RARITY_INFERENCE_COMPARISON_REPORT.md) contains the full results and limitations.

The held-out test set has not been evaluated. Final method selection and thesis scope approval are still pending.

## Repository layout

- `src/` — dataset preparation, training, inference, evaluation and figure scripts.
- `configs/` — dataset and experiment settings, including the frozen protocols.
- `docs/evidence/` — technical reports, aggregate metrics and verification records.
- `docs/figures/` — thesis figures, result charts and source indexes.
- `tests/` — checks for data handling and the main pipeline logic.
- `data/` and `outputs/` — local dataset files, checkpoints, predictions and logs; these are not committed.

## Reports and figures

- [Dataset audit](docs/evidence/dataset_audit/DATASET_AUDIT_REPORT.md) and [preparation](docs/evidence/dataset_preparation/DATASET_PREPARATION_REPORT.md).
- [Common training protocol](docs/evidence/training_protocol/TRAINING_PROTOCOL_REPORT.md) and [rarity-only training verification](docs/evidence/rarity_training_completion/RARITY_TRAINING_COMPLETION_REPORT.md).
- [Baseline SAHI analysis](docs/evidence/error_analysis_sahi/ERROR_ANALYSIS_SAHI_REPORT.md) and [adaptive refinement validation](docs/evidence/adaptive_refinement_validation/ADAPTIVE_REFINEMENT_VALIDATION_REPORT.md).
- [Completed A–F comparison](docs/evidence/rarity_inference_comparison/RARITY_INFERENCE_COMPARISON_REPORT.md) and [results table](docs/evidence/rarity_inference_comparison/six_condition_matrix.csv).
- [Figure index](docs/figures/README.md), including 20 polygon-to-box examples and three saved-prediction comparisons.

The figures use existing project data. Their manifests record the source artifacts and image hashes. Text regions are redacted in exported image examples; original files are unchanged.

## Local setup

The project uses Python 3.12 and pinned dependencies. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

With the local dataset and saved results available, regenerate the presentation figures with:

```powershell
.\.venv\Scripts\python.exe -m wrist_figures
.\.venv\Scripts\python.exe -m wrist_figures.verify
```

This only reads existing artifacts and renders figures. Detailed training and evaluation commands remain in the evidence reports.
