# Pediatric Wrist Detection Thesis

Implementation workspace for a new RTU Type 2 bachelor thesis study using the GRAZPEDWRI-DX pediatric wrist X-ray dataset, YOLOv8s, class-aware training, and adaptive local refinement.

**Status:** Phase 1B dataset preparation. Seed-42 patient splits and the shared eight-class grayscale dataset are prepared. See the preparation report for complete validation evidence. No model training or experiment results have been produced. The 2 x 3 design is student-approved and subject to supervisor alignment for the formal thesis scope.

## Start here

1. Read `AGENTS.md`, `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`.
2. Install the pinned dependencies in the project Python environment.
3. Read [the dataset preparation report](docs/evidence/dataset_preparation/DATASET_PREPARATION_REPORT.md) and the earlier [source audit](docs/evidence/dataset_audit/DATASET_AUDIT_REPORT.md).
4. Follow `docs/ROADMAP.md`. The next phase is bounded GPU feasibility at 768/1024, before official training.

The previous repository and experiments are archived separately. Do not copy its historical A-F results into this project's experimental outcomes.

## Reproduce the audit

Run from the repository root in PowerShell with Python 3.12 or newer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\wrist-audit.exe
```

The command reads `data/raw/grazpedwri-dx/` and writes to ignored `outputs/dataset_audit/`. It checks every released image header and PNG integrity, hashes every source file, cross-checks all three annotation formats, and decodes a deterministic image subset. `--decode-sample` changes the base sample size; up to eight images per category and eight empty-box images are also included. `--workers` controls image-read concurrency. The recorded sample count is the actual coverage, not the base sample size.

Detailed image, patient, object, and source-path records stay local. Never add `data/` or `outputs/` to Git. The audit does not create splits, change pixels, or generate model inputs. Exit code 2 preserves findings while signaling blocking validation issues; consult `summary.json` and `issues.csv` before preparation.

Verify the pinned official release and export only aggregate evidence:

```powershell
.\.venv\Scripts\python.exe -m wrist_audit.provenance --fetch
.\.venv\Scripts\python.exe -m wrist_audit.exports
.\.venv\Scripts\python.exe -m wrist_audit.evidence
```

The provenance command fetches v2 metadata and the approximately 26 MB annotation archive into ignored outputs. It checks the official MD5 values and compares archive members with the local source without extraction. Omit `--fetch` to reuse these downloaded files offline. It does not download the four large image ZIPs.

The optional `exports` command rechecks every annotation export from that verified archive without repeating image hashing/decoding. It requires a successful local-equivalence receipt. The evidence command combines these explicitly recorded validation batches and exports compact aggregate JSON/CSV files to `docs/evidence/dataset_audit/`; inspect them before committing. The original full-scan receipt remains local, including any resolved tooling errors. In the first audit, empty VOC `<size/>` elements exposed a parser assumption; this was corrected and all export comparisons were rerun. The current full-audit parser supports those empty elements directly.

The planning files and `AGENTS.md` are intentionally local and Git-ignored. Shareable audit evidence is tracked under `docs/evidence/`.

## Reproduce preparation

The approved policy is in `configs/dataset_preparation.json`: patient proportions 70/15/15 with seed 42, Supervisely labels, the fixed eight-class mapping, and nearest-integer full-range linear conversion from unsigned 16-bit to 8-bit grayscale. Keep all images, omit only `foreignbody` boxes, axes and image tags, and preserve all target geometry. These omissions treat foreign-body objects as unlabeled background rather than ignored regions.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\wrist-prepare.exe freeze
.\.venv\Scripts\wrist-prepare.exe pilot
```

The source audit must already exist locally. `freeze` verifies metadata and Supervisely hashes against it, allocates whole patients once, checks image balance, and freezes a local manifest. Repeating it loads and verifies the existing frozen files. It never searches seeds. `pilot` uses 16 training-only images, independently checks pixels and source geometry, and writes private comparison sheets under `outputs/dataset_preparation/visual_review/`.

After actually reviewing those sheets and finding the approved mapping acceptable, record the observations and continue. For an existing passed review of the same frozen policy, resume at `prepare` rather than replacing its review receipt:

```powershell
.\.venv\Scripts\wrist-prepare.exe record-pilot-review --note "Your actual technical observations"
.\.venv\Scripts\wrist-prepare.exe prepare
.\.venv\Scripts\wrist-prepare.exe validate
.\.venv\Scripts\wrist-prepare.exe loader
.\.venv\Scripts\python.exe -m wrist_prepare.evidence
```

Prepared data live in ignored `data/prepared/grazpedwri_detection/`, with conventional `images/{train,val,test}`, `labels/{train,val,test}` and `dataset.yaml`. The local YAML resolves this dataset root explicitly and requests three loader channels. Source images and annotations are never overwritten. Each original produces one lossless 8-bit PNG at its original dimensions and one label file, including empty files when appropriate. There is no resizing, augmentation, crop generation or enhancement in the prepared files.

`prepare` uses protected atomic writes and a local SQLite completion journal. An interrupted run can repeat the same command: completed outputs are hash-checked and reused; differing existing files cause an error. At most the small uncommitted journal block needs recomputation. Do not change or delete frozen manifests to obtain a different allocation. To investigate a changed policy, preserve this dataset/state and use separate explicit output paths after methodological review.

`validate` independently decodes every original and prepared PNG with OpenCV, checks all mapped pixels, verifies original hashes, reconstructs boxes from source points, and checks complete split membership without patient overlap. `loader` scans every split through the installed Ultralytics detection dataset and checks sample three-channel batches with augmentation disabled. Its 320-pixel tensor check is not the selected training resolution and runs no model. The installed PyTorch build is CPU-only; a compatible CUDA build must be selected before the next GPU feasibility stage.

Patient/image manifests, source-object traceability, pixel review sheets, settings and caches stay under ignored `outputs/` or `data/`. Only aggregate evidence is exported to `docs/evidence/dataset_preparation/`. Training-only support and geometry statistics are provided for the later class-aware strategy; no weights, crop parameters or augmentation settings have been selected. The held-out test split is frozen and must not guide method development.
