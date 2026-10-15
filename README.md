# Pediatric Wrist Detection Thesis

Implementation workspace for a new RTU Type 2 bachelor thesis study using the GRAZPEDWRI-DX pediatric wrist X-ray dataset, YOLOv8s, class-aware training, and adaptive local refinement.

**Status:** Phase 1 source audit. The original dataset is available locally and a reproducible inspection workflow is implemented. Splits, image preparation, training, and experiment results remain pending. The 2 x 3 design is student-approved and subject to supervisor alignment for the formal thesis scope.

## Start here

1. Read `AGENTS.md`, `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`.
2. Install the pinned audit dependencies in the project Python environment.
3. Read [the dataset audit report](docs/evidence/dataset_audit/DATASET_AUDIT_REPORT.md) before deciding split, class-exclusion, or image-preparation policies.
4. Follow `docs/ROADMAP.md`. The current task stops before creating splits or detector inputs.

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
