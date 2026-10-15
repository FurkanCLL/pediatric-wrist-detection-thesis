# Pediatric Wrist Detection Thesis

Implementation workspace for a new RTU Type 2 bachelor thesis study using the GRAZPEDWRI-DX pediatric wrist X-ray dataset, YOLOv8s, class-aware training, and adaptive local refinement.

**Status:** Project bootstrap. Dataset not downloaded or audited; no models trained; no experiment results yet. The 2 x 3 design is student-approved and subject to supervisor alignment for the formal thesis scope.

## Start here

1. Read `AGENTS.md`, `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`.
2. Create a local Python environment. Use `requirements.txt` only once verified dependencies are pinned.
3. Confirm authoritative GRAZPEDWRI-DX archive, metadata, license and documentation before downloading into `data/raw/`.
4. Follow `docs/ROADMAP.md`, beginning with patient-level metadata and annotation audit. Do not train or evaluate on unverified image-based splits.

The previous repository and experiments are archived separately. Do not copy its historical A-F results into this project's experimental outcomes.
