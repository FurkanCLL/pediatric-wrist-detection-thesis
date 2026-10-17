# Pediatric Wrist Detection Thesis

Implementation workspace for a new RTU Type 2 bachelor thesis study using the GRAZPEDWRI-DX pediatric wrist X-ray dataset, YOLOv8s, class-aware training, and adaptive local refinement.

**Status:** Phase 4B rarity-only training integration and readiness verification are complete: **READY WITH DOCUMENTED LIMITATIONS**. All 240 plans match Phase 4A exactly; 187 tests and an isolated two-epoch real-data diagnostic pass, with 28,122 consumed draws, 1,725 successful updates and zero AMP skips. Actual epoch-specific batches, augmentations and checkpoint reloads are verified. Read the [readiness report](docs/evidence/rarity_training_readiness/RARITY_TRAINING_READINESS_REPORT.md) before the student's manual 80-epoch launch. The official rarity-only run has **not** started. The approved matrix remains Standard/Rarity-only x Standard/SAHI Context/CG-ALR V2. Baseline, frozen common protocol/data and previous negative inference evidence are preserved; D/E/F/test evaluation is not performed. Formal supervisor scope alignment remains pending.

## Start here

1. Read `AGENTS.md`, `docs/THESIS.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`.
2. Install the pinned dependencies in the project Python environment.
3. Read [the validation error-analysis and SAHI findings](docs/evidence/error_analysis_sahi/ERROR_ANALYSIS_SAHI_REPORT.md), [the class-aware design and recommendation](docs/evidence/class_aware_design/CLASS_AWARE_DESIGN_REPORT.md), [baseline preservation](docs/evidence/class_aware_design/BASELINE_PRESERVATION_REPORT.md), [the frozen training protocol and original manual commands](docs/evidence/training_protocol/TRAINING_PROTOCOL_REPORT.md), [the GPU feasibility report](docs/evidence/gpu_feasibility/GPU_FEASIBILITY_REPORT.md), [dataset preparation](docs/evidence/dataset_preparation/DATASET_PREPARATION_REPORT.md), and the earlier [source audit](docs/evidence/dataset_audit/DATASET_AUDIT_REPORT.md).
4. Review [the adaptive validation findings](docs/evidence/adaptive_refinement_validation/ADAPTIVE_REFINEMENT_VALIDATION_REPORT.md) alongside [the historical design](docs/evidence/adaptive_refinement_design/ADAPTIVE_REFINEMENT_DESIGN_REPORT.md). `configs/adaptive_refinement_execution.json` records student approval for this completed validation study; the original proposed config retains its historical flags/hash. Further optimization, class-aware training and held-out evaluation require separate authorization.

Historical feasibility decision: review [the Phase 4A report](docs/evidence/error_guided_training_feasibility/ERROR_GUIDED_TRAINING_FEASIBILITY_REPORT.md), [exact proposed sampler](docs/evidence/error_guided_training_feasibility/SAMPLER_PROPOSAL.md) and [future loader feasibility](docs/evidence/error_guided_training_feasibility/LOADER_FEASIBILITY.md). That Phase 4A configuration remains a historical simulation-only record. Phase 4B separately approves its rarity-only comparator and integration; the [readiness report](docs/evidence/rarity_training_readiness/RARITY_TRAINING_READINESS_REPORT.md) supplies the current manual-launch gate.

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

`validate` independently decodes every original and prepared PNG with OpenCV, checks all mapped pixels, verifies original hashes, reconstructs boxes from source points, and checks complete split membership without patient overlap. `loader` scans every split through the installed Ultralytics detection dataset and checks sample three-channel batches with augmentation disabled. Its 320-pixel tensor check is not the selected training resolution and runs no model. Phase 1B used CPU-only PyTorch; the later Phase 2A report records the verified CUDA build.

Patient/image manifests, source-object traceability, pixel review sheets, settings and caches stay under ignored `outputs/` or `data/`. Only aggregate evidence is exported to `docs/evidence/dataset_preparation/`. Training-only support and geometry statistics are provided for the later class-aware strategy; no weights, crop parameters or augmentation settings have been selected. The held-out test split is frozen and must not guide method development.

## GPU diagnostics

`configs/training_candidate.json` preserves the earlier 30-epoch diagnostic proposal. The approved official 80-epoch configuration is separate at `configs/training_official.json`. The diagnostic command enforces one epoch for the smoke test. It cannot launch the official standard or class-aware experiment.

The completed run used 768, batch 6, AMP with initial loss scale 128 and four-step accumulation. Training took 295.93 seconds; full trainer-integrated validation took 38.22 seconds. Read the [report and aggregate evidence](docs/evidence/gpu_feasibility/GPU_FEASIBILITY_REPORT.md) before launching another diagnostic. Reuse existing successful receipts and checkpoints when continuing this phase.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m wrist_train.environment weights
.\.venv\Scripts\python.exe -m wrist_train.environment verify
.\.venv\Scripts\wrist-feasibility.exe benchmark --batch 6 --workers 2 --attempt benchmark_b6_amp_01
.\.venv\Scripts\wrist-feasibility.exe smoke --batch 6 --workers 2 --attempt smoke_b6_amp_01
```

Use new attempt names for reruns; existing attempts are never replaced. CUDA is mandatory. Start at batch 6, then use 5 and 4 if memory errors justify them. Benchmark inputs come from a fixed hash-ordered training subset. A separate local YAML contains only train/validation paths. The test split is not evaluated. Full checkpoints, logs, source identities and detailed diagnostic metrics stay under ignored `outputs/gpu_feasibility/`.

For the existing completed attempts, aggregate export is read-only with respect to GPU work:

```powershell
.\.venv\Scripts\python.exe -m wrist_train.evidence --benchmark benchmark_b6_amp_02 --smoke smoke_b6_amp_01
```

The benchmark includes augmented loading, loss, backward, AdamW and EMA updates. It measures five warmup steps followed by twenty timed steps. Every benchmark step updates the optimizer to include its state and cost; the smoke test uses the configured accumulation. Numerical checks cover training loss, gradients, parameters, raw validation predictions and trainer-integrated validation loss. Initial AMP scale calibration and any skipped updates are recorded explicitly. The framework's automatic OOM batch reduction is blocked so a failed configuration remains distinguishable.

The first benchmark stopped at the diagnostic overflow limit with the native 65,536 initial loss scale. A same-batch FP32/AMP comparison identified finite gradients at scale 128, and the corrected benchmark and complete smoke passed without overflow. `runtime.amp_initial_scale` is applied by the project trainer, not a native `YOLO.train()` argument. The failed attempt and comparison remain local; their aggregate receipts are public.

Mild intensity augmentation uses HSV value only, tested on equal grayscale channels; hue/saturation and optional Albumentations transforms are disabled. In this pinned framework a one-epoch run disables scheduler warmup, even when a longer candidate run requests it. The smoke receipt reports this actual behavior. Passing one epoch does not establish convergence or guarantee numerical stability throughout a longer run.

## Standard baseline: completed; original launch interface

The standard baseline is complete. Read [its preservation report](docs/evidence/class_aware_design/BASELINE_PRESERVATION_REPORT.md) for checkpoint identity and actual validation results. The [protocol report](docs/evidence/training_protocol/TRAINING_PROTOCOL_REPORT.md) preserves the original readiness record and launch/recovery instructions below. A fresh start refuses the existing output directory; do not rename, delete or retrain the preserved baseline. This interface does not launch a class-aware run.

From this repository root in PowerShell, first run the non-training check:

```powershell
.\.venv\Scripts\wrist-train.exe preflight
```

After reviewing its result, the following explicit command starts the official standard baseline:

```powershell
.\.venv\Scripts\wrist-train.exe start --confirm-standard-baseline
```

Outputs stay under ignored `outputs/training/standard_baseline/`. Existing runs cannot be overwritten. In another terminal, use `wrist-train.exe status` or `monitor` with the same virtual-environment path. After the earlier training process exits, inspect recovery with preflight before deliberately resuming:

```powershell
.\.venv\Scripts\wrist-train.exe preflight --resume --checkpoint last
.\.venv\Scripts\wrist-train.exe resume --confirm-resume-standard-baseline --checkpoint last
```

Resume restores the completed-epoch model/optimizer/EMA/schedule/scaler state into a new segment. Worker randomness and partial accumulation are not bitwise restored. The report explains verified predecessor/periodic fallback when the latest file is incomplete. Keep all segments and disclose actual successful/skipped updates in later comparisons. Class-aware production integration, final inference-protocol selection and adaptive refinement remain future work; the validation-only SAHI feasibility study is complete.

## Class-aware design: non-training analysis only

The [design report](docs/evidence/class_aware_design/CLASS_AWARE_DESIGN_REPORT.md) compares three proposed hybrid policies. Each retains 14,061 draws and all 4,264 training patients per epoch. Candidate A requests 2% extra rare-class views, caps image/patient repetition, protects fracture-positive and negative exposure, and leaves bone lesion coverage-only. Conservative geometric crops apply only to additional draws; every selected parent retains a full-image view. Exposure gains are not model performance results.

Source is isolated in `src/wrist_design/`; `configs/class_aware_candidates.json` remains proposed. Read the report before rerunning analysis. These commands need existing private frozen data/baseline outputs and never train a model:

```powershell
.\.venv\Scripts\python.exe -m wrist_design.baseline
.\.venv\Scripts\python.exe -m wrist_design.analyze
.\.venv\Scripts\python.exe -m wrist_design.simulate
.\.venv\Scripts\python.exe -m wrist_design.geometry
.\.venv\Scripts\python.exe -m wrist_design.verify
.\.venv\Scripts\python.exe -m wrist_design.prototype --attempt visual_review_repeat
```

Detailed draws, patient mappings and image comparisons stay in ignored `outputs/class_aware_design/`. Use a new prototype attempt name to retain previous figures; a new run needs a fresh technical visual review. Only aggregate evidence is tracked. The baseline lock and prepared dataset remain unchanged. Policy approval, production loader integration and any later class-aware training require a separate reviewed step.

## Validation error analysis and SAHI feasibility

Phase 3B uses the preserved epoch-22 EMA checkpoint and the frozen validation split only. Read the [error-analysis and SAHI report](docs/evidence/error_analysis_sahi/ERROR_ANALYSIS_SAHI_REPORT.md), [predefined comparison](configs/inference_feasibility.json), [native validation reproduction](docs/evidence/error_analysis_sahi/reference_reproduction.json) and [original-coordinate evaluation bridge](docs/evidence/error_analysis_sahi/evaluation_bridge.json). These are development findings from a validation-selected model, not final held-out A–F results.

Install the additive inference lock after the existing CUDA environment; do not upgrade its pinned training packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

The completed-attempt commands below document the sequence. Inference output directories refuse reuse; do not repeat these GPU passes over successful existing evidence. Reproduction needs a deliberately separate preserved output namespace or fresh local state with the same frozen inputs. The original baseline directory is never an inference destination.

```powershell
.\.venv\Scripts\python.exe -m wrist_eval.reference
.\.venv\Scripts\python.exe -m wrist_eval.feasibility pilot standard_serial
.\.venv\Scripts\python.exe -m wrist_eval.feasibility pilot sahi_context
.\.venv\Scripts\python.exe -m wrist_eval.feasibility pilot sahi_detail
.\.venv\Scripts\python.exe -m wrist_eval.feasibility gate
.\.venv\Scripts\python.exe -m wrist_eval.feasibility full standard_serial
.\.venv\Scripts\python.exe -m wrist_eval.feasibility full sahi_context
.\.venv\Scripts\python.exe -m wrist_eval.feasibility full sahi_detail
.\.venv\Scripts\python.exe -m wrist_eval.bridge
.\.venv\Scripts\python.exe -m wrist_eval.analyze
.\.venv\Scripts\python.exe -m wrist_eval.bootstrap
.\.venv\Scripts\python.exe -m wrist_eval.qualitative
.\.venv\Scripts\python.exe -m wrist_eval.verify
```

`analyze`, `bootstrap` and `bridge` read saved predictions without new neural inference. `verify` includes only a nine-image deterministic prediction replay plus integrity checks; it is not a repeated full benchmark. `qualitative` writes protected, private overlays and requires actual visual inspection before its public receipt is marked reviewed. All medical images, patient/image identities, detailed predictions and raw timing logs remain under ignored `outputs/error_analysis_sahi/`. Only aggregate evidence is tracked. No command in this section trains a model, evaluates test images or implements the final adaptive system. Class-aware candidates A/B/C remain unapproved and unchanged.

## Adaptive refinement: proposed design only

Read [the CG-ALR technical specification](docs/evidence/adaptive_refinement_design/ADAPTIVE_REFINEMENT_DESIGN_REPORT.md), [proposed parameters](configs/adaptive_refinement_proposed.json) and [design checks](docs/evidence/adaptive_refinement_design/design_verification.json). The proposal selects weak fracture/periosteal predictions, plans at most two broad local views and permits anchored confirmation or guarded coordinate blending. V1/V2 are development alternatives; one common adaptive rule must eventually serve both checkpoints in the existing A-F matrix.

The proposed configuration and design report retain their original Phase 3C.1 unapproved status as historical records. The student subsequently approved Phase 3C.2 through the separate execution manifest. The implementation consumes the preserved specification only after checking that approval and its hash. The completed validation findings below supersede the old implementation-status narrative; final inference selection remains open.


## Adaptive refinement: completed validation

The isolated CG-ALR core retains global rows and uses the same loaded checkpoint for zero to two local views. V1 confirms weak target scores; V2 adds guarded coordinate blending. Global outputs match preserved standard predictions exactly on all 3,088 validation images. Both use 1.416 model calls/image on average. The full comparison, 500 patient-bootstrap resamples, actual standalone timing, private technical figure reviews and preservation receipts are under `docs/evidence/adaptive_refinement_validation/`. The final suite passes 152 tests.

Private prediction bundles, radiographs, overlays and timing records remain in ignored `outputs/adaptive_refinement/`. Preserve completed attempts rather than repeating inference. Offline verification of saved evidence:

```powershell
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_validation accounting
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis fp_uncertainty
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis geometry
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_analysis ties
.\.venv\Scripts\python.exe -m wrist_eval.adaptive_validation decisions
```

The report records original pilot/full/timing commands, immutable source identities, all eight class metrics, operating points and uncertainty. No algorithm adjustment or next training/test phase is authorized by this completed study. The small V2-over-V1 numerical gain cannot override their negative standard comparison and excess false positives.


## Training-only error-guided feasibility: completed

`src/wrist_mining/` isolates the Phase 4A training-only inference, difficulty analysis, private technical audit and matched full-image sampler prototype. Potential unmatched predictions never become verified negative labels or sampling rewards. All existing model outputs are retained under ignored `outputs/error_guided_training_feasibility/`; no successful GPU pass needs repeating.

The standard, rarity-only and PAEG-CAT simulations use the same 14,061 epoch draw budget and original labels, with all 4,264 patients, >=95% distinct images in each fracture stratum, image/patient caps and every view of the 16 lesion patients exactly once. Both candidates realize 625 extra draws per epoch. The [report](docs/evidence/error_guided_training_feasibility/ERROR_GUIDED_TRAINING_FEASIBILITY_REPORT.md) explains the 702 requested versus 625 realized replacements, teacher/ranking limitations, all class/patient exposure distributions and the SIMPLIFY recommendation.

Saved-evidence verification without new model inference:

```powershell
.\.venv\Scripts\python.exe -m wrist_mining.verify
.\.venv\Scripts\python.exe -m wrist_mining.preservation after
```

Phase 4A ended with the isolated prototype, 171 tests and no training integration. Phase 4B subsequently approves and verifies a separate rarity-only finite-loader subclass, preserving the standard trainer and the original common protocol. The historical Phase 4A results and proposals remain unchanged.


## Rarity-only training: manual launch ready

The [Phase 4B readiness report](docs/evidence/rarity_training_readiness/RARITY_TRAINING_READINESS_REPORT.md) records exact source/policy identities, the finite lazy loader, actual batch/label/augmentation checks, two-epoch GPU health, checkpoint/recovery verification and remaining reproducibility limits. Diagnostic outputs stay in `outputs/rarity_training_readiness/diagnostics/two_epoch_01`. The future official output is `outputs/training/rarity_only`; existing runs cannot be overwritten.

From the repository root, verify readiness without training:

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch preflight
```

The student can then start the official **80 epochs** manually:

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch start --confirm-rarity-only-80-epochs
```

Use `python -m wrist_rarity.launch status` or `monitor` from the same project environment for progress. The report gives confirmed resume commands and post-training checks. Prefer an uninterrupted comparison: finite workers have a documented epoch seed schedule, while recovery still discards pending gradients and resets native update cadence. Source plans replay, but resumed optimization is not bitwise identical.

Reserve approximately **10–12 hours** and preferably **6 GiB free disk** for the future model. Current preflight reports 4.944 GiB free VRAM; diagnostic peak allocated/reserved VRAM is 1.905/2.395 GiB. No official fitting, D/E/F inference or held-out test evaluation is triggered by completing readiness.
