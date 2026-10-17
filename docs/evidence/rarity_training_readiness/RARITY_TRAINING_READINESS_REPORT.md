# Rarity-only training is ready for a manual 80-epoch launch

Phase 4B engineering readiness report, 2026-10-17. **READY WITH DOCUMENTED LIMITATIONS.** The approved sampler, actual native training-loop consumption, two-worker transition, numerical health, checkpoint save/reload and original-protocol preservation pass. The official run has **not** started. The student launches it manually after reviewing this report.

## 1. Approved scope and implementation

The direct Phase 4B request approves patient-aware **rarity-only** sampling as the second YOLOv8s training condition. The two-checkpoint by three-inference matrix is Standard/Rarity-only x Standard/SAHI Context/CG-ALR V2. This approval selects existing components; it does not erase prior negative inference findings or claim supervisor/institutional scope acceptance. No D/E/F evaluation or held-out test access occurs here.

The authoritative scientific policy remains the completed Phase 4A report/proposal. No error guidance, potential-negative reward, crop, loss function or coefficient is added. The isolated two-epoch diagnostic uses actual prepared train/validation data and original COCO initialization, and is **not an official experiment or thesis performance result**.

| New file | Responsibility |
| --- | --- |
| `src/wrist_rarity/policy.py` | Return the unchanged Phase 4A rarity-only comparator and enforce its constraints |
| `src/wrist_rarity/data.py` | Indexed native dataset, deterministic epoch requests and lazy finite two-worker loader |
| `src/wrist_rarity/trainer.py` | Inherit frozen optimization/validation/recovery; audit actual batches and augmentations |
| `src/wrist_rarity/protocol.py` | Additive source/config lock and read-only input/environment preflight |
| `src/wrist_rarity/launch.py` | Separate diagnostic/manual start/resume/status/monitor actions and exclusive outputs |
| `src/wrist_rarity/verify.py` | Reconcile saved plans, actual batches, checkpoint states and preservation |
| `tests/test_rarity_training.py` | Focused real-worker/native-transform and launch/identity/plan failure fixtures |
| `configs/rarity_training.json` | Approved condition, original 80-epoch reference and separate two-epoch diagnostic |

The standard trainer, pinned framework, common training JSON/lock, old sampler/evaluator/preparation sources, dependencies and earlier experiment evidence are unchanged. Local THESIS/ROADMAP/DECISIONS record the direct approval and stopping gate; these guidance files remain intentionally Git-ignored. The implementation and pre-GPU gates were committed/pushed at `de8c7a4` before diagnostic fitting. See `INTEGRATION_PROTOCOL.md` for the pre-GPU design record.

| Identity | SHA-256 |
| --- | --- |
| Original common protocol | 281a36934cfef95599c92729bced3c0d2d1adbe589a35495f45f2c58fe259f01 |
| Additive rarity condition | 802c3f295182b346eb2a809e262cf7309b4d6dcea40de188c2bc2b8ddc58f2d5 |
| Rarity configuration | 921ca5e6f782da92fb6d7c64923df590f74b53e3053018f9c8f5e57111c5cea2 |
| Phase 4A sampler proposal | c7d09a95f1d3242cbf32127880d7c51bf0ff295974c4fb70304cd5f8ff2283f2 |
| Original verified COCO initialization | 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a |
| Frozen patient split | ee49e4ae1810ac1fb9541261e2d8df43eb32b5ca827433e1f9f22c20bb3c9c49 |

## 2. Exact sampler and actual constraints

`RarityPlans` takes the first result of the unchanged `MatchedPopulation.paired` implementation. Its required difficulty argument is a zero placeholder; no mined feature/prediction/teacher is opened. The second paired result is unused and cannot affect the rarity result. This preserves the original coverage skeleton, weighted rarity tickets, feasibility reservation, caps, fallback restoration and final permutation exactly.

The inherited rarity score for classes {periostealreaction, pronatorsign, softtissue, boneanomaly} is `r_c=min(2,sqrt(1590/I_c)*min(1,P_c/100))`, zero below 32 independent training patients, and `R_i=max_c(r_c*presence_ic)/2`. No new score is fitted. Repetition does not create independent patient information.

All **240 plans** (80 epochs x seeds 42, 1042, 2042) match the preserved Phase 4A rarity plans **element for element**. At runtime, seed 42 and zero-based epochs 0–79 select those same plans. Every real delivered draw in both diagnostic epochs also matches its plan, including the epoch-2 transition. No average substitutes for a per-epoch check.

| Constraint | Verified in each diagnostic epoch |
| --- | --- |
| Source-image draws | 14,061 |
| Realized replacements / unique images | 625 (4.4449%) / 13,436 |
| Fracture-positive / no-fracture-box draws | 9,307 / 4,754 |
| Distinct fracture-positive views | 8,842 / 9,307 =95.00376% |
| Distinct no-fracture-box views | 4,594 / 4,754 =96.63441% |
| Independent patients | 4,264 / 4,264 |
| Maximum image repetition | Two draws |
| Patient extra cap | min(3,ceil(.5*N_p)); zero for lesion patients |
| Lesion cohort | All 66 views of 16 patients exactly once |
| Physical batches | 2,343 x six + final three =2,344 |
| Targets/geometry | Original prepared labels and full-image geometry, then unchanged native augmentation |

The requested near-5% pool realizes 625 because the no-fracture-box rarity pool is cap-limited; the original 77 restored coverage views remain restored. No scientific rule was relaxed. No-fracture-box is an annotation stratum, not verified clinical normality. All source labels are checked before native augmentation; ordinary affine/box filtering can change post-augmentation targets exactly as in the baseline. No source annotation or pixel file is modified.

## 3. Finite loader, epoch boundaries and worker randomness

The real pinned Ultralytics loop is inherited, not copied or patched. Its `on_train_epoch_start` callback selects the checked plan before iteration. A finite loader starts two newly seeded workers per epoch, retains native `seed_worker`, prefetch factor 4, batch 6/drop-last-false and native YOLO transforms/collation. It rejects concurrent or unfinished iterators at an epoch switch and shuts workers down on completion/failure.

Source inspection found that the native progress loop constructs `enumerate(train_loader)` twice. A normal eager finite loader could start a discarded worker pool and advance RNG. The new generator delays `DataLoader.__iter__` until first consumption, so the unused iterator performs no prefetch or RNG draw. This behavior is exercised by synthetic worker tests and the actual native two-epoch loop.

Each request contains epoch, position, source index and repeat-audit flag. The wrapper checks native image/cache order and prepared annotation signatures, then uses native `dataset[index]`. The main process verifies the delivered path, original-label signature, worker ID/seed, epoch and exact position before CUDA/loss use. Private per-batch receipts and saved plans permit reconstruction of **actual consumed batches**, not merely comparison of intended lists. Both occurrences of all 625 repeated sources have transformed-tensor hashes without changing RNG consumption.

Epoch 0's generator uses the native single-GPU seed formula; later epochs use the explicit epoch increment. All 160 Torch worker seeds and their 32-bit NumPy/Python seeds are distinct across 80 epochs. A fresh launch replays the seed schedule. The generated-image tests consume native YOLO augmentations with two real workers over three epochs, then repeat the whole launch and obtain identical receipts/augmentation hashes, while repeated images receive different tensors.

Validation delegates directly to the original loader builder: native infinite iterator, rect padding, batch 12, four workers and no augmentation. It never receives training source-index requests. It remains the same patient-separated validation set, with no test entry in the run YAML. Preflight reads the shared frozen metadata identity and patient-split audit, but opens no test image or prepared-label file. No test record affects the sampling policy.

## 4. Automated and real GPU verification

| Gate | Result |
| --- | --- |
| Full existing plus focused suite | 187 pass; 16 new rarity tests; 31.692s |
| Actual synthetic worker consumption | Three epochs x two independent launches; both consecutive boundaries and native transforms/label pairing match |
| Failure cases | Stale epoch/order/label/worker seed, invalid plan, incomplete consumption, changed source lock and unconfirmed launch reject |
| Optimization/validation inheritance | Same native loop, optimizer step, scaler setup, recovery and validation construction |
| Plan reconciliation | All 240 ordered plans exactly match Phase 4A |
| Dependency check | No broken requirements; no new dependency |
| Raw/prepared data preservation | 121,973 stat entries unchanged; train/val bytes rechecked in preflight |
| Baseline preservation | All 36 artifacts retain their hashes |

The pre-GPU 187-test gate included two native worker epochs repeated across two launches. The final 187-test suite extends that fixture to three consecutive epochs/two launches, checking both epoch 0-to-1 and 1-to-2 transitions without changing training source or repeating GPU fitting. Initial focused testing had one fixture error: it referenced an incorrect optimizer-budget result key. Only the test was corrected before the passing full suite; the failed log is retained. No GPU fitting attempt preceded the automated gates. No successful diagnostic training is rerun.

### Two complete diagnostic epochs

| Epoch | Actual draws / batches | Attempts / successes / skips | AMP scale start → end | Different augmented pairs | Train seconds | Train peak allocated / reserved GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,061 / 2,344 | 1139 / 1139 / 0 | 128 → 128 | 625 / 625 | 469.76 | 1.905 / 1.992 |
| 2 | 14,061 / 2,344 | 586 / 586 / 0 | 128 → 128 | 625 / 625 | 401.17 | 1.905 / 2.395 |

| Validation pass | Images / batches | Seconds | Peak allocated / reserved GiB | Augmentation |
| --- | --- | --- | --- | --- |
| 1 | 3,088 / 258 | 41.28 | 0.728 / 2.395 | False |
| 2 | 3,088 / 258 | 27.77 | 0.728 / 2.395 | False |

| Epoch | train/box_loss | train/cls_loss | train/dfl_loss | val/box_loss | val/cls_loss | val/dfl_loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.51504 | 1.46129 | 1.36379 | 1.72860 | 1.10304 | 1.36040 |
| 2 | 1.34520 | 0.93723 | 1.26200 | 1.45778 | 0.91720 | 1.23601 |

The diagnostic finishes cleanly with 1725 attempts, 1725 successful AdamW/EMA updates and 0 protected AMP skips. The native two-epoch nominal budget is 1,725 attempts. Initial scale 128, warmup accumulation 1–4 and steady accumulation 4 use the unchanged control policy. Non-finite loss, live/EMA state, validation loss/raw prediction or persistent overflow would stop the run. Successful steps require finite gradients/norm/state; any protected overflow skip is counted rather than hidden. Total launch/setup/train/checkpoint wall time after preflight is 997.37s (16.62 min).

All original learning arguments in the actual native `args.yaml` match the completed baseline. The only differences are diagnostic `epochs=2` and isolated data/project/name/save_dir paths. This is a real two-epoch cosine schedule, **not** the first two epochs of an 80-epoch cosine trajectory. It verifies loading, sampling, optimization/AMP, validation and checkpoint infrastructure; it does not prove 80-epoch convergence, long-run numerical health or any AP benefit. Diagnostic validation/AP values are not promoted to thesis findings.

### Save, reload and recovery

| Checkpoint | Completed epoch | SHA-256 | Synthetic CUDA reload |
| --- | --- | --- | --- |
| last.pt | 2 | 39f3484cd030a895554d811f4c632cd996f714e743b0d65f58d8c8639969201e | True |
| best.pt | 2 | 39f3484cd030a895554d811f4c632cd996f714e743b0d65f58d8c8639969201e | True |

Hash-verified last/best reload with finite FP32 model/EMA/optimizer/scaler/scheduler state and matching condition/consumption identity. Both saved EMAs produce finite synthetic CUDA predictions after reload; this is no dataset evaluation. The actual completed-epoch 1 predecessor is also restored through inherited live-model/optimizer/EMA/scaler/scheduler/accounting code on CPU with exact tensor/state equality and no forward/backward/update. Its pending 3 microbatches are explicitly not recovered. Completed-budget continuation is rejected. The next epoch plan matches the actual epoch 2 plan.

## 5. Official protocol, resources and limitations

Official training inherits the original common JSON with 80 epochs. Only its data source-index policy and necessary loader/audit infrastructure differ. Architecture/eight classes, verified COCO initialization, 768, batch 6/nbs 24, AdamW, LR 0.001/weight decay 0.0005, cosine/lrf 0.01, one warmup epoch, AMP initial 128 and original growth/backoff/skip guards, augmentation, seed 42, validation and latest-tie best eight-class mAP50-95 stay fixed. Nominal budget is 1,124,880 draws and 47,433 optimizer attempts. Actual successful/skipped updates must be reported.

**Documented limitations:** finite workers restart with deterministic epoch seeds while the baseline workers persist. This changes worker lifecycle, cached buffers and RNG consumption; individual augmented views/minibatch pairings are not matched to the baseline. The augmentation policy/distribution is unchanged. Exact CUDA tensor/bitwise training equality is not promised. Resume replays source plans and worker seeds at the completed-epoch boundary, but pending gradients and native optimizer cadence are not restored, so uninterrupted and resumed optimization are not exactly equivalent. Record every interruption/rollback and actual update count. The two-epoch diagnostic cannot certify sustained 80-epoch stability or performance. Its nominal 1,725 attempts remain below the 2,000-success scaler growth interval; actual growth/backoff behavior is covered by the existing focused tests and baseline evidence, not a new long-horizon rarity run. These are disclosed infrastructure/optimization limits, not a change to the scientific sampling rule.

The final non-training preflight verifies Python 3.12.5, pinned packages/CUDA 12.6, NVIDIA GeForce RTX 4050 Laptop GPU, actual CUDA operation, all 34,298 prepared train/val files and journal/COCO/split hashes. It reports 4.944/5.997 GiB free/total VRAM and 93.91 GiB free disk, opens zero test files, confirms 80 epochs and leaves the official output absent. Full input hashing can take several minutes. Hard minimums remain 3.5 GiB free VRAM and 5 GiB free disk; free 4+ GiB VRAM is preferable.

The completed 80-epoch baseline measured 8.56 h of train+validation epoch work. The current second diagnostic epoch takes about 429 s of train+validation work; extrapolating that single steady epoch gives roughly 9.6 h before extra overhead. Reserve approximately **10–12 hours** for the future run, including startup/checkpoints and ordinary overhead. This is a planning allowance, not a measured 80-epoch rarity run. The short trajectory and transient hardware/background load limit precision, so the earlier Phase 4A 9–10 h estimate is not a guarantee. Baseline full-state checkpoint storage is 2.42 GiB across 12 files. Allow at least 5 GiB free (preflight enforced), preferably 6 GiB for the new condition, audit logs, atomic temporary files and recovery segments; extra continuations need extra space. Private diagnostic weights/logs and all previous evidence remain preserved.

## 6. Exact manual launch and operation

Run these commands from the repository root using its existing environment. Importing the module, reading the configuration, checking status or running preflight never starts training.

**Verified preflight (no training):**

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch preflight
```

**Official 80-epoch start — for the student to execute manually:**

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch start --confirm-rarity-only-80-epochs
```

This confirmed start rechecks current inputs and readiness, exclusively claims `outputs/training/rarity_only`, uses original COCO weights and 80 epochs, then logs actual training progress. No smoke output or baseline directory can be used as this new output. An existing official directory is rejected. No successful smoke starts this command automatically.

**Progress from another terminal:**

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch status
.\.venv\Scripts\python.exe -m wrist_rarity.launch monitor
```

Use `active_run.json` to identify the active segment. Its `progress.log`, `progress.jsonl`, `results.csv`, `per_class.jsonl`, `events.jsonl`, `sampling/epochNNN_plan.npz`, `sampling/epochNNN_consumed.jsonl` and `sampling/epochNNN_verified.json` retain progress, exact consumption and health evidence privately. Last/best/rolling predecessors and completed ten-epoch snapshots have SHA sidecars in `weights/`. `completed.json` indicates clean completion; `failure.json` preserves errors. Do not remove earlier segments/checkpoints or treat a partial epoch as complete.

**Interruption/recovery, only when an unfinished verified checkpoint exists:**

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.launch preflight --resume --checkpoint last
.\.venv\Scripts\python.exe -m wrist_rarity.launch resume --confirm-resume-rarity-only --checkpoint last
```

Recovery requires identical common/condition/source identity, verified live FP32 state and an unfinished 80-epoch budget. It creates a new continuation segment and preserves earlier files/results. If the newest checkpoint pair is invalid, inspect and explicitly select `last.previous` or a valid periodic checkpoint such as `epoch070`; do not bypass hash checks. A root-relative earlier-segment checkpoint path is supported when needed. A completed 80-epoch checkpoint cannot continue beyond the budget. Best inference snapshots are not substitutes for last/periodic recovery state. Prefer an uninterrupted official comparison. After a continuation, review and report discarded gradients, changed update cadence and actual attempts before claiming the uninterrupted update budget was reproduced. If interrupted before the first complete checkpoint, preserve the failed output and plan a fresh COCO start in an absent official directory; do not overwrite it or claim exact continuation.

**After the future official run, verify its completed lineage and preserved baseline:**

```powershell
.\.venv\Scripts\python.exe -m wrist_rarity.verify run
```

The command follows the selected continuation chain, reconciles every completed epoch's actual batches/labels/caps, validates scalar metrics/update accounting and reloads hash-verified last/best. It writes private post-training verification. It does not run D/E/F or test inference. Preserve/report discarded rollback work separately. Synthetic CUDA reload can be requested explicitly with `--cuda-reload`; it is not a dataset evaluation.

For a read-only baseline/data preservation check after the future run, keep the Phase 4B receipts intact and compare inventories:

```powershell
@'
import json
from wrist_rarity.protocol import PUBLIC
from wrist_eval.common import baseline_inventory, source_snapshot
from wrist_train.protocol import canonical_hash
expected = json.loads((PUBLIC / "preservation_before.json").read_text())
assert canonical_hash(baseline_inventory()) == expected["baseline_inventory_sha256"]
assert canonical_hash(source_snapshot()) == expected["data_stat_inventory_sha256"]
print("Baseline and immutable data inventories match")
'@ | .\.venv\Scripts\python.exe -
```

## 7. Evidence and readiness decision

Public compact evidence: `authorization.json`, `condition_lock.json`, `pre_gpu_tests.json`, `plan_reconciliation.json`, before/after preservation, `smoke_verification.json`, `recovery_verification.json`, current preflight/argument/seed checks and publication receipt. Private source-index/label/augmentation records, predictions, radiographs, logs and full checkpoints remain ignored. No patient IDs or image records are published.

**READY WITH DOCUMENTED LIMITATIONS:** all required engineering gates pass, including the real two-complete-epoch native transition and checkpoint reload. The known worker-lifecycle and epoch-boundary-resume differences are disclosed; no invalid sampling comparison, unresolved numerical failure or source-preservation issue remains. Exact uninterrupted/resumed stochastic optimization and full-budget performance are unmeasured. The official run remains at 80 epochs and has not started.

Stop here for student review and manual launch. No D/E/F evaluation, held-out test access or further experimental-design change is performed.
