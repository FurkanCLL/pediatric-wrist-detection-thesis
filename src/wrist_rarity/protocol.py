"""Additive condition identity; the original common protocol and lock remain immutable."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from wrist_train.common import ROOT
from wrist_train.protocol import load_protocol, canonical_hash, environment_check, output_check, verify_inputs
from wrist_train.launch import write_json
from .data import label_signature
from .policy import RarityPlans

PUBLIC = ROOT/"docs/evidence/rarity_training_readiness"
LOCAL = ROOT/"outputs/rarity_training_readiness"
RUN_ROOT = ROOT/"outputs/training/rarity_only"
CONFIG = ROOT/"configs/rarity_training.json"
LOCK = PUBLIC/"condition_lock.json"
SOURCES = ["src/wrist_rarity/"+name for name in ("__init__.py", "policy.py", "data.py", "protocol.py", "trainer.py", "launch.py", "verify.py")]
SOURCES += ["src/wrist_mining/sampling.py", "src/wrist_design/sampling.py", "src/wrist_design/common.py"]


def extension_sources():
    return {name: hashlib.sha256((ROOT/name).read_text(encoding="utf-8").encode()).hexdigest() for name in SOURCES}


def condition_definition():
    common, original = load_protocol()
    cfg = json.loads(CONFIG.read_text())
    proposal = json.loads((ROOT/"configs/error_guided_sampler_proposed.json").read_text())
    if cfg["common_protocol_sha256"] != original["protocol_sha256"] or cfg["phase4a_sampler_proposal_sha256"] != canonical_hash(proposal):
        raise ValueError("The approved common protocol or sampler reference differs")
    fixed = {"condition": "patient_aware_rarity_only", "official_epochs": 80, "diagnostic_epochs": 2,
             "seed": 42, "source_draws": 14061, "realized_replacements": 625,
             "official_output": "outputs/training/rarity_only", "diagnostic_output_parent": "outputs/rarity_training_readiness/diagnostics",
             "automatic_official_launch": False, "test_access": False}
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Approved rarity-only settings changed")
    # The original framework lock omits these two dataset implementation files.
    import ultralytics
    base = Path(ultralytics.__file__).parent
    dataset_sources = {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ("data/base.py", "data/dataset.py")}
    return common, original, {"config_sha256": canonical_hash(cfg), "common_protocol_sha256": original["protocol_sha256"],
        "source_sha256": extension_sources(), "extra_framework_sha256": dataset_sources,
        "sampler_proposal_sha256": canonical_hash(proposal), "official_epochs": 80}


def load_condition(diagnostic=False):
    common, identity, definition = condition_definition()
    if json.loads(LOCK.read_text()) != definition:
        raise ValueError("Rarity condition source/configuration changed after verification")
    protocol, identity = deepcopy(common), deepcopy(identity)
    epochs = 2 if diagnostic else 80
    protocol["training"]["epochs"] = epochs
    identity.update(training_epochs=epochs, rarity_condition_sha256=canonical_hash(definition),
                    rarity_source_sha256=definition["source_sha256"], diagnostic_only=diagnostic,
                    condition="patient_aware_rarity_only")
    return protocol, identity


def source_files_and_labels(rows):
    prepared = ROOT/"data/prepared/grazpedwri_detection"
    files, signatures = [], []
    for row in rows:
        if row["split"] != "train" or Path(row["filestem"]).name != row["filestem"]:
            raise ValueError("Non-training or invalid source metadata")
        files.append(str((prepared/"images/train"/(row["filestem"]+".png")).resolve()))
        lines = (prepared/"labels/train"/(row["filestem"]+".txt")).read_text().splitlines()
        values = np.asarray([[float(v) for v in line.split()] for line in lines], dtype=np.float32).reshape(-1, 5)
        if len(values) != len(row["boxes"]) or not np.isfinite(values).all():
            raise ValueError("Prepared annotations disagree with frozen metadata")
        signatures.append(label_signature(values))
    return files, signatures


def preflight(resume=False, diagnostic=False, directory=None):
    started = time.perf_counter()
    protocol, identity = load_condition(diagnostic)
    environment = environment_check(protocol)
    if diagnostic and directory is None:
        raise ValueError("A diagnostic attempt directory is required")
    target = Path(directory) if diagnostic else RUN_ROOT
    if diagnostic and target.resolve().parent != (LOCAL/"diagnostics").resolve():
        raise ValueError("Diagnostic output must be a separate named attempt")
    disk = output_check(target, resume)
    if disk < protocol["preflight"]["minimum_free_disk_gib"]:
        raise RuntimeError("Insufficient disk space for recovery state")
    counts = verify_inputs(protocol, identity["prepared_journal_sha256"])
    plans = RarityPlans()
    source_files_and_labels(plans.rows)
    result = {"passed": True, "diagnostic_only": diagnostic, "training_epochs": protocol["training"]["epochs"],
        "common_protocol_sha256": identity["protocol_sha256"], "condition_sha256": identity["rarity_condition_sha256"],
        "environment": environment, "free_disk_gib": disk, "images": counts, "test_files_opened": 0,
        "first_plan": plans.validate(plans.plan(42, 0)), "seconds": time.perf_counter()-started,
        "output": target.relative_to(ROOT).as_posix(), "training_started": False}
    LOCAL.mkdir(parents=True, exist_ok=True)
    write_json(LOCAL/("diagnostic_preflight.json" if diagnostic else "official_preflight.json"), result)
    return protocol, identity, result
