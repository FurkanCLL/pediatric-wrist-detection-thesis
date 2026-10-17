"""Read-only preservation checks without opening validation or test images/labels."""

import argparse
import json
import subprocess

from .common import ROOT, PUBLIC, LOCAL, config, identity, sha, save
from wrist_eval.common import baseline_inventory, source_snapshot
from wrist_train.protocol import load_protocol


def verify(stage):
    cfg = config()
    baseline = baseline_inventory()
    expected = json.loads((ROOT/"outputs/error_analysis_sahi/baseline_inventory_before.json").read_text())
    if baseline != expected:
        raise ValueError("Preserved baseline files changed")
    snapshot = source_snapshot()
    old_snapshot = json.loads((ROOT/"outputs/error_analysis_sahi/source_snapshot_before.json").read_text())
    if identity(snapshot) != identity(old_snapshot):
        raise ValueError("Raw/prepared stat inventory changed")
    _, lock = load_protocol()
    protected = ["src/wrist_train", "src/wrist_design", "src/wrist_eval", "src/wrist_prepare", "src/wrist_audit",
        "configs/class_aware_candidates.json", "configs/training_official.json", "configs/dataset_preparation.json",
        "configs/inference_feasibility.json", "configs/adaptive_refinement_proposed.json", "configs/adaptive_refinement_execution.json",
        "docs/evidence/class_aware_design", "docs/evidence/error_analysis_sahi", "docs/evidence/adaptive_refinement_validation",
        "docs/evidence/adaptive_refinement_design", "docs/evidence/training_protocol", "docs/evidence/dataset_preparation"]
    if subprocess.check_output(["git", "diff", "042bbc9", "--", *protected], cwd=ROOT).strip():
        raise ValueError("Frozen implementation or historical evidence changed")
    result = {"stage": stage, "baseline_files": len(baseline), "baseline_inventory_sha256": identity(baseline),
        "all_baseline_hashes_match": True, "raw_prepared_stat_files": len(snapshot), "all_stat_entries_match": True,
        "frozen_protocol_sha256": lock["protocol_sha256"], "study_config_sha256": identity(cfg),
        "protected_pre_phase_source_config_evidence_unchanged": True,
        "no_validation_or_test_image_label_content_opened": True, "official_training_performed": False,
        "historical_matrix_unchanged": True}
    save(PUBLIC/f"preservation_{stage}.json", result)
    print(f"Preservation {stage}: {len(baseline)} baseline artifacts and {len(snapshot)} stat entries match", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["before", "after"])
    verify(parser.parse_args().stage)
