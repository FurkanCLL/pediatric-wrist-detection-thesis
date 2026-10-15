"""Export aggregate diagnostic evidence after successful benchmark and smoke gates."""

import argparse
import json

from .budget import optimizer_budget
from .common import OUTPUT, ROOT, save_json, verify_frozen_data
from wrist_prepare.common import digest


def export(benchmark_name, smoke_name):
    benchmark = json.loads((OUTPUT / benchmark_name / "result.json").read_text())
    smoke = json.loads((OUTPUT / smoke_name / "result.json").read_text())
    assert benchmark["status"] == smoke["status"] == "passed"
    assert smoke["epochs_completed"] == 1 and not smoke["test_evaluated"]
    assert not smoke["official_training_started"]
    _, frozen = verify_frozen_data()
    environment = json.loads((OUTPUT / "environment/cuda_verification.json").read_text())
    assert environment["cuda_available"] and environment["allocated_after_cleanup_bytes"] == environment["baseline_allocated_bytes"]
    directory = ROOT / "docs/evidence/gpu_feasibility"
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "environment.json", environment)
    save_json(directory / "dependency_changes.json", json.loads((OUTPUT / "environment/dependency_changes.json").read_text()))
    save_json(directory / "verification.json", json.loads((OUTPUT / "completion_verification.json").read_text()))
    save_json(directory / "hardware.json", json.loads((OUTPUT / "environment/hardware.json").read_text(encoding="utf-8-sig")))
    save_json(directory / "pretrained_assets.json", json.loads((OUTPUT / "environment/pretrained_assets.json").read_text()))
    save_json(directory / "pretrained_verification.json", json.loads((OUTPUT / "environment/pretrained_verification.json").read_text()))
    save_json(directory / "cuda_wheels.json", json.loads((OUTPUT / "environment/wheel_verification.json").read_text()))
    save_json(directory / "benchmark.json", benchmark)
    precision = json.loads((OUTPUT / "precision_probe_01/result.json").read_text())
    for row in precision["results"]:
        row["nonfinite_gradient_layer_count"] = len(row.pop("nonfinite_gradient_layers"))
    save_json(directory / "precision_probe.json", precision)
    # Keep diagnostic AP/loss values local: these checkpoints are not study results.
    metrics = smoke.pop("diagnostic_metrics_only")
    smoke["metrics_artifact"] = {"rows": 1, "columns": list(metrics), "all_values_finite": True,
                                 "interpretation": "technical diagnostic only; no performance conclusion"}
    save_json(directory / "smoke_validation.json", smoke)
    settings = json.loads((OUTPUT / smoke_name / "resolved_config.json").read_text())
    estimates = []
    validation = smoke["validation_runs"][0]["seconds"]
    for epochs in (10, 20, 30):
        one = epochs * (smoke["training_s"] + validation)
        estimates.append({"epochs": epochs, "one_model_hours": one / 3600, "both_models_hours": 2 * one / 3600,
                          "both_models_hours_with_15_percent_margin": 2 * one * 1.15 / 3600,
                          "budget": optimizer_budget(14061, smoke["batch"], settings["nbs"], epochs, settings["warmup_epochs"])})
    save_json(directory / "duration_estimates.json", {"basis": "one measured diagnostic epoch and trainer-integrated validation; linear extrapolation, not a guarantee",
              "measured_training_s": smoke["training_s"], "measured_validation_s": validation,
              "additional_final_validation_per_model_s": validation, "estimates": estimates})
    failures = [{"attempt": path.parent.name, "error_type": json.loads(path.read_text())["type"],
                 "stage": json.loads(path.read_text())["request"]["action"]}
                for path in sorted(OUTPUT.glob("*/failure.json"))]
    save_json(directory / "failed_attempts.json", {"count": len(failures), "attempts": failures,
              "details": "Full tracebacks and any data paths remain in ignored local outputs"})
    package = ROOT / ".venv/Lib/site-packages/ultralytics"
    source_files = ("engine/trainer.py", "engine/validator.py", "models/yolo/detect/train.py", "data/augment.py", "data/build.py", "utils/torch_utils.py", "utils/metrics.py", "cfg/default.yaml")
    # A later documentation commit must not change the identity of the completed run.
    smoke_provenance = json.loads((OUTPUT / smoke_name / "provenance.json").read_text())
    implementation = smoke_provenance["implementation_commit"]
    save_json(directory / "provenance.json", {"implementation_commit": implementation,
              "candidate_config_sha256": smoke_provenance["candidate_config_sha256"],
              "split_manifest_sha256": frozen["split_manifest_sha256"],
              "framework_source_sha256": {name: digest((package / name).read_bytes()) for name in source_files},
              "benchmark_attempt": benchmark_name, "smoke_attempt": smoke_name,
              "benchmark_start_commit": json.loads((OUTPUT / benchmark_name / "provenance.json").read_text())["implementation_commit"],
              "benchmark_uncommitted_source_changes": json.loads((OUTPUT / benchmark_name / "provenance.json").read_text()).get("uncommitted_source_changes", "not recorded in early receipt"),
              "smoke_start_commit": json.loads((OUTPUT / smoke_name / "provenance.json").read_text())["implementation_commit"],
              "smoke_started_with_uncommitted_source_changes": json.loads((OUTPUT / smoke_name / "provenance.json").read_text())["uncommitted_source_changes"],
              "official_protocol_frozen": False, "test_evaluated": False})
    print(json.dumps({"exported": str(directory.relative_to(ROOT)), "official_protocol_frozen": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--smoke", required=True)
    args = parser.parse_args()
    export(args.benchmark, args.smoke)
