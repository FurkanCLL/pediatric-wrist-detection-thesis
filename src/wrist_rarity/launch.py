"""Separate manual 80-epoch launch and explicitly bounded diagnostic commands."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

from wrist_train.common import OUTPUT, write_data_yaml
from wrist_train.launch import write_json, run_lock, prepare_directory, resume_state, status, active_directory
from wrist_train.checkpoints import load_recovery
from .protocol import ROOT, LOCAL, PUBLIC, RUN_ROOT, preflight, load_condition


def check_condition_checkpoint(payload, identity):
    # Baseline checkpoint checks alone do not know about the additive sampling condition.
    for key in ("rarity_condition_sha256", "rarity_source_sha256", "diagnostic_only", "condition"):
        if payload["official"].get(key) != identity[key]:
            raise ValueError("Checkpoint rarity-condition identity differs: "+key)
    receipt = payload["official"].get("rarity_completed_epoch", {})
    if receipt.get("epoch") != payload["epoch"]+1 or not receipt.get("consumed_indices_exact_match"):
        raise ValueError("Checkpoint lacks matching completed source-consumption evidence")


def training_gate(identity, diagnostic):
    gate = json.loads((PUBLIC/"pre_gpu_tests.json").read_text())
    plans = json.loads((PUBLIC/"plan_reconciliation.json").read_text())
    if not gate["passed"] or gate["condition_sha256"] != identity["rarity_condition_sha256"] or not plans["all240_exact_match_phase4a"] or plans["condition_sha256"] != identity["rarity_condition_sha256"]:
        raise ValueError("Pre-GPU automated gates are missing or stale")
    if not diagnostic:
        smoke = json.loads((PUBLIC/"smoke_verification.json").read_text())
        if not smoke["passed"] or smoke["condition_sha256"] != identity["rarity_condition_sha256"]:
            raise ValueError("The verified two-epoch integration smoke is missing or stale")


def train(diagnostic=False, attempt=None, resume=False, checkpoint_name="last"):
    if diagnostic:
        if resume or not attempt or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", attempt):
            raise ValueError("Use a new simple diagnostic attempt name; diagnostic resume is disabled")
        root = LOCAL/"diagnostics"/attempt
    else:
        root = RUN_ROOT
    protocol, identity, preflight_result = preflight(resume, diagnostic, root if diagnostic else None)
    training_gate(identity, diagnostic)
    checkpoint = None
    source = OUTPUT/"weights/yolov8s.pt"
    if not resume:
        directory = prepare_directory(root, identity, False)
    with run_lock(root):
        if resume:
            source, checkpoint = resume_state(root, checkpoint_name, identity)
            check_condition_checkpoint(checkpoint, identity)
            directory = prepare_directory(root, identity, True)
        data, _ = write_data_yaml(directory)
        overrides = protocol["training"].copy()
        overrides.update(model=str(source), data=str(data), project=str(directory.parent), name=directory.name,
                         exist_ok=True, resume=str(source) if resume else False)
        write_json(directory/"condition_protocol.json", protocol)
        write_json(directory/"launch.json", {"identity": identity, "diagnostic_only": diagnostic, "resume": resume,
                   "source": str(source), "preflight": preflight_result,
                   "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "started_utc": datetime.now(timezone.utc).isoformat()})
        if resume:
            from wrist_train.official import restore_results_csv
            from wrist_train.checkpoints import publish_checkpoint
            restore_results_csv(directory/"results.csv", checkpoint["train_results"])
            best = {"epoch": checkpoint["official"]["best_epoch"], "model": checkpoint["best_ema"], "ema": checkpoint["best_ema"],
                    "best_ema": checkpoint["best_ema"], "train_args": overrides, "best_fitness": checkpoint["best_fitness"],
                    "official": {**identity, "format_version": 1, "resumable": False, "best_snapshot_only": True,
                                 "rarity_completed_epoch": checkpoint["official"]["rarity_best_epoch_evidence"]}}
            publish_checkpoint(directory/"weights/best.pt", best)
        write_json(root/"active_run.json", {"directory": directory.relative_to(root).as_posix(),
                   "resume_checkpoint": str(source.relative_to(root)) if resume else None})
        started = time.perf_counter()
        try:
            from wrist_train.diagnostics import Telemetry
            from .trainer import RarityTrainer
            with Telemetry() as telemetry:
                trainer = RarityTrainer(protocol, identity, directory, overrides, checkpoint)
                trainer.train()
            write_json(directory/"completed.json", {"epochs": protocol["training"]["epochs"],
                       "diagnostic_only": diagnostic, "wall_seconds": time.perf_counter()-started,
                       "accounting": trainer.monitor.state_dict(), "validation_runs": trainer.validation_runs,
                       "telemetry": telemetry.summary(), "best_epoch": trainer.best_epoch+1,
                       "successful_updates": trainer.monitor.successful, "skipped_updates": trainer.monitor.skipped})
        except BaseException as error:
            write_json(directory/"failure.json", {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
            raise


def monitor(root=RUN_ROOT):
    positions = {}
    while True:
        directory = active_directory(root)
        path = directory/"progress.log"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                stream.seek(positions.get(path, 0))
                text = stream.read()
                positions[path] = stream.tell()
            if text:
                print(text, end="", flush=True)
        if (directory/"completed.json").exists() or (directory/"failure.json").exists():
            return
        time.sleep(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--resume", action="store_true")
    check.add_argument("--checkpoint", default="last")
    start = commands.add_parser("start")
    start.add_argument("--confirm-rarity-only-80-epochs", action="store_true")
    restart = commands.add_parser("resume")
    restart.add_argument("--confirm-resume-rarity-only", action="store_true")
    restart.add_argument("--checkpoint", default="last")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--attempt", required=True)
    commands.add_parser("status")
    commands.add_parser("monitor")
    args = parser.parse_args(argv)
    # Configuration inspection and preflight can never fall through to model fitting.
    if args.action == "start" and not args.confirm_rarity_only_80_epochs:
        parser.error("Start requires --confirm-rarity-only-80-epochs")
    if args.action == "resume" and not args.confirm_resume_rarity_only:
        parser.error("Resume requires --confirm-resume-rarity-only")
    if args.action == "preflight":
        _, identity, result = preflight(args.resume)
        training_gate(identity, False)
        if args.resume:
            _, payload = resume_state(RUN_ROOT, args.checkpoint, identity)
            check_condition_checkpoint(payload, identity)
            result["resume_completed_epoch"] = payload["epoch"]+1
        print(json.dumps(result, indent=2), flush=True)
    elif args.action == "smoke":
        train(True, args.attempt)
    elif args.action == "start":
        train()
    elif args.action == "resume":
        train(resume=True, checkpoint_name=args.checkpoint)
    elif args.action == "status":
        status(RUN_ROOT)
    else:
        monitor()


if __name__ == "__main__":
    main()
