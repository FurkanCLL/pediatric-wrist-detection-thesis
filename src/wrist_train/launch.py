"""Explicit manual start/resume commands; preflight never starts training."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

from .common import OUTPUT, configure_runtime, write_data_yaml
from .checkpoints import atomic_bytes, load_recovery, publish_checkpoint
from .protocol import RUN_ROOT, preflight

configure_runtime()


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # inaccessible processes are treated as active
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def run_lock(root):
    path = root / "run.lock"
    if path.exists():
        existing = json.loads(path.read_text())
        if process_alive(existing["pid"]):
            raise RuntimeError("Another official training process holds the run lock")
        path.rename(root / ("stale-lock-" + uuid.uuid4().hex + ".json"))
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_utc": datetime.now(timezone.utc).isoformat()}, stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def active_directory(root):
    metadata = json.loads((root / "active_run.json").read_text())
    directory = (root / metadata["directory"]).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Active output directory escapes the official run")
    return directory


def checkpoint_path(root, name):
    if name in {"last", "last.previous"} or re.fullmatch(r"epoch\d{3}", name):
        path = active_directory(root) / "weights" / (name + ".pt")
    else:
        path = root / name
    path = path.resolve()
    if path.suffix != ".pt" or not path.is_relative_to(root.resolve()):
        raise ValueError("Resume accepts only official checkpoint files inside this run")
    if not path.is_file():
        raise FileNotFoundError("No completed checkpoint at that location; inspect active_run.json and prior segments")
    return path


def prepare_directory(root, identity, resume):
    if not resume:
        root.mkdir(exist_ok=False)
        write_json(root / "run_identity.json", identity)
        return root
    existing = json.loads((root / "run_identity.json").read_text())
    if existing != identity:
        raise ValueError("Existing official run has a different frozen identity")
    # Each continuation gets its own files; earlier checkpoints and metrics remain intact.
    directory = root / "resumes" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def resume_state(root, name, identity):
    if json.loads((root / "run_identity.json").read_text()) != identity:
        raise ValueError("Existing run identity differs from the frozen protocol")
    source = checkpoint_path(root, name)
    return source, load_recovery(source, identity)


def train_manually(resume=False, checkpoint_name="last"):
    protocol, identity, _ = preflight(resume)
    checkpoint = None
    source = OUTPUT / "weights/yolov8s.pt"
    if not resume:
        directory = prepare_directory(RUN_ROOT, identity, False)
    with run_lock(RUN_ROOT):
        if resume:
            # Select and verify recovery bytes only after excluding another training writer.
            source, checkpoint = resume_state(RUN_ROOT, checkpoint_name, identity)
            directory = prepare_directory(RUN_ROOT, identity, True)
        data, _ = write_data_yaml(directory)
        overrides = protocol["training"].copy()
        overrides.update(model=str(source), data=str(data), project=str(directory.parent), name=directory.name,
                         exist_ok=True, resume=str(source) if resume else False)
        # Native exist_ok is safe here only after our exclusive directory claim and run lock.
        write_json(directory / "frozen_protocol.json", protocol)
        write_json(directory / "launch.json", {"identity": identity, "resume": resume,
            "source": str(source), "started_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_revision": identity["implementation_revision"]})
        if resume:
            from .official import restore_results_csv
            restore_results_csv(directory / "results.csv", checkpoint["train_results"])
            # The selected recovery file embeds the best weights known at that epoch, even after a rollback.
            best = {"epoch": checkpoint["official"]["best_epoch"], "model": None,
                    "ema": checkpoint["best_ema"], "best_ema": checkpoint["best_ema"],
                    "train_args": overrides, "best_fitness": checkpoint["best_fitness"],
                    "official": {**identity, "format_version": 1, "resumable": False}}
            publish_checkpoint(directory / "weights/best.pt", best)
        write_json(RUN_ROOT / "active_run.json", {"directory": directory.relative_to(RUN_ROOT).as_posix(),
            "resume_checkpoint": str(source.relative_to(RUN_ROOT)) if resume else None})
        try:
            from .official import OfficialTrainer
            trainer = OfficialTrainer(protocol, identity, directory, overrides, checkpoint)
            trainer.train()
            write_json(directory / "completed.json", {"epochs": protocol["training"]["epochs"], "successful_updates": trainer.monitor.successful,
                                                     "skipped_updates": trainer.monitor.skipped})
        except BaseException as error:
            write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                   "traceback": traceback.format_exc()})
            raise


def status(root=RUN_ROOT):
    directory = active_directory(root)
    print("Active output: " + str(directory), flush=True)
    progress = directory / "progress.log"
    print("\n".join(progress.read_text().splitlines()[-2:]) if progress.exists() else "No completed epoch summary yet.", flush=True)


def monitor():
    positions = {}
    while True:
        directory = active_directory(RUN_ROOT)
        path = directory / "progress.log"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                stream.seek(positions.get(path, 0))
                text = stream.read()
                positions[path] = stream.tell()
            if text:
                print(text, end="", flush=True)
        if (directory / "completed.json").exists() or (directory / "failure.json").exists():
            return
        time.sleep(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--resume", action="store_true")
    check.add_argument("--checkpoint", default="last")
    start = commands.add_parser("start")
    start.add_argument("--confirm-standard-baseline", action="store_true")
    restart = commands.add_parser("resume")
    restart.add_argument("--confirm-resume-standard-baseline", action="store_true")
    restart.add_argument("--checkpoint", default="last")
    commands.add_parser("status")
    commands.add_parser("monitor")
    args = parser.parse_args(argv)
    # Reject accidental starts before preflight, directory creation or trainer construction.
    if args.action == "start" and not args.confirm_standard_baseline:
        parser.error("Start requires --confirm-standard-baseline")
    if args.action == "resume" and not args.confirm_resume_standard_baseline:
        parser.error("Resume requires --confirm-resume-standard-baseline")
    if args.action == "preflight":
        _, identity, result = preflight(args.resume)
        if args.resume:
            _, payload = resume_state(RUN_ROOT, args.checkpoint, identity)
            result["resume"] = {"completed_epoch": payload["epoch"] + 1, "saved_amp_scale": payload["scaler"]["scale"],
                                "accounting": payload["accounting"]}
        print(json.dumps(result, indent=2), flush=True)
    elif args.action == "start":
        train_manually()
    elif args.action == "resume":
        train_manually(True, args.checkpoint)
    elif args.action == "status":
        status()
    else:
        monitor()


if __name__ == "__main__":
    main()
