"""Rerun the frozen verifier without replacing its existing report."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from wrist_rarity import verify
from wrist_rarity.protocol import RUN_ROOT


def write_new(path, value):
    # Exclusive creation keeps every previous audit attempt available for inspection.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inventory(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def rerun(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("Audit output must be outside the original run")
    destination.mkdir(parents=True, exist_ok=False)
    before = inventory(RUN_ROOT)
    write_new(destination/"run_inventory_before.json", before)

    def redirected(path, value):
        # Only the report destination changes; all frozen checks execute unchanged.
        if Path(path).resolve() != (RUN_ROOT/"post_training_verification.json").resolve():
            raise ValueError("Unexpected write requested by the frozen verifier")
        write_new(destination/"original_verification.json", value)

    with patch.object(verify, "write_json", side_effect=redirected):
        verify.verify_run(RUN_ROOT, diagnostic=False, cuda_reload=False)
    after = inventory(RUN_ROOT)
    if before != after:
        raise ValueError("Original training evidence changed during verification")
    write_new(destination/"original_rerun.json", {
        "passed": True, "frozen_verification_logic_unchanged": True,
        "report_write_redirected_only": True, "original_run_files_unchanged": len(before),
        "training_started": False, "new_inference": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    rerun(parser.parse_args().destination)
