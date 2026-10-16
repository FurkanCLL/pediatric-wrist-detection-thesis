"""Bind the unchanged historical proposal to the student's Phase 3C.2 authorization."""

import json
from pathlib import Path

from .common import ROOT, identity


def approved_settings():
    manifest = json.loads((ROOT/"configs/adaptive_refinement_execution.json").read_text(encoding="utf-8"))
    proposal_path = ROOT/manifest["proposal_path"]
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if not manifest["student_approved"] or not manifest["execution_authorized"] or manifest["split"] != "val":
        raise ValueError("No validation-only execution authorization")
    if identity(proposal) != manifest["proposal_canonical_sha256"]:
        raise ValueError("Historical CG-ALR proposal changed after approval")
    if manifest["proposal_canonical_sha256"] != "a2e96ff54fdee455aefcb698b8a49008572a22decdb535cc2e65cd6d73997cc6":
        raise ValueError("Unrecognized approved algorithm identity")
    return proposal, manifest
