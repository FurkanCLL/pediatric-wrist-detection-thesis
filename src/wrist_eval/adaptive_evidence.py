"""Preservation receipts, private pilot overlays and the full-validation gate."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from .common import ROOT, identity, save, baseline_inventory, source_snapshot, verify_validation_files, validation_records
from .adaptive_config import approved_settings
from .adaptive_study import LOCAL, PUBLIC, source_identity


def preservation(stage):
    from wrist_train.protocol import load_protocol
    cfg, manifest = approved_settings()
    print("Checking all preserved baseline artifacts and frozen dataset", flush=True)
    baseline = baseline_inventory()
    snapshot = source_snapshot()
    previous = ROOT/"outputs/error_analysis_sahi"
    old_baseline = json.loads((previous/"baseline_inventory_before.json").read_text())
    old_snapshot = json.loads((previous/"source_snapshot_before.json").read_text())
    if baseline != old_baseline or identity(snapshot) != identity(old_snapshot):
        raise ValueError("Baseline or raw/prepared stat inventory changed")
    _, lock = load_protocol()
    file_checks = verify_validation_files(validation_records())
    protected = ["src/wrist_train", "src/wrist_design", "src/wrist_prepare", "src/wrist_audit",
                 "configs/adaptive_refinement_proposed.json", "configs/training_official.json",
                 "configs/class_aware_candidates.json", "configs/dataset_preparation.json",
                 "configs/inference_feasibility.json", "docs/evidence/error_analysis_sahi",
                 "docs/evidence/class_aware_design", "docs/evidence/training_protocol",
                 "docs/evidence/dataset_audit", "docs/evidence/dataset_preparation",
                 "docs/evidence/gpu_feasibility", "docs/evidence/adaptive_refinement_design"]
    if subprocess.check_output(["git", "diff", "05f38a7", "--", *protected], cwd=ROOT).strip():
        raise ValueError("Protected pre-phase tracked state changed")
    old_source = json.loads((ROOT/"docs/evidence/error_analysis_sahi/implementation_provenance.json").read_text())
    import hashlib
    for name, wanted in old_source["normalized_source_sha256"].items():
        if hashlib.sha256((ROOT/name).read_text(encoding="utf-8").encode()).hexdigest() != wanted:
            raise ValueError("Phase 3B source changed")
    result = {"stage": stage, "baseline_artifacts": len(baseline), "all_baseline_hashes_unchanged": True,
        "baseline_inventory_sha256": identity(baseline), "raw_prepared_stat_files": len(snapshot),
        "raw_prepared_stat_inventory_unchanged": True, "validation_sha256": file_checks,
        "frozen_training_protocol_sha256": lock["protocol_sha256"], "proposal_sha256": identity(cfg),
        "execution_manifest_sha256": identity(manifest), "previous_source_config_evidence_unchanged": True,
        "no_training": True, "no_test_image_or_label_contents_opened": True}
    save(PUBLIC/f"preservation_{stage}.json", result)
    print({"preservation": stage, "artifacts": len(baseline), "validation_files": file_checks}, flush=True)


def active_records(stage):
    attempt = json.loads((LOCAL/f"{stage}_active.json").read_text())["attempt"]
    directory = LOCAL/stage/attempt
    records = [json.loads(line) for line in (directory/"bundles.jsonl").read_text().splitlines()]
    return directory, records


def pilot_figures():
    from PIL import Image, ImageFont
    from .common import image_path
    from .analyze import truth
    from .qualitative import draw_panel
    directory, records = active_records("pilot")
    figures = directory/"qualitative"
    figures.mkdir(exist_ok=False)
    selections = []
    criteria = [(f"roi_count_{k}", lambda r, k=k: len(r["plan"]["rois"]) == k) for k in range(3)]
    criteria += [("confidence_update", lambda r: bool(r["variants"]["V1"]["diagnostics"]["score_changes"])),
                 ("geometry_update", lambda r: any(g["reason"] == "accepted" for g in r["variants"]["V2"]["diagnostics"]["geometry"])),
                 ("geometry_veto", lambda r: any(g["reason"] == "collision_veto" for g in r["variants"]["V2"]["diagnostics"]["geometry"])),
                 ("protected_match", lambda r: r["variants"]["V1"]["diagnostics"]["protected_matches"] > 0)]
    used, absent = set(), []
    for label, predicate in criteria:
        eligible = [i for i, r in enumerate(records) if predicate(r)]
        chosen = next((i for i in eligible if i not in used), None)
        if chosen is not None:
            used.add(chosen)
            selections.append((label, chosen))
        else:
            absent.append({"category": label, "eligible_images": len(eligible)})
    for i in range(len(records)):
        if len(selections) >= 6:
            break
        if i not in used:
            used.add(i)
            selections.append(("pilot_order_coverage", i))
    by_stem = {r["filestem"]: r for r in validation_records()}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    public_cases = []
    for case, (category, index) in enumerate(selections, 1):
        item = records[index]
        row = by_stem[item["stem"]]
        image = Image.open(image_path(row)).convert("RGB")
        sheet = Image.new("RGB", (1560, 1020), "black")
        arrays = [np.asarray(item["global"], dtype=np.float32).reshape(-1, 6)] + [
            np.asarray(item["variants"][v]["pred"], dtype=np.float32).reshape(-1, 6) for v in ("V1", "V2")]
        for panel, (name, pred) in enumerate(zip(("standard global", "CG-ALR V1", "CG-ALR V2"), arrays)):
            sheet.paste(draw_panel(image, truth(row), pred, name, font), (panel*520, 0))
        sheet.save(figures/f"case_{case:02d}_{category}.png")
        public_cases.append({"case": case, "category": category, "pilot_ordinal": index+1,
                             "review": "pending technical visual inspection"})
    save(PUBLIC/"pilot_qualitative.json", {"cases": public_cases, "absent_or_already_selected": absent,
         "selection": "First unused predefined pilot image per diagnostic category, then pilot order to six; no AP-based selection",
         "images_private": True, "clinical_review": False})


def pilot_gate():
    cfg, manifest = approved_settings()
    summary = json.loads((PUBLIC/"pilot_summary.json").read_text())
    visuals = json.loads((PUBLIC/"pilot_qualitative.json").read_text())
    geometry = json.loads((PUBLIC/"pilot_geometry.json").read_text())
    limits = cfg["phase_3c2_plan"]["proposed_success"]
    checks = {"images": summary["images"] == 24, "global_reference_exact": summary["global_reference_exact"],
        "source_identity": summary["implementation_source_sha256"] == source_identity(),
        "proposal_identity": summary["proposal_sha256"] == identity(cfg),
        "replay_exact": len(summary["deterministic_replay"]) == 3 and all(r["exact"] for r in summary["deterministic_replay"]),
        "allocated_memory": summary["allocated_peak_bytes"]/2**30 <= limits["pilot_peak_allocated_gib_max"],
        "time_budget": all(t["mean"]*3088 <= limits["projected_full_run_seconds_per_variant_max"] for t in summary["timing"].values()),
        "pass_budget": 1 <= summary["mean_passes"] <= 3,
        "visual_review": bool(visuals["cases"]) and all(c["review"] == "technically reviewed" for c in visuals["cases"]),
        "crop_visual_review": geometry["source_pixels_match"] and all(c["review"] == "technically reviewed" for c in geometry["cases"])}
    result = {"passed": all(checks.values()), "checks": checks, "implementation_source_sha256": source_identity(),
              "execution_manifest_sha256": identity(manifest), "no_ap_based_configuration_changes": True}
    save(PUBLIC/"pilot_gate.json", result)
    if not result["passed"]:
        raise ValueError("Pilot gate failed; do not run full validation")
    print(result, flush=True)


# Show actual selected pixel crops separately from final-prediction overlays.
def pilot_geometry():
    from PIL import Image, ImageDraw, ImageFont
    from .common import image_path
    directory, records = active_records("pilot")
    destination = directory/"geometry"
    destination.mkdir(exist_ok=False)
    by_stem = {r["filestem"]: r for r in validation_records()}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    cases = []
    for case, count in enumerate((1, 2), 1):
        index = next(i for i, r in enumerate(records) if len(r["plan"]["rois"]) == count)
        record = records[index]
        source = Image.open(image_path(by_stem[record["stem"]])).convert("RGB")
        panels = [(source, "Global image; cyan ROI boundaries", None)]
        for roi, view in zip(record["plan"]["rois"], record["local_views"]):
            x1, y1, x2, y2 = roi["box"]
            crop = source.crop((x1, y1, x2, y2))
            if not np.array_equal(np.asarray(crop), np.asarray(source)[y1:y2, x1:x2]):
                raise ValueError("Visualized crop differs from source pixels")
            panels.append((crop, f"ROI {roi['id']}; local target boxes before fusion", view))
        sheet = Image.new("RGB", (1560, 1020), "#101418")
        for column, (image, title, view) in enumerate(panels):
            scale = min(500/image.width, 900/image.height)
            image_resized = image.resize((round(image.width*scale), round(image.height*scale)))
            ox, oy = column*520+(520-image_resized.width)//2, 75
            sheet.paste(image_resized, (ox, oy))
            draw = ImageDraw.Draw(sheet)
            draw.text((column*520+10, 12), title, fill="white", font=font)
            if view is None:
                boxes = [(roi["box"], "#42ddff", f"ROI {roi['id']}") for roi in record["plan"]["rois"]]
            else:
                boxes = [(p[:4], "#ff9b36", f"c{int(p[5])} {p[4]:.2f}") for p in view["predictions"] if p[4] >= .25 and p[5] in (0, 3)]
            for box, color, label in boxes:
                coords = [ox+box[0]*scale, oy+box[1]*scale, ox+box[2]*scale, oy+box[3]*scale]
                draw.rectangle(coords, outline=color, width=2)
                draw.text((coords[0], max(50, coords[1]-14)), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
        sheet.save(destination/f"case_{case:02d}_rois_{count}.png")
        cases.append({"case": case, "pilot_ordinal": index+1, "rois": count, "review": "pending technical visual inspection"})
    save(PUBLIC/"pilot_geometry.json", {"cases": cases, "source_pixels_match": True,
         "purpose": "Technical crop-boundary, source-pixel and local-coordinate inspection; no new inference"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preserve_before", "preserve_after", "pilot_figures", "pilot_geometry", "pilot_gate"])
    action = parser.parse_args().action
    if action.startswith("preserve_"):
        preservation(action.split("_")[1])
    elif action == "pilot_figures":
        pilot_figures()
    elif action == "pilot_geometry":
        pilot_geometry()
    else:
        pilot_gate()
