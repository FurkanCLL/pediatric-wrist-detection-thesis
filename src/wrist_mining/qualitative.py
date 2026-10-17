"""Bounded private training audit; no clinical negative adjudication."""

import json
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .common import ROOT, LOCAL, PUBLIC, CLASSES, training_records, training_path, identity, save, sha
from .analysis import predictions
from wrist_eval.analyze import truth
from wrist_eval.qualitative import draw_panel


def run():
    rows = training_records()
    ranked = sorted(rows, key=lambda r: identity({"purpose": "phase4a_qualitative", "seed": 42, "stem": r["filestem"]}))
    details = {r["stem"]: r for r in (json.loads(line) for line in (LOCAL/"analysis/best_details.jsonl").read_text().splitlines())}
    pred = predictions()
    features = np.load(LOCAL/"analysis/features.npz")
    uncertain = {r["filestem"]: bool(flag) for r, flag in zip(rows, features["uncertain"])}
    selected, absent, used = [], [], set()
    for c in range(8):
        eligible = []
        for row in ranked:
            values = pred[row["filestem"]]
            matched = np.array(details[row["filestem"]]["operating"]["0.5"]["matched_pred"])
            if np.any((values[:, 5] == c) & (values[:, 4] >= .25) & (matched < 0)):
                eligible.append(row)
        patients = set()
        count = 0
        for row in eligible:
            if row["filestem"] in used or row["patient_id"] in patients:
                continue
            values = pred[row["filestem"]]
            match = np.array(details[row["filestem"]]["operating"]["0.5"]["matched_pred"])
            index = int(np.flatnonzero((values[:, 5] == c) & (values[:, 4] >= .25) & (match < 0))[0])
            selected.append((f"potential_fp_{CLASSES[c]}", row, "prediction", index, len(eligible)))
            used.add(row["filestem"]); patients.add(row["patient_id"]); count += 1
            if count == 2:
                break
        if count < 2:
            absent.append({"category": f"potential_fp_{CLASSES[c]}", "selected": count, "eligible_images": len(eligible)})
    for c in (0, 3, 4, 5, 6, 7):
        eligible = [r for r in ranked if any(o["class_id"] == c and o["difficulty"] > 0 for o in details[r["filestem"]]["objects"])]
        candidate = next((r for r in eligible if r["filestem"] not in used), None)
        if candidate:
            index = next(i for i, o in enumerate(details[candidate["filestem"]]["objects"]) if o["class_id"] == c and o["difficulty"] > 0)
            selected.append((f"hard_positive_{CLASSES[c]}", candidate, "annotation", index, len(eligible)))
            used.add(candidate["filestem"])
        else:
            absent.append({"category": f"hard_positive_{CLASSES[c]}", "selected": 0, "eligible_images": len(eligible)})
    controls = []
    for row in ranked:
        objects = [o for o in details[row["filestem"]]["objects"] if o["class_id"] in (0, 3, 4, 5, 6, 7)]
        if objects and all(o["difficulty"] == 0 for o in objects) and row["filestem"] not in used:
            controls.append(row)
    for row in controls[:2]:
        selected.append(("correct_clinical_control", row, "annotation", next(i for i, b in enumerate(row["boxes"]) if b["class_id"] in (0, 3, 4, 5, 6, 7)), len(controls)))
    if len(selected) > 24:
        raise ValueError("Qualitative audit budget exceeded")
    directory = LOCAL/"qualitative"
    directory.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    public, private, sheets = [], [], []
    for ordinal, (category, row, focus_type, focus_index, eligible) in enumerate(selected, 1):
        image = Image.open(training_path(row)).convert("RGB")
        values, gt = pred[row["filestem"]], truth(row)
        focus = values[focus_index, :4] if focus_type == "prediction" else gt[focus_index, :4]
        margin = max(float(max(focus[2:]-focus[:2])), min(image.size)*.15)
        x0, y0 = max(0, math.floor(focus[0]-margin)), max(0, math.floor(focus[1]-margin))
        x1, y1 = min(image.width, math.ceil(focus[2]+margin)), min(image.height, math.ceil(focus[3]+margin))
        panel = Image.new("RGB", (1040, 1020), "#101418")
        panel.paste(draw_panel(image, gt, values, f"Case {ordinal}: {category}", font), (0, 0))
        crop = image.crop((x0, y0, x1, y1))
        scale = min(500/crop.width, 850/crop.height)
        crop = crop.resize((round(crop.width*scale), round(crop.height*scale)))
        ox, oy = 520+(520-crop.width)//2, 100
        panel.paste(crop, (ox, oy))
        draw = ImageDraw.Draw(panel)
        draw.text((532, 15), "GT green | selected focus yellow", font=font, fill="white")
        draw.text((532, 40), f"Released uncertainty mark: {uncertain[row['filestem']]}", font=font, fill="white")
        for i, b in enumerate(gt):
            clipped = [max(x0, float(b[0])), max(y0, float(b[1])), min(x1, float(b[2])), min(y1, float(b[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            xy = [ox+(clipped[0]-x0)*scale, oy+(clipped[1]-y0)*scale, ox+(clipped[2]-x0)*scale, oy+(clipped[3]-y0)*scale]
            draw.rectangle(xy, outline="#47ef76", width=2)
            draw.text((xy[0], xy[1]), f"G{i} c{int(b[4])}", fill="#47ef76", font=font, stroke_width=1, stroke_fill="black")
        xy = [ox+(focus[0]-x0)*scale, oy+(focus[1]-y0)*scale, ox+(focus[2]-x0)*scale, oy+(focus[3]-y0)*scale]
        draw.rectangle(xy, outline="yellow", width=3)
        focus_class = int(values[focus_index, 5] if focus_type == "prediction" else gt[focus_index, 4])
        reason = details[row["filestem"]]["operating"]["0.5"]["potential_fp_reasons"][focus_index] if focus_type == "prediction" else details[row["filestem"]]["operating"]["0.5"]["gt_reasons"][focus_index]
        draw.text((532, 70), f"Focus c{focus_class}: {reason}", font=font, fill="yellow")
        filename = f"case_{ordinal:02}.png"
        panel.save(directory/filename)
        sheets.append(panel)
        public.append({"case": ordinal, "category": category, "class": CLASSES[focus_class], "eligible_images": eligible,
            "descriptive_reason": reason, "source_uncertainty_mark": uncertain[row["filestem"]],
            "review": "pending actual technical inspection", "verified_clinical_negative": False})
        private.append({"case": ordinal, "stem": row["filestem"], "patient_id": row["patient_id"],
                        "focus_type": focus_type, "focus_index": focus_index, "figure": filename})
    for start in range(0, len(sheets), 4):
        montage = Image.new("RGB", (2080, 2040), "black")
        for i, panel in enumerate(sheets[start:start+4]):
            montage.paste(panel, ((i % 2)*1040, (i//2)*1020))
        montage.save(directory/f"review_sheet_{start//4+1:02}.png")
    save(directory/"selection.json", private)
    save(PUBLIC/"qualitative_audit.json", {"cases": public, "absent_categories": absent,
        "selection_sha256": sha(directory/"selection.json"), "images_private": True,
        "selection": "Predefined hash ordering; no hand-picked replacements", "technical_review_completed": False,
        "clinical_adjudication": False, "verified_negative_pool_size": 0})


if __name__ == "__main__":
    run()
