"""Bounded private pixel/label comparisons from prepared training images only."""

from collections import Counter
import argparse
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw

from .common import CLASSES, LOCAL, PUBLIC, ROOT, save_json, sha256, training_records
from .crops import crop_pixels, propose_crop
from .sampling import DesignPopulation


def key(row):
    return hashlib.sha256(("42:"+row["filestem"]).encode()).hexdigest()


def annotated(image, boxes, title):
    canvas = Image.fromarray(image).convert("RGB")
    scale = min(480/canvas.width, 600/canvas.height)
    canvas = canvas.resize((round(canvas.width*scale), round(canvas.height*scale)))
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        coords = [value*scale for value in box["xyxy"]]
        color = "#36dd66" if box["class_id"] == 0 else "#ffd040"
        draw.rectangle(coords, outline=color, width=2)
        draw.text((coords[0]+2, coords[1]+2), str(box["class_id"]), fill=color)
    panel = Image.new("RGB", (500, 650), "#151515")
    panel.paste(canvas, ((500-canvas.width)//2, 38))
    ImageDraw.Draw(panel).text((10, 10), title, fill="white")
    return panel


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="visual_review")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.attempt):
        parser.error("Attempt names must be simple local directory names")
    specification = json.loads((ROOT / "configs/class_aware_candidates.json").read_text())
    population = DesignPopulation(training_records(), specification)
    output = LOCAL / args.attempt
    output.mkdir(exist_ok=False)
    # A is the smallest intervention; compare B/C on the same parents rather than cherry-picking new images.
    settings = specification["candidates"]["A"]
    weights = population.weights(settings)[0]
    rows = population.rows
    plans = [propose_crop(r, settings["crop"], specification["target_classes"], weights) for r in rows]
    selected = {}
    def select(label, eligible):
        eligible = list(eligible)
        if eligible:
            index = min(eligible, key=lambda i: key(rows[i]))
            selected.setdefault(index, []).append(label)
    for c, name in enumerate(CLASSES):
        select("class_"+name, np.flatnonzero(population.present[:, c]))
    for c in specification["target_classes"]:
        select("accepted_"+CLASSES[c], (i for i, p in enumerate(plans) if p["accepted"] and p["target_class"] == c))
    for reason in sorted({p["reason"] for p in plans if not p["accepted"]}):
        select("fallback_"+reason, (i for i, p in enumerate(plans) if p["reason"] == reason))
    for label, index in (("narrowest", min(range(len(rows)), key=lambda i: rows[i]["width"]/rows[i]["height"])),
                         ("widest", max(range(len(rows)), key=lambda i: rows[i]["width"]/rows[i]["height"]))):
        selected.setdefault(index, []).append(label)
    select("accepted_border", (i for i, p in enumerate(plans) if p["accepted"] and
                              (p["rect"][0] == 0 or p["rect"][1] == 0 or p["rect"][2] == rows[i]["width"] or p["rect"][3] == rows[i]["height"])))
    select("multiple_annotations", (i for i, p in enumerate(plans) if p["accepted"] and len(p["boxes"]) >= 4))
    receipts = []
    for case, (index, categories) in enumerate(selected.items(), 1):
        row = rows[index]
        source = ROOT / "data/prepared/grazpedwri_detection/images/train" / (row["filestem"]+".png")
        label = ROOT / "data/prepared/grazpedwri_detection/labels/train" / (row["filestem"]+".txt")
        original_hash, original_label_hash = sha256(source), sha256(label)
        # Confirm that the prototype starts from exactly the labels used by the prepared training dataset.
        prepared_labels = [list(map(float, line.split())) for line in label.read_text().splitlines()]
        if len(prepared_labels) != len(row["boxes"]):
            raise ValueError("Prepared label count differs from frozen source records")
        for encoded, box in zip(prepared_labels, row["boxes"]):
            class_id, cx, cy, width, height = encoded
            decoded = [(cx-width/2)*row["width"], (cy-height/2)*row["height"],
                       (cx+width/2)*row["width"], (cy+height/2)*row["height"]]
            if class_id != box["class_id"] or not np.allclose(decoded, box["xyxy"], rtol=0, atol=1e-8):
                raise ValueError("Prepared YOLO labels disagree with crop annotation inputs")
        with Image.open(source) as opened:
            if opened.mode != "L" or opened.size != (row["width"], row["height"]):
                raise ValueError("Prepared pixels differ from the frozen record")
            image = np.asarray(opened).copy()
        image.flags.writeable = False
        original = annotated(image, row["boxes"], f"Case {case:02}: full image; labels 0-7")
        panels = [original]
        private = {"case": case, "filestem": row["filestem"], "patient_id": row["patient_id"], "categories": categories, "plans": {}}
        for name, policy in specification["candidates"].items():
            class_weights = population.weights(policy)[0]
            plan = propose_crop(row, policy["crop"], specification["target_classes"], class_weights)
            pixels = crop_pixels(image, plan)
            left, top, right, bottom = plan["rect"]
            if not np.array_equal(pixels, image[top:bottom, left:right]):
                raise ValueError("Crop pixel selection changed source intensities")
            # Independently reconstruct every retained coordinate from the source and crop edges.
            for retained, source_index in zip(plan["boxes"], plan["source_indices"]):
                x1, y1, x2, y2 = row["boxes"][source_index]["xyxy"]
                expected = [max(x1, left)-left, max(y1, top)-top, min(x2, right)-left, min(y2, bottom)-top]
                if retained["xyxy"] != expected:
                    raise ValueError("Crop annotation round trip failed")
            panels.append(annotated(pixels, plan["boxes"], name+": "+plan["reason"]))
            private["plans"][name] = plan
            Image.fromarray(pixels).save(output / f"case{case:02}_{name}_pixels.png")
        sheet = Image.new("RGB", (2000, 650))
        for column, panel in enumerate(panels):
            sheet.paste(panel, (500*column, 0))
        sheet.save(output / f"case{case:02}_comparison.png")
        if (original_hash, original_label_hash) != (sha256(source), sha256(label)):
            raise ValueError("Source image or label was modified")
        receipts.append(private)
    save_json(output / "cases.json", receipts)
    result = {"status": "technical pixel and annotation checks passed; visual review pending",
              "cases": len(receipts), "comparisons": 3*len(receipts), "scope": "prepared training images only",
              "categories": dict(Counter(category for r in receipts for category in r["categories"])),
              "policies": {name: dict(Counter(r["plans"][name]["reason"] for r in receipts)) for name in specification["candidates"]},
              "source_images_and_labels_unchanged": True, "new_dataset_files_written": False,
              "holdout_images_opened": 0, "model_training_performed": False}
    save_json(PUBLIC / "prototype_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
