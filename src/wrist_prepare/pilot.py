"""Deterministic training-only pilot selection and private review sheets."""

import json
import math

import numpy as np
from PIL import Image, ImageDraw

from .common import CLASSES, digest


def select_pilot(records, audit, count=16):
    training = sorted((r for r in records if r["split"] == "train"), key=lambda r: digest(("42:" + r["filestem"]).encode()))
    selected = {}
    def add(record, reason):
        selected.setdefault(record["filestem"], {"record": record, "reasons": []})["reasons"].append(reason)
    for c, name in enumerate(CLASSES):
        add(next(r for r in training if any(o["class_id"] == c for o in r["boxes"])), name)
    add(next(r for r in training if not r["boxes"]), "empty target labels")
    add(min(training, key=lambda r: max(r["width"], r["height"])), "small dimensions")
    add(max(training, key=lambda r: max(r["width"], r["height"])), "large dimensions")
    add(min(training, key=lambda r: r["width"] / r["height"]), "narrow aspect")
    add(max(training, key=lambda r: r["width"] / r["height"]), "wide aspect")
    samples = json.loads((audit / "pixel_sample.json").read_text())
    by_stem = {r["filestem"]: r for r in training}
    candidates = [r for r in samples if r["filestem"] in by_stem]
    for key, operation, reason in (("mean", min, "low mean intensity"), ("mean", max, "high mean intensity"),
                                   ("zero_fraction", max, "substantial black background")):
        sample = operation(candidates, key=lambda r: r["pixels"][key])
        add(by_stem[sample["filestem"]], reason)
    for record in training:
        if len(selected) >= count:
            break
        add(record, "deterministic coverage")
    return list(selected.values())


def review_sheets(selection, raw, pilot, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for page in range(math.ceil(len(selection) / 4)):
        sheet = Image.new("RGB", (1200, 2200), "#f0f0f0")
        draw = ImageDraw.Draw(sheet)
        for row, entry in enumerate(selection[page * 4:(page + 1) * 4]):
            record = entry["record"]
            stem, split = record["filestem"], record["split"]
            with Image.open(raw / "images" / (stem + ".png")) as source:
                # This high-byte preview is only for display; it is never a detector input.
                original_preview = Image.fromarray((np.asarray(source).astype(np.uint32) // 256).astype(np.uint8))
            with Image.open(pilot / "images" / split / (stem + ".png")) as prepared:
                converted = prepared.copy()
            for column, image in enumerate((original_preview, converted)):
                factor = min(560 / image.width, 495 / image.height)
                shown = image.convert("RGB").resize((round(image.width * factor), round(image.height * factor)))
                overlay = ImageDraw.Draw(shown)
                for obj in record["boxes"]:
                    overlay.rectangle(tuple(v * factor for v in obj["xyxy"]), outline="#e73f3f", width=1)
                x, y = column * 600 + (600 - shown.width) // 2, row * 550 + 45
                sheet.paste(shown, (x, y))
                title = f"Pilot {page * 4 + row + 1}: " + ("source high-byte preview" if column == 0 else "prepared 8-bit PNG")
                draw.text((column * 600 + 12, row * 550 + 8), title, fill="black")
                draw.text((column * 600 + 12, row * 550 + 25), ", ".join(entry["reasons"]), fill="black")
        path = output / f"pilot_review_{page + 1:02d}.png"
        sheet.save(path)
        paths.append(path)
    return paths
