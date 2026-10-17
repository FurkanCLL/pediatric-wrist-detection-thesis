"""Trace unchanged-class AP differences without changing the frozen evaluator."""

import inspect
import json

import numpy as np
from ultralytics.utils.metrics import ap_per_class

from wrist_eval.common import CLASSES, identity, sha
from wrist_eval.analyze import truth
from wrist_eval.metrics import native_stats
from wrist_inference_comparison.analysis import inputs
from wrist_inference_comparison.common import (ROOT, PUBLIC, read, write_new, require, study_path,
                                               install_access_guard)


def check(directory):
    rows, predictions, _, _ = inputs(directory)
    path = directory/"full_F/bundles.jsonl"
    require(sha(path) == read(directory/"full_F/summary.json")["bundles_sha256"], "F bundle changed")
    bundles = [json.loads(line) for line in path.read_text().splitlines()]
    require([v["stem"] for v in bundles] == [r["filestem"] for r in rows], "Bundle order changed")
    collected = {c: {key: [] for key in ("id", "conf", "class", "tp")} for c in ("D", "F")}
    offset = 0
    for row, bundle in zip(rows, bundles):
        d, f = (predictions[c][row["filestem"]] for c in ("D", "F"))
        ids = np.asarray(bundle["variants"]["V2"]["diagnostics"]["output_global_ids"])
        protected = ~np.isin(d[:, 5], (0, 3))
        # Restore original global IDs before comparing both raw rows and native TP flags.
        require(np.array_equal(d[protected], f[np.argsort(ids)][protected]), "Protected class predictions changed")
        stats = {"D": native_stats(truth(row), d), "F": native_stats(truth(row), f)}
        require(np.array_equal(stats["D"]["tp"][protected], stats["F"]["tp"][np.argsort(ids)][protected]),
                "Protected-class native matching changed")
        for c, values, global_ids in (("D", d, np.arange(len(d))), ("F", f, ids)):
            for key, value in (("id", global_ids+offset), ("conf", values[:, 4]),
                               ("class", values[:, 5]), ("tp", stats[c]["tp"])):
                collected[c][key].append(value)
        offset += len(d)
    flat = {c: {key: np.concatenate(values) for key, values in data.items()} for c, data in collected.items()}
    # This is exactly the pinned evaluator's global sort; no alternative AP is calculated.
    ordered = {c: {key: values[np.argsort(-data["conf"])] for key, values in data.items()} for c, data in flat.items()}
    report = {}
    metrics = read(directory/"analysis/comparison.json")["metrics"]
    for index in (1, 2, 4, 5, 6, 7):
        name = CLASSES[index]
        a = {key: value[ordered["D"]["class"] == index] for key, value in ordered["D"].items()}
        b = {key: value[ordered["F"]["class"] == index] for key, value in ordered["F"].items()}
        require(np.array_equal(a["conf"], b["conf"]), "Unchanged-class sorted confidences differ")
        d_lookup = {int(i): j for j, i in enumerate(a["id"])}
        require(all(a["conf"][d_lookup[int(i)]] == b["conf"][j] and
            np.array_equal(a["tp"][d_lookup[int(i)]], b["tp"][j]) for j, i in enumerate(b["id"])),
            "Equal-confidence permutation does not explain unchanged-class ranking")
        report[name] = {"prediction_rows": len(a["id"]), "all_image_rows_and_native_TP_flags_identical": True,
            "sorted_confidences_bitwise_equal": a["conf"].tobytes() == b["conf"].tobytes(),
            "reordered_equal_confidence_positions": int(np.sum(a["id"] != b["id"])),
            "ranked_TP_flag_entries_differing_across_10_IoUs": int(np.sum(a["tp"] != b["tp"])),
            "reported_AP_delta_F_minus_D": {key: metrics["F"]["classes"][name][key]-metrics["D"]["classes"][name][key]
                for key in ("ap50", "ap75", "ap50_95")}}
    result = {"passed": True, "images": 3088, "unchanged_classes": report,
        "cause_verified": "Pinned ap_per_class uses np.argsort(-conf), default unstable ordering. Identical rows/TP flags receive different within-equal-confidence rankings after global row reordering.",
        "evaluator_source_sha256": identity(inspect.getsource(ap_per_class)),
        "diagnostic_script_sha256": sha(ROOT/"docs/evidence/rarity_inference_comparison/check_ap_ties.py"),
        "no_evaluator_or_predictions_changed": True, "no_alternative_AP_or_method_selection": True}
    write_new(directory/"analysis/ap_tie_diagnostic.json", result)
    write_new(PUBLIC/"ap_tie_diagnostic.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    install_access_guard()
    check(study_path("study_01"))
