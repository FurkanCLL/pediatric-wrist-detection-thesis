"""Associate released image stems with explicit metadata identifiers."""

from collections import Counter, defaultdict
import csv
import math
import re


FILENAME = re.compile(r"(?P<patient>\d{4})_(?P<time>\d+)_(?P<study>\d+)_WRI-(?P<side>[LR])(?P<projection>[123])_(?P<gender>[FMO])(?P<age>\d+)$")


def filename_disagreements(row):
    match = FILENAME.fullmatch(row["filestem"])
    if match is None:
        return ["filename_pattern"]
    issues = []
    # Compare numeric identifiers numerically so harmless zero padding is preserved.
    for field, part in (("patient_id", "patient"), ("study_number", "study"),
                        ("timehash", "time")):
        try:
            if int(row[field]) != int(match[part]):
                issues.append(field)
        except (ValueError, TypeError):
            issues.append(field)
    # CSV ages include fractional years; the filename records completed years.
    try:
        age = float(row["age"])
        if not math.isfinite(age) or age < 0 or int(age) != int(match["age"]):
            issues.append("age")
    except (ValueError, TypeError, OverflowError):
        issues.append("age")
    for field, part in (("laterality", "side"), ("projection", "projection"), ("gender", "gender")):
        if row[field] != match[part]:
            issues.append(field)
    return issues


def read_metadata(path):
    # The released CSV begins with a UTF-8 BOM.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"filestem", "patient_id", "study_number", "timehash", "gender", "age", "laterality", "projection"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Missing required metadata columns or empty metadata")
    by_stem, problems = {}, []
    patients, studies, times = set(), set(), set()
    study_times, time_studies, genders = defaultdict(set), defaultdict(set), defaultdict(set)
    study_rows = defaultdict(list)
    for row in rows:
        stem = row["filestem"]
        if stem in by_stem:
            problems.append((stem, "duplicate_filestem"))
        by_stem[stem] = row
        if not row["patient_id"].strip() or not row["study_number"].strip():
            problems.append((stem, "missing_identity"))
        problems.extend((stem, "filename_" + issue) for issue in filename_disagreements(row))
        patient = row["patient_id"]
        study = (patient, row["study_number"])
        timestamp = (patient, row["timehash"])
        patients.add(patient)
        studies.add(study)
        times.add(timestamp)
        study_times[study].add(row["timehash"])
        study_rows[study].append(row)
        time_studies[timestamp].add(row["study_number"])
        genders[patient].add(row["gender"])
    summary = {
        "rows": len(rows), "unique_filestems": len(by_stem),
        "exact_duplicate_rows": len(rows) - len({tuple(r.items()) for r in rows}),
        "patients": len(patients), "patient_study_groups": len(studies),
        "patient_timehash_groups": len(times),
        "studies_with_multiple_timehashes": sum(len(x) > 1 for x in study_times.values()),
        "timehashes_with_multiple_studies": sum(len(x) > 1 for x in time_studies.values()),
        "patients_with_inconsistent_gender": sum(len(x) > 1 for x in genders.values()),
        "gender_counts": dict(Counter(r["gender"] for r in rows)),
        "age_min_years": min(float(r["age"]) for r in rows),
        "age_max_years": max(float(r["age"]) for r in rows),
        "groups_with_both_lateralities": sum(len({r["laterality"] for r in group}) > 1 for group in study_rows.values()),
        "groups_with_repeated_side_projection": sum(len({(r["laterality"], r["projection"]) for r in group}) < len(group) for group in study_rows.values()),
        "blank_fields": {k: sum(not r[k].strip() for r in rows) for k in rows[0]},
        "issue_counts": dict(Counter(issue for _, issue in problems)),
        "images_per_patient": list(Counter(r["patient_id"] for r in rows).values()),
        "images_per_study": list(Counter((r["patient_id"], r["study_number"]) for r in rows).values()),
    }
    return by_stem, summary, problems
