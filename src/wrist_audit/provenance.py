"""Verify the official release metadata and annotation archive without extraction."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

from .audit import validate_output_path, write_json


RELEASE_API = "https://api.figshare.com/v2/articles/14825193/versions/2"


def check_archive(raw, archive, expected_md5):
    data = archive.read_bytes()
    actual = hashlib.md5(data).hexdigest()
    if actual != expected_md5:
        raise ValueError("Downloaded annotation archive does not match the official MD5")
    counts, mismatches = Counter(), []
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if ".." in parts or PurePosixPath(member.filename).is_absolute() or any(":" in part or "\\" in part for part in parts):
                raise ValueError("Unsafe archive member name")
            local = raw.joinpath(*parts)
            # Read archive members in memory; never extract over the user's source.
            content = source.read(member)
            counts["archive_files"] += 1
            if not local.is_file():
                counts["missing_local"] += 1
                mismatches.append({"path": member.filename, "issue": "missing_local"})
            elif hashlib.sha256(content).digest() != hashlib.sha256(local.read_bytes()).digest():
                counts["different_local"] += 1
                mismatches.append({"path": member.filename, "issue": "different_content"})
            else:
                counts["identical_local"] += 1
    return {"md5": actual, "official_md5_match": True, "counts": dict(counts)}, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/grazpedwri-dx"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--fetch", action="store_true", help="Fetch pinned v2 metadata and the 26 MB annotation archive")
    args = parser.parse_args()
    validate_output_path(args.raw, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "figshare_metadata.json"
    archive_path = args.output / "folder_structure.zip"
    if args.fetch:
        with urllib.request.urlopen(RELEASE_API, timeout=60) as response:
            metadata = json.load(response)
        write_json(metadata_path, metadata)
        archive_file = next(f for f in metadata["files"] if f["name"] == archive_path.name)
        with urllib.request.urlopen(archive_file["download_url"], timeout=60) as response:
            archive_path.write_bytes(response.read())
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["id"] != 14825193 or metadata["version"] != 2:
        raise ValueError("Expected the pinned GRAZPEDWRI-DX v2 release")
    files = {f["name"]: f for f in metadata["files"]}
    csv_bytes = (args.raw / "dataset.csv").read_bytes()
    csv_matches = hashlib.md5(csv_bytes).hexdigest() == files["dataset.csv"]["computed_md5"]
    archive_summary, mismatches = check_archive(args.raw, archive_path, files["folder_structure.zip"]["computed_md5"])
    summary = {
        "release_api": RELEASE_API, "doi": metadata["doi"], "version": metadata["version"],
        "title": metadata["title"], "license": metadata["license"],
        "published_date": metadata["published_date"],
        "official_files": [{k: f[k] for k in ("name", "size", "computed_md5")} for f in metadata["files"]],
        "dataset_csv_official_md5_match": csv_matches,
        "folder_archive": archive_summary,
        "image_archive_checksums_verified": False,
        "image_archive_limitation": "Original image ZIPs were not supplied to this workflow; no full re-download was performed.",
    }
    write_json(args.output / "source_release.json", summary)
    write_json(args.output / "archive_mismatches.json", mismatches)
    print(json.dumps(summary, indent=2))
    return 0 if csv_matches and not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
