"""Retrieve/verify original NASA PCoE MATLAB files for the archived notebooks.

The public NASA archive contains nested ZIPs, not directly accessible CSV
substitutes. Extracts only requested .mat payloads to data/Bxxxx.mat;
the .mat files and source ZIP are intentionally excluded from Git history.

Use --archive to point to an already-downloaded (or Google Drive cached) ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from scipy.io import loadmat

NASA_ARCHIVE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_ARCHIVE_SHA256 = "82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e"
EXPECTED_DISCHARGE = {
    "B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132,
    "B0029": 40, "B0030": 40, "B0031": 40, "B0032": 40,
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".download")
    print(f"NASA original ZIP: {NASA_ARCHIVE_URL}", flush=True)
    try:
        request = urllib.request.Request(
            NASA_ARCHIVE_URL, headers={"User-Agent": "Battery-Reproduction/1.0"}
        )
        with urllib.request.urlopen(request, timeout=90) as source, tmp.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if file_hash(tmp) != NASA_ARCHIVE_SHA256:
            raise ValueError("Downloaded NASA archive SHA-256 differs from the version in the dataset manifest")
        os.replace(tmp, path)
        print("Downloaded archive SHA-256 verified", flush=True)
    finally:
        if tmp.exists():
            tmp.unlink()


def check_mat_bytes(blob: bytes, cell: str) -> tuple[int, int]:
    mat = loadmat(io.BytesIO(blob))
    if cell not in mat:
        raise ValueError(f"MATLAB file has no {cell} object")
    cycles = mat[cell][0, 0]["cycle"][0]
    discharges = 0
    samples = 0
    for i in range(len(cycles)):
        row = cycles[i]
        if row["type"][0] == "discharge":
            discharges += 1
            discharge = row["data"][0, 0]
            samples += len(discharge["Voltage_measured"][0])
    if discharges != EXPECTED_DISCHARGE[cell]:
        raise ValueError(f"{cell} has {discharges} discharges; expected {EXPECTED_DISCHARGE[cell]}")
    if samples == 0:
        raise ValueError(f"{cell} has zero discharge samples")
    return discharges, samples


def extract_archive(archive: Path, wanted: list[str], dest: Path) -> dict:
    extracted: dict[str, dict] = {}
    with zipfile.ZipFile(archive) as outer:
        for member in outer.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".zip"):
                continue
            if set(wanted).issubset(extracted):
                break
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as nested:
                for item in nested.infolist():
                    cell = Path(item.filename).stem.upper()
                    if item.is_dir() or not item.filename.lower().endswith(".mat"):
                        continue
                    if cell not in wanted or cell in extracted:
                        continue
                    blob = nested.read(item)
                    discharges, samples = check_mat_bytes(blob, cell)
                    target = dest / (cell + ".mat")
                    tmp = dest / ("." + cell + ".mat.tmp")
                    try:
                        tmp.write_bytes(blob)
                        os.replace(tmp, target)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
                    extracted[cell] = {
                        "file": target.name, "sha256": hashlib.sha256(blob).hexdigest(),
                        "discharges": discharges, "samples": samples,
                    }
                    print(f"Verified {cell}.mat: {discharges} discharge cycles, {samples} measurements", flush=True)
    missing = sorted(set(wanted).difference(extracted))
    if missing:
        raise RuntimeError(f"Required original MAT files not found in archive: {missing}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/5.+Battery+Data+Set.zip"))
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--cells", nargs="+", choices=sorted(EXPECTED_DISCHARGE),
                        default=sorted(EXPECTED_DISCHARGE))
    parser.add_argument("--manifest-out", type=Path, default=None,
                        help="Write provenance JSON to persistent Drive, not the public repo")
    args = parser.parse_args()
    wanted = sorted(set(args.cells))
    args.dest.mkdir(parents=True, exist_ok=True)

    # If the original .mat files survived a Colab restart, skip the large download.
    existing = {}
    for cell in wanted:
        path = args.dest / f"{cell}.mat"
        if not path.exists():
            continue
        blob = path.read_bytes()
        n, samples = check_mat_bytes(blob, cell)
        existing[cell] = {
            "file": path.name, "sha256": hashlib.sha256(blob).hexdigest(),
            "discharges": n, "samples": samples,
        }
        print(f"Existing verified {cell}: {n} cycles, {samples} samples")
    if len(existing) == len(wanted):
        info = existing
        print("All original MATLAB files already verified; archive download skipped")
    else:
        if not args.archive.is_file():
            try:
                download_archive(args.archive)
            except Exception as err:
                raise RuntimeError(
                    f"Automatic NASA download unavailable: {err}. Download the official "
                    f"archive manually, upload or place it on Drive, then provide --archive."
                ) from err
        checksum = file_hash(args.archive)
        if checksum != NASA_ARCHIVE_SHA256:
            raise ValueError(f"Original archive checksum mismatch: {checksum}")
        info = extract_archive(args.archive, wanted, args.dest)
    report = {
        "source_url": NASA_ARCHIVE_URL,
        "expected_source_zip_sha256": NASA_ARCHIVE_SHA256,
        "original_mat_files": info,
        "protocol": "Original sample-level NASA PCoE records used by legacy 2023 notebooks",
    }
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
