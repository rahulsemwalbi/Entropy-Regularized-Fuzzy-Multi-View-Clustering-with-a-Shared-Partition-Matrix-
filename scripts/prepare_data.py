#!/usr/bin/env python
"""Extract the original images and ground-truth masks from the three dataset zips into data/ and check them.

  python scripts/prepare_data.py --zips /path/to/folder/with/the/zips

Extracted (original images only -- the RSUA npy_images / npy_masks arrays and the Non_Covid_Pneumonia subset are
NOT used or extracted):
  data/ISBI2016_ISIC_Part1_Training_Data/ISIC_<id>.jpg                     (900)
  data/ISBI2016_ISIC_Part1_Training_GroundTruth/ISIC_<id>_Segmentation.png (900)
  data/Data Chest X-Ray RSUA (Validated)/{Covid,Non_Covid}/{images,masks}/*.bmp
Prints the image / mask pairing per subset (pairs are matched by ID).
"""
import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIPS = {"ISBI2016_ISIC_Part1_Training_Data.zip": r"\.jpg$",
        "ISBI2016_ISIC_Part1_Training_GroundTruth.zip": r"\.png$",
        "Data Chest X-Ray RSUA (Validated)-20230618T025557Z-001.zip": r"/(Covid|Non_Covid)/(images|masks)/[^/]+\.bmp$"}


def extract(zips_dir: Path, data: Path):
    for name, pattern in ZIPS.items():
        z = zipfile.ZipFile(zips_dir / name)
        members = [m for m in z.namelist() if re.search(pattern, m)]
        for m in members:
            if ".." in Path(m).parts or m.startswith("/"):
                raise ValueError(f"unsafe path in {name}: {m}")
        z.extractall(data, members)
        print(f"{name}: extracted {len(members)} files")


def check(data: Path):
    img = {p.stem for p in (data / "ISBI2016_ISIC_Part1_Training_Data").glob("*.jpg")}
    gt = {p.stem.replace("_Segmentation", "") for p in (data / "ISBI2016_ISIC_Part1_Training_GroundTruth").glob("*.png")}
    print(f"ISIC: {len(img)} images, {len(gt)} masks, {len(img & gt)} matched by ID")
    rsua = data / "Data Chest X-Ray RSUA (Validated)"
    for sub in ("Covid", "Non_Covid"):
        ids_i = {int(re.search(r"(\d+)$", p.stem).group(1)) for p in (rsua / sub / "images").glob("*.bmp")}
        ids_m = {int(re.search(r"(\d+)$", p.stem).group(1)) for p in (rsua / sub / "masks").glob("*.bmp")}
        rng = f"IDs {min(ids_i)}-{max(ids_i)}" if ids_i else ""
        print(f"RSUA {sub:20s}: {len(ids_i)} images, {len(ids_m)} masks, {len(ids_i & ids_m)} matched {rng}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True, help="folder containing the three dataset zip files")
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()
    Path(a.data).mkdir(parents=True, exist_ok=True)
    extract(Path(a.zips), Path(a.data))
    check(Path(a.data))
