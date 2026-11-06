"""Synthetic stand-ins for the datasets (the real images are never used by the tests)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ISIC_IDS = ["ISIC_9000001", "ISIC_9000002", "ISIC_9000003"]


def _skin_image(rng, size=(240, 260)):
    h, w = size
    img = np.empty((h, w, 3), np.float64)
    img[:] = (120, 150, 210)                                    # BGR skin tone (inside the YCrCb skin range)
    yy, xx = np.mgrid[:h, :w]
    cy, cx = rng.uniform(0.35, 0.65) * h, rng.uniform(0.35, 0.65) * w
    lesion = ((yy - cy) / (0.22 * h)) ** 2 + ((xx - cx) / (0.18 * w)) ** 2 < 1
    img[lesion] = (40, 60, 110)                                 # dark lesion
    img += rng.normal(0, 8, img.shape)
    for _ in range(4):                                          # a few thin dark "hairs"
        r0, c0 = rng.integers(0, h), rng.integers(0, w)
        cv2.line(img, (int(c0), int(r0)), (int(c0 + 60), int(r0 + 40)), (20, 20, 20), 1)
    return np.clip(img, 0, 255).astype(np.uint8), (lesion * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def isic(tmp_path_factory):
    root = tmp_path_factory.mktemp("isic")
    img_dir, gt_dir = root / "ISBI2016_ISIC_Part1_Training_Data", root / "ISBI2016_ISIC_Part1_Training_GroundTruth"
    img_dir.mkdir(); gt_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in ISIC_IDS:
        img, mask = _skin_image(rng)
        cv2.imwrite(str(img_dir / f"{i}.jpg"), img)
        cv2.imwrite(str(gt_dir / f"{i}_Segmentation.png"), mask)
    lst = root / "ids.txt"
    lst.write_text("\n".join(ISIC_IDS) + "\n")
    return {"root": root, "images": img_dir, "masks": gt_dir, "list": lst}


def _chest(rng, size=(96, 96)):
    h, w = size
    img = np.full((h, w), 190.0)
    yy, xx = np.mgrid[:h, :w]
    lungs = np.zeros((h, w), bool)
    for cx in (0.3, 0.7):
        lungs |= ((yy - 0.5 * h) / (0.32 * h)) ** 2 + ((xx - cx * w) / (0.15 * w)) ** 2 < 1
    img[lungs] = 70
    img += rng.normal(0, 10, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), (lungs * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def rsua(tmp_path_factory):
    """Non_Covid subset only (IDs 208-239)."""
    root = tmp_path_factory.mktemp("rsua") / "Data Chest X-Ray RSUA (Validated)"
    rng = np.random.default_rng(1)
    for sub, ids in (("Non_Covid", range(208, 240)),):
        (root / sub / "images").mkdir(parents=True); (root / sub / "masks").mkdir(parents=True)
        for i in ids:
            img, mask = _chest(rng)
            cv2.imwrite(str(root / sub / "images" / f"CXR_Images_{i}.bmp"), img)
            cv2.imwrite(str(root / sub / "masks" / f"CXR_Mask_{i}.bmp"), mask)
    return root
