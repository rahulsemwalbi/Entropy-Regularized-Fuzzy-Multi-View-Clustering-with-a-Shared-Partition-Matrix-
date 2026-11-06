"""Dermatology views of the four proposed methods ("rgb", "lab", "morph" = 9 views).

Input: features.image of ImageFeature (hair removal + skin repaint, 200 x 200, BGR uint8).
Views (all uint8 0-255, NOT normalised, as for the other proposed-method inputs):
  "rgb"   : R, G, B
  "lab"   : L, a, b  (OpenCV CIELAB for uint8: L, a + 128, b + 128 scaled to 0-255)
  "morph" : on the grey image, the morphology views of the X-ray code with the same parameters:
            black-hat (elliptical 60 x 60), morphological gradient (dilation - erosion, elliptical r = 3),
            external gradient (dilation - image, r = 3)
"""
import cv2
import numpy as np

KINDS = ("rgb", "lab", "morph")


def derma_views(image_bgr, kinds=("lab", "morph")):
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown view kinds {sorted(unknown)}; choose from {KINDS}")
    views = []
    if "rgb" in kinds:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        views += [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]
    if "lab" in kinds:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        views += [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]
    if "morph" in kinds:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        r = 3
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        dil, ero = cv2.dilate(gray, k), cv2.erode(gray, k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)))
        views += [blackhat, cv2.subtract(dil, ero), cv2.subtract(dil, gray)]
    return np.array(views)
