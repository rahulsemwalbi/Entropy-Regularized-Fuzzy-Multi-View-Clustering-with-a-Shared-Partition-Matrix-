# COVID / non-COVID chest X-ray: set-up code executed before every method (imports, evaluation measures,
# normalisation and IoU helpers).

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.util import view_as_windows
from scipy.stats import entropy
import os
import pandas as pd#from google.colab.patches import cv2_imshow
import pywt

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to the range [0, 1]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Intersection‑Over‑Union for two binary masks."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return (inter + eps) / (union + eps)

    
def dice_coefficient(preds, labels):
    intersection = np.sum(preds[labels==1])
    return 2. * intersection / (np.sum(preds) + np.sum(labels))


def mse_and_rmse(preds, labels):
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    return mse, rmse

def psnr(preds, labels, max_pixel=255.0):
    mse = np.mean((preds - labels) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))



  # Assuming pixel values are in 0-255

def compute_metrics(preds, labels):
    """
    Compute the accuracy, precision, recall, F1-score, and confusion matrix for the given predictions and ground truth labels.
    
    Args:
    preds (numpy.ndarray): Predicted binary image of shape (a, b).
    labels (numpy.ndarray): Ground truth binary image of shape (a, b).

    Returns:
    dict: Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix.
    """
    # Flatten the arrays for simplicity
    preds = preds.flatten()
    labels = labels.flatten()
    
    # True positives, false positives, true negatives, false negatives
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    # Metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confusion Matrix
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1_score": F1_score,
        "confusion_matrix": confusion_matrix
    }

"""
compare_mvkfcm_kmeans.py
========================

This module provides two functions for segmentation and comparison on a single
image: one using a 1‑D multi‑view kernel fuzzy C‑means (MVKFCM) and the
other using a simple k‑means.  It reads a grayscale image and its
ground‑truth mask, converts the relevant views into flattened 1‑D arrays,
performs clustering, computes Intersection‑Over‑Union (IoU) scores, and
optionally saves the resulting segmentation masks.

Usage example:

    python3 compare_mvkfcm_kmeans.py --image /path/to/CXR_Images_8.bmp \
                                     --mask /path/to/CXR_Mask_8.bmp \
                                     --save_output /home/oai/share/segmentation_results

This will run both MVKFCM and k‑means on the provided image, compute the
IoU of each segmentation with respect to the ground truth, and save the
segmentation masks to the specified directory.  It prints the IoU values
to stdout.

Note: This script assumes OpenCV (cv2), NumPy and Matplotlib are available
in the Python environment.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to the range [0, 1]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Intersection‑Over‑Union for two binary masks."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return (inter + eps) / (union + eps)
