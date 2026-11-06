# ISIC dermatology: set-up code executed before every method (imports, normalisation helper, ImageFeature,
# plotting helper, evaluation measures).

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.util import view_as_windows
from scipy.stats import entropy
import os
import pandas as pd#from google.colab.patches import cv2_imshow

def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalise an array to the range [0, 1]."""
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

class ImageFeature:
    def __init__(self, image_path,resize_par):

        self.image_path = image_path
        self.hair_removed_image=None
        self.img_org_shape=None
        self.resize=resize_par
        self.hair_removel()
        self.img_PreProcess()
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    def img_PreProcess(self):

      img_YCrCb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
      lower_bound = np.array([0, 135, 85])
      upper_bound = np.array([255, 180, 135])
      YCrCb_mask = cv2.inRange(img_YCrCb, lower_bound, upper_bound)
      kernel = np.ones((3, 3), np.uint8)
      YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, kernel)
      skin_pixels = self.image[YCrCb_mask == 255]
      average_skin_color = np.median(skin_pixels, axis=0).astype(int)
      non_skin_mask = cv2.bitwise_not(YCrCb_mask)
      repainted_img = self.image.copy()
      non_skin_indexes = np.where(non_skin_mask == 255)
      repainted_img[non_skin_indexes[0], non_skin_indexes[1]] = average_skin_color
      self.image=repainted_img



    def hair_removel(self):

      src = cv2.imread(self.image_path)
      if src is None:
            raise ValueError(f"Failed to load image: {image_path}")
 
      self.img_org_shape=src.shape
      if self.resize is not None:
          src= cv2.resize(src, self.resize, interpolation=cv2.INTER_NEAREST)#cv2.resize(src, self.resize)
      # Convert the original image to grayscale
      grayScale = cv2.cvtColor( src, cv2.COLOR_RGB2GRAY )

      kernel = cv2.getStructuringElement(1,(17,17))
      blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
      ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

      # inpaint the original image depending on the mask
      dst = cv2.inpaint(src,thresh2,1,cv2.INPAINT_TELEA)

      self.image=dst#dst
      self.hair_removed_image=dst


    def window_average_feature(self, window_shape=(2, 3)):
        return np.lib.stride_tricks.sliding_window_view(self.gray_image, window_shape).mean(axis=(-1, -2))

    def compute_lbp(self, P=8, R=1, method='default'):
        lbp=local_binary_pattern(self.gray_image, P, R, method)
        if self.non_skin_index != None:
          median=np.median(lbp[self.skin_index[0], self.skin_index[1]]).astype(int)
          lbp[self.non_skin_index[0], self.non_skin_index[1]] = median
        return lbp


    def convert_to_rgb_hsv(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        return normalize(image_rgb), normalize(hsv_image)

    def calculate_local_entropy(self, window_size=5):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        pad_size = window_size // 2
        padded_image = np.pad(self.gray_image, pad_size, mode='reflect')
        windows = view_as_windows(padded_image, (window_size, window_size))
        entropy_image = np.zeros(self.gray_image.shape)
        for i in range(entropy_image.shape[0]):
            for j in range(entropy_image.shape[1]):
                window = windows[i, j]
                hist, _ = np.histogram(window, bins=256, range=(0, 256), density=True)
                entropy_image[i, j] = entropy(hist, base=2)

        if self.non_skin_index !=None:
          median=np.median(entropy_image[self.skin_index[0], self.skin_index[1]]).astype(int)
          entropy_image[self.non_skin_index[0], self.non_skin_index[1]] = median
        return entropy_image

class ImagePlotting:
    @staticmethod
    def plot_image(image, title='Image', cmap='gray'):
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')

    @staticmethod
    def plot_window_average(gray_image, window_means):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ImagePlotting.plot_image(gray_image, 'Original Image')
        plt.subplot(1, 2, 2)
        ImagePlotting.plot_image(window_means, 'Window Mean Features', cmap='hot')
        plt.colorbar()
        plt.show()

    @staticmethod
    def plot_lbp(image, lbp_image):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ImagePlotting.plot_image(image, 'Original Image')
        plt.subplot(1, 2, 2)
        ImagePlotting.plot_image(lbp_image, 'LBP Image')
        plt.show()

    @staticmethod
    def rgb_hsv(image_rgb, hsv_image):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        ImagePlotting.plot_image(image_rgb, 'Original Image')
        plt.subplot(1, 2, 2)
        ImagePlotting.plot_image(hsv_image, 'HSV Image')
        plt.show()

    @staticmethod
    def plot_entropy(entropy_image):
        plt.imshow(entropy_image, cmap='hot')
        plt.colorbar()
        plt.title('Local Entropy')
        plt.show()

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

# def iou_score(preds, labels):
#     intersection = np.logical_and(preds, labels)
#     union = np.logical_or(preds, labels)
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score

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
