# COVID / non-COVID chest X-ray, fcm: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, XRAY_DIR, FIRST_ID, END_ID.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(RUN_SEED) 
import tensorflow as tf
tf.random.set_seed(RUN_SEED)
# ---------------------------
# Helpers
# ---------------------------

# ---------------------------
# Plain FCM for 1D arrays
# data: (N,) or (N,1)
# ---------------------------

def fcm1d(data, C=2, m=2.0, max_iter=40, tol=1e-6, seed=None):
    """
    Plain fuzzy C-means on 1D (or 1-feature) data.
    Returns: centers (C,1), memberships (C,N), labels (N,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure (N,1)
    data = np.asarray(data).reshape(-1, 1).astype(np.float64)  # (N,1)
    N, F = data.shape  # F should be 1 for 1D

    # Random init centers within data range
    dmin, dmax = data.min(axis=0), data.max(axis=0)     # (1,)
    centers = dmin + (dmax - dmin) * np.random.rand(C, F)

    # Random memberships then normalize columns
    U = np.random.rand(C, N)
    U /= (U.sum(axis=0, keepdims=True) + 1e-12)

    prev_loss = None

    for _ in range(max_iter):
        # Distances: (C,N)
        diff = centers[:, None, :] - data[None, :, :]      # (C,N,1)
        dist2 = np.sum(diff * diff, axis=2)                # (C,N)
        dist2 = np.maximum(dist2, 1e-12)                   # avoid zeros

        # Update memberships (standard FCM formula)
        # u_ik = 1 / sum_l ( (d_ik/d_lk) ** (1/(m-1)) )
        exp = 1.0 / (m - 1.0)
        ratio = (dist2[:, None, :] / dist2[None, :, :])**exp   # (C,C,N)
        U = 1.0 / np.sum(ratio, axis=1)                        # (C,N)

        # Update centers
        Um = U**m                                              # (C,N)
        numer = Um @ data                                      # (C,1)
        denom = Um.sum(axis=1, keepdims=True) + 1e-12          # (C,1)
        centers = numer / denom                                 # (C,1)

        # Optional: simple loss to check convergence
        loss = np.sum((U**m) * dist2)
        if prev_loss is not None and abs(loss - prev_loss) < tol:
            break
        prev_loss = loss

    labels = np.argmax(U, axis=0)  # (N,)
    return centers, U, labels

# ---------------------------
# Simple k-means for 1D arrays
# ---------------------------

# ---------------------------
# Driver: compare FCM vs K-means on a single image
# ---------------------------

r1 = None   # needed by the default argument of run_compare

def run_compare(image_path, mask_path, resize=(150,150), seed=42, r=r1):
    # Load image & GT
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(image_path)
    gt  = cv2.imread(mask_path,  cv2.IMREAD_GRAYSCALE)
    if gt is None: raise FileNotFoundError(mask_path)
    
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
        gt  = cv2.resize(gt,  resize, interpolation=cv2.INTER_NEAREST)
    if resize is None:
        resize=img.shape
    # Flatten to 1D feature vector and normalize
    data = img.reshape(-1)   # (N,)

    # ---- FCM (1D) ----
    _, U, fcm_labels = fcm1d(data, C=2, m=2.0, max_iter=40, tol=1e-6, seed=seed)
    fcm_mask0 = (fcm_labels == 0).astype(np.uint8).reshape(img.shape)
    fcm_mask1 = (fcm_labels == 1).astype(np.uint8).reshape(img.shape)

    GT=np.where(gt >= 1,1,0)

    fcm_iou0 = iou_score(fcm_mask0, GT)
    fcm_iou1 = iou_score(fcm_mask1, GT)
    iou=(fcm_iou0,fcm_iou1)
    

        
    index=np.argmax(iou)
    result=(fcm_mask0,fcm_mask1)

    results = compute_metrics(result[index], GT)
    dice = dice_coefficient(result[index], GT)
    mse, rmse = mse_and_rmse(result[index], GT)
    psnr_value = psnr(result[index] * 255, GT * 255)
    r1['IoU'].append(iou[index])
    r1['Dice'].append(dice)
    r1['MSE'].append(mse)
    r1['RMSE'].append(rmse)
    r1['PSNR'].append(psnr_value)
    r1['accuracy'].append(results['accuracy'])
    r1['precision'].append(results['precision'])
    r1['recall'].append(results['recall'])
    r1['F1'].append(results['F1_score'])
    r1['confusion_matrix'].append(results['confusion_matrix'])

r1= { 'IoU' :[],
      'Dice':[],
      'MSE':[],
      'RMSE':[],
      'PSNR':[],
      'accuracy':[],
      'precision':[],
      'recall':[],
      'F1':[],
      'confusion_matrix':[]
      
}
for i in range(FIRST_ID,END_ID,1):
    run_compare(XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp",\
                XRAY_DIR+"masks/CXR_Mask_" + str(i)+".bmp",\
                resize=(150,150), seed=RUN_SEED, r=r1)
    
for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
    r1[v].append(sum(r1[v])/len(r1[v]))

r1['confusion_matrix'].append('Null')
print("*"*20+"RESULT 1"+"*"*20)
df1 = pd.DataFrame(r1)
df1.to_csv("./1D_NON_COVID/FCM_COVID.csv")
