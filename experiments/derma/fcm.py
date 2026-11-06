# ISIC dermatology, fcm: segments every image of `images` and writes the CSV of per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, IMAGE_DIR, images, directory_path_test.

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
np.random.seed(RUN_SEED)
def initialize_centers(data, C):
    #idx = np.random.choice(len(data), C, replace=False)
    #return data[idx]
    #np.random.randint(1, size=(2,3)) + 5e-2
    rng = np.random.default_rng(RUN_SEED)
    v = data.min(axis=0) + (data.max(axis=0) - data.min(axis=0)) * rng.random(size=(C, data.shape[1]))
    return v
    

def update_membership(data, centers, m):
    # Expand data and centers to form matrices for vectorized computation
    data_expanded = np.expand_dims(data, 1)  # Shape (N, 1, 3)
    centers_expanded = np.expand_dims(centers, 0)  # Shape (1, C, 3)
    
    # Calculate pairwise Euclidean distances using broadcasting
    distances = np.linalg.norm(data_expanded - centers_expanded, axis=2)  # Shape (N, C)
    #distances = np.maximum(distances, np.finfo(np.float64).eps)  # Avoid division by zero
    distances = np.maximum(distances, 1e-5)
    # Calculate inverse distances raised to the power for fuzzy membership
    inv_dist_power = 1.0 / distances ** (2 / (m - 1))
    
    # Normalize to get memberships
    membership = inv_dist_power / np.sum(inv_dist_power, axis=1, keepdims=True)

    return membership.T  # Transpose to match expected shape (C, N)

def update_centers(data, membership, m):
    # Weighted sum of points
    um = membership ** m
    numerator = np.dot(um, data)
    denominator = um.sum(axis=1, keepdims=True)
    
    return numerator / denominator

def fuzzy_c_means(image_path, C, m=2, max_iter=40, error=1e-5):
    # Load image and normalize pixel values
    features = ImageFeature(image_path,(200,200))
    image_rgb,hsv_image=features.convert_to_rgb_hsv()
    image = image_rgb
    #image = img_as_float(io.imread(image_path))
    data =np.reshape(image, (-1, 3))  # Flatten image to a list of pixels

    # Initialize cluster centers
    centers = initialize_centers(data, C)
    u=None
    for iteration in range(max_iter):
        # Update memberships
        u = update_membership(data, centers, m)

        # Update centers
        new_centers = update_centers(data, u, m)

        # Check for convergence
        if np.linalg.norm(new_centers - centers) < error:
            #centers = new_centers
            break
        centers = new_centers


    final_result = np.argmax(u, axis=0)   # shape = (height, width)
    final_result=final_result.reshape((200, 200))
    result1 = (final_result == 0).astype(np.uint8)
    result2 = (final_result == 1).astype(np.uint8)
    #result1=result1.reshape(image.shape[0],image.shape[1])#np.reshape(u[0], (image.shape[0],image.shape[1]))
    #result2=result2.reshape(image.shape[0],image.shape[1])#np.reshape(u[1], (image.shape[0],image.shape[1]))
    return (result1,result2)

# # Example usage
# image_path = 'path_to_your_image.jpg'  # Path to the image file
# C = 3  # Number of clusters
# centers, membership = fuzzy_c_means(image_path, C)

import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os



r1= { 'id':[],
      'IoU' :[],
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

np.random.seed(RUN_SEED)
for i in range(len(images)):
    if i>=0 and i<=143:
        image_path = os.path.join(IMAGE_DIR, images[i] + '.jpg')
    elif i>=144 and i<=756:
        image_path = os.path.join(IMAGE_DIR, images[i] + '.jpg')
    else:
        image_path = os.path.join(IMAGE_DIR, images[i] + '.jpg')
    
    result1,result2=fuzzy_c_means(image_path, 2)
  
    img=cv2.imread(os.path.join(directory_path_test, images[i] + "_Segmentation.png"))
    img=cv2.resize(img, (200, 200),interpolation=cv2.INTER_NEAREST)
    GT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    GT=np.where(GT >= 1,1,0)
    if result1.shape == GT.shape:   
        iou1 = iou_score(result1, GT)
        iou2 = iou_score(result2, GT)
        if iou1 > iou2:
            results = compute_metrics(result1, GT)
            dice = dice_coefficient(result1, GT)
            mse, rmse = mse_and_rmse(result1, GT)
            psnr_value = psnr(result1 * 255, GT * 255)
            r1['id'].append(images[i])
            r1['IoU'].append(iou1)
            r1['Dice'].append(dice)
            r1['MSE'].append(mse)
            r1['RMSE'].append(rmse)
            r1['PSNR'].append(psnr_value)
            r1['accuracy'].append(results['accuracy'])
            r1['precision'].append(results['precision'])
            r1['recall'].append(results['recall'])
            r1['F1'].append(results['F1_score'])
            r1['confusion_matrix'].append(results['confusion_matrix'])
        else:
            results = compute_metrics(result2, GT)
            dice = dice_coefficient(result2, GT)
            mse, rmse = mse_and_rmse(result2, GT)
            psnr_value = psnr(result2 * 255, GT * 255)
            r1['id'].append(images[i])
            r1['IoU'].append(iou2)
            r1['Dice'].append(dice)
            r1['MSE'].append(mse)
            r1['RMSE'].append(rmse)
            r1['PSNR'].append(psnr_value)
            r1['accuracy'].append(results['accuracy'])
            r1['precision'].append(results['precision'])
            r1['recall'].append(results['recall'])
            r1['F1'].append(results['F1_score'])
            r1['confusion_matrix'].append(results['confusion_matrix'])
        
          
for v in [ 'id', 'IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
    if v=='id':
        r1[v].append('mean')
    else:
        r1[v].append(sum(r1[v])/len(r1[v]))

r1['confusion_matrix'].append('Null')
print("*"*20+"RESULT 1"+"*"*20)
df1 = pd.DataFrame(r1)
df1.to_csv("FINAL_result_FCM_900.csv")
