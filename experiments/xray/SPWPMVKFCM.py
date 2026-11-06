# COVID / non-COVID chest X-ray, SPWPMVKFCM: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, RUN_LAMBDA, RUN_SIGMA, XRAY_DIR, FIRST_ID, END_ID and the model class
# WEIGHT_POINT_MVJFC (= SPWPMVKFCM of spmvfcm/methods.py).

#NO OUTLIER
import numpy as np
import cv2
from skimage import filters, feature, morphology, exposure
from scipy.ndimage import uniform_filter
from skimage.filters import gabor
from skimage import feature, morphology, measure, color
from skimage.segmentation import clear_border
import cv2
from scipy.ndimage import binary_fill_holes  # ← make sure this is here
import numpy as np
np.random.seed(RUN_SEED) 
def comput3(param_value,fuzzy_value,sigma_value):

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

                
        img = cv2.imread(XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)#cv2.resize(img, (150, 150))
        # 1. Intensity View (original)
        # view_raw =img
        # hist_eq = cv2.equalizeHist(img)
        # view_hist = hist_eq
    
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # view_clahe = clahe.apply(img)
        
        # # 4. Gamma Correction (try 0.5 and 2.0)
        # gamma1 = 0.5
        # gamma2 = 2.0
        # view_gamma1 = np.power(img/255.0, gamma1)
        # view_gamma1 = (view_gamma1 * 255).astype(np.uint8)
        
        # view_gamma2 = np.power(img/255.0, gamma2)
        # view_gamma2 = (view_gamma2 * 255).astype(np.uint8)
        
        # # 5. Negative Image
        # view_negative = (255 - img)
        
        # # -- Stack as channels as needed (pick 3 or more) --
        # # stacked_views = np.stack([view_raw, view_clahe, view_gamma1], axis=-1)  # Example
        
        # # --- Visualize ---
        # x= [view_raw, view_hist,view_clahe,view_gamma1,view_gamma2,view_negative]

        gamma2 = 2.2
        view_gamma2 = np.power(img/255.0, gamma2)
        view_gamma2 = (view_gamma2 * 255).astype(np.uint8)

        r = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        
        # Morphological gradients
        dil = cv2.dilate(img, kernel)
        ero = cv2.erode(img, kernel)
        morph_grad = cv2.subtract(dil, ero)        # edge-strength
        internal_grad = cv2.subtract(img, ero)    # inside edges
        external_grad = cv2.subtract(dil, img)      # outside edges
    
        #LL, (LH, HL, HH) = pywt.dwt2(img.astype(np.float32), 'haar')
       
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    
        # Black-hat transform
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
     
       
        x = [view_gamma2,blackhat, morph_grad,external_grad]

        x=np.array(x)
        shape_x = x.shape
        num_view = shape_x[0]
        C=2
        
        shape_u = (C,) + shape_x[1:]
        u = np.random.rand(*shape_u)
        u /= u.sum(axis=0, keepdims=True)
        
        x_init = x.reshape(len(x), -1).astype(float)
        v = x_init.min(axis=1)[:, None] + (x_init.max(axis=1) - x_init.min(axis=1))[:, None] * np.random.rand(len(x), C)
        
        
        w = np.full(x.shape, 1/len(x))        





        
    

    
        m=WEIGHT_POINT_MVJFC(x, w, v, u,C, iterations=40, fuzzy=fuzzy_value, sigma=sigma_value, param=param_value,div_error_corr=1e-10)
        m.optimize()
       
    

        final_result = np.argmax(m.u, axis=0)   # shape = (height, width)
        result1 = np.where(final_result == 0, 1, 0)
        result2 = np.where(final_result == 1, 1, 0)
      


    
        img=cv2.imread(XRAY_DIR+'masks/CXR_Mask_' + str(i)+'.bmp',cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)#cv2.resize(img, (150, 150))
       
     
        GT=np.where(img >= 1,1,0)
    
        if result1.shape == GT.shape:
            iou1 = iou_score(result1, GT)
            iou2 = iou_score(result2, GT)
            if iou1 > iou2:
                results = compute_metrics(result1, GT)
                dice = dice_coefficient(result1, GT)
                mse, rmse = mse_and_rmse(result1, GT)
                psnr_value = psnr(result1 * 255, GT * 255)
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
        
          
    for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
        r1[v].append(sum(r1[v])/len(r1[v]))

    r1['confusion_matrix'].append('Null')
    print("*"*20+"RESULT 1"+"*"*20)
    df1 = pd.DataFrame(r1)
    df1.to_csv("WEIGHT_POINT_MVJFC.csv")
comput3(RUN_LAMBDA, 2.0, RUN_SIGMA)
