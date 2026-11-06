# COVID / non-COVID chest X-ray, dec: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, XRAY_DIR, FIRST_ID, END_ID.

#NO OUTLIER

from clustpy.deep import DCN
from clustpy.data import create_subspace_data
from clustpy.deep import DipEncoder
from clustpy.deep import DEC
import numpy as np
np.random.seed(RUN_SEED) 
import tensorflow as tf
tf.random.set_seed(RUN_SEED)
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


        img = cv2.imread(XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp", cv2.IMREAD_GRAYSCALE)  # grayscale
        img = cv2.resize(img, (150, 150),interpolation=cv2.INTER_NEAREST)  # resize for speed
        h, w = img.shape
        
        
        img = np.expand_dims(img, axis=-1)
        image_rgb=normalize(img)
        image_rgb_2d = image_rgb.reshape(-1, image_rgb.shape[-1])
        
        #dipencoder = DipEncoder(2, pretrain_epochs=3, clustering_epochs=3)
        #dipencoder.fit(image_rgb)
        #true_label=(dipencoder.labels_)

        dec = DEC(n_clusters=2, pretrain_epochs=2, clustering_epochs=2)
        dec.fit(image_rgb_2d)
        true_label=(dec.dec_labels_)
        
        img=cv2.imread(XRAY_DIR+'masks/CXR_Mask_' + str(i)+'.bmp', cv2.IMREAD_GRAYSCALE)
        GT=cv2.resize(img, (150, 150),interpolation=cv2.INTER_NEAREST)

        GT=np.where(GT >= 1,1,0)
        GT=GT.ravel()
        result1=result2=true_label
       
  
        result1 = np.where(true_label == 0, 1, 0)
        result2 = np.where(true_label == 1, 1, 0)
        var=0

        if var == 0:
             iou1 = iou_score(result1, GT)
             iou2 = iou_score(result2, GT)
           
           
             if iou1 > iou2:
                 results = compute_metrics(result1, GT)
         
                 dice = dice_coefficient(result1, GT)
             
                 mse, rmse = mse_and_rmse(result1, GT)
           
                 psnr_value = psnr(result1 * 255, GT* 255)
                
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
        else:
            r1['IoU'].append(0)
            r1['Dice'].append(0)
            r1['MSE'].append(0)
            r1['RMSE'].append(0)
            r1['PSNR'].append(0)
            r1['accuracy'].append(0)
            r1['precision'].append(0)
            r1['recall'].append(0)
            r1['F1'].append(0)
            r1['confusion_matrix'].append([])      

    
    for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
      r1[v].append(sum(r1[v])/len(r1[v]))

    r1['confusion_matrix'].append('Null')
    print("*"*20+"RESULT 1"+"*"*20)
    #df1 = pd.DataFrame(r1)
    df = pd.DataFrame(r1)
    df.to_csv("./1D_NON_COVID/DEC_COVID.csv")
comput3(100000000000.0, 2.0, 10000.0)
