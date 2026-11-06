# ISIC dermatology, dipencoder: segments every image of `images` and writes the CSV of per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, IMAGE_DIR, images, directory_path_test.

#NO OUTLIER
from clustpy.deep import DCN
from clustpy.data import create_subspace_data
from clustpy.deep import DipEncoder
from clustpy.deep import DEC
from clustpy.deep import DipDECK
from clustpy.data import create_subspace_data
from clustpy.deep import DipEncoder
np.random.seed(RUN_SEED)
def comput3(param_value,fuzzy_value,sigma_value):

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
    

    for i in range(len(images)):

        if i>=0 and i<=143:
            features = ImageFeature(os.path.join(IMAGE_DIR, images[i] + ".jpg"),(200,200))
        elif i>=144 and i<=756:
            features = ImageFeature(os.path.join(IMAGE_DIR, images[i] + ".jpg"),(200,200))
        else:
            features = ImageFeature(os.path.join(IMAGE_DIR, images[i] + ".jpg"),(200,200))

        image_rgb, hsv_image = features.convert_to_rgb_hsv()
        ImagePlotting.plot_image(features.hair_removed_image)
        rgb_b,rgb_g,rgb_r = image_rgb[:,:,0],image_rgb[:,:,1],image_rgb[:,:,2]#cv2.split(image_rgb)
        image_rgb_2d = image_rgb.reshape(-1, image_rgb.shape[-1])

        #dipencoder = DipEncoder(2, pretrain_epochs=3, clustering_epochs=3)
        #dipencoder.fit(image_rgb)
        #true_label=(dipencoder.labels_)

        #dcn = DCN(n_clusters=2, pretrain_epochs=3, clustering_epochs=3)
        #dcn.fit(image_rgb_2d)
        #dec = DEC(n_clusters=2, pretrain_epochs=3, clustering_epochs=3)
        #dec.fit(image_rgb_2d)
        #dipdeck = DipDECK(pretrain_epochs=3, clustering_epochs=3)
        #dipdeck.fit(image_rgb_2d)
        dipencoder = DipEncoder(2, pretrain_epochs=2, clustering_epochs=2)
        dipencoder.fit(image_rgb_2d)
        true_label=(dipencoder.labels_)
        
        img=cv2.imread(os.path.join(directory_path_test, images[i] + "_Segmentation.png"))
        img=cv2.resize(img, (200, 200),interpolation=cv2.INTER_NEAREST)
        GT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
       

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
    #df1 = pd.DataFrame(r1)
    df = pd.DataFrame(r1)
    df.to_csv("FINAL_RESULT_dipencoder_900_1.csv")
comput3(100000000000.0, 2.0, 10000.0)
