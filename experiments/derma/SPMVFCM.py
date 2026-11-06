# ISIC dermatology, SPMVFCM: segments every image of `images` and writes the CSV of per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, RUN_LAMBDA, IMAGE_DIR, images, directory_path_test,
# the model class MVJFC (spmvfcm/methods.py) and derma_views (spmvfcm/views.py).

np.random.seed(RUN_SEED)

#NO OUTLIER
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

        image_rgb = cv2.cvtColor(features.image, cv2.COLOR_BGR2RGB)
    
      #ImagePlotting.rgb_hsv(image_rgb, hsv_image)
        #ImagePlotting.plot_image(features.hair_removed_image)
    
        orgb_b,orgb_g,orgb_r = cv2.split(features.hair_removed_image)
    
        rgb_b,rgb_g,rgb_r = image_rgb[:,:,0],image_rgb[:,:,1],image_rgb[:,:,2]#cv2.split(image_rgb)
      #hsv_image_b,hsv_image_g,hsv_image_r=hsv_image[:,:,0],hsv_image[:,:,1],hsv_image[:,:,2]#cv2.split(hsv_image)
    
        x = derma_views(features.image, ('rgb', 'lab', 'morph'))
    
        x=np.array(x)
        shape_x = x.shape
        num_view = shape_x[0]
        C=2
        shape_u = (C,) + shape_x[1:]
        w = np.full(len(x), 1/len(x))
        error=5e-2
    
    
      #v=np.random.randint(255, size=(len(x), C)) + 5e-2
        #v= np.repeat(np.random.randint(1, size=C)[np.newaxis, :], len(x), axis=0)+5e-2
        rng = np.random.default_rng(RUN_SEED)
        x_init = x.reshape(len(x), -1).astype(float)
        v = x_init.min(axis=1)[:, None] + (x_init.max(axis=1) - x_init.min(axis=1))[:, None] * rng.random(size=(len(x), C))
        shape_u = (C,) + shape_x[1:]
        u = np.random.rand(*shape_u)
        u /= u.sum(axis=0, keepdims=True)
    
    
        m=MVJFC(x, w, v, u,C, iterations=40, fuzzy=fuzzy_value, param=param_value,div_error_corr=1e-10)
        m.optimize()
    
    
    
        final_result = np.argmax(m.u, axis=0)   # shape = (height, width)
        final_result=final_result.reshape((200, 200))
        result1 = (final_result == 0).astype(np.uint8)
        result2 = (final_result == 1).astype(np.uint8)
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
        # axs[0].imshow(result1, cmap='Greys',  interpolation='nearest')
        # axs[0].set_title('Result0')
        # axs[0].axis('off')  # Hide axes for image
        # axs[1].imshow(result2, cmap='Greys',  interpolation='nearest')
        # axs[1].set_title('Result1')
        # axs[1].axis('off')  # Hide axes for image
        # plt.show()      
  
    
        img=cv2.imread(os.path.join(directory_path_test, images[i] + "_Segmentation.png"))
        img=cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)#cv2.resize(img, (200, 200))
        GT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
        GT=np.where(GT >= 1,1,0)
      #GT = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
        
      # fig, axs = plt.subplots(1, 1, figsize=(5, 5))
      # axs.imshow(GT, cmap='Greys',  interpolation='nearest')
      # axs.set_title('Orignal_segmentation')
      # axs.axis('off')
      # plt.show()    
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
    df1.to_csv("FINAL_RESULT_no_kernal_900_1.csv")
comput3(RUN_LAMBDA, 2.0, 10000.0)