# ISIC dermatology, kmeans: segments every image of `images` and writes the CSV of per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, IMAGE_DIR, images, directory_path_test.

def kmeans_segmentation(X, k=2, max_iters=40, seed=RUN_SEED):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]                  # make it (N,1)
    N, D = X.shape

    # 0–255 init per dimension
    rng = np.random.default_rng(RUN_SEED)
    centers = (X.min(axis=0) + (X.max(axis=0) - X.min(axis=0)) * rng.random(size=(k, D))).astype(float)
    #centers = rng.integers(0, 1, size=(k, D)).astype(np.float32)

    labels = np.zeros(N, dtype=np.int32)
    for _ in range(max_iters):
        diff = centers[:, None, :] - X[None, :, :]    # (k,N,D)
        dists = np.sum(diff * diff, axis=2)           # (k,N)
        new_labels = np.argmin(dists, axis=0)

        new_centers = centers.copy()
        for i in range(k):
            m = (new_labels == i)
            if np.any(m):
                new_centers[i] = X[m].mean(axis=0)
            else:
                new_centers[i] = (X.min(axis=0) + (X.max(axis=0) - X.min(axis=0)) * rng.random(D)).astype(np.float32)

        moved = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers, labels = new_centers, new_labels
        if moved < 1e-6:
            break
    return labels, centers

def KMEANS1D_segmentations( image_path, k=2,max_iters=40,atol=1e-5,sigma = None,param=1e11,iterations=40,seed=RUN_SEED,save_output=None, r=None):
    
    # read image and mask
    features = ImageFeature(image_path,(200,200))
    image_rgb,hsv_image=features.convert_to_rgb_hsv()
    img = image_rgb
    pixels = img.reshape(-1, 3)
    k_labels, k_centroids = kmeans_segmentation(pixels, k=2, max_iters=iterations, seed=seed)
    labels=k_labels.reshape(img.shape[0],img.shape[1])
    result1 = np.where(labels == 0, 1, 0).astype(np.uint8)
    result2 = np.where(labels == 1, 1, 0).astype(np.uint8)
    return result1,result2

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


 
    result1,result2=KMEANS1D_segmentations(image_path, k=2,max_iters=40,atol=1e-5)
    img = cv2.imread(os.path.join(directory_path_test, images[i] + "_Segmentation.png"))
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
df1.to_csv("FINAL_result_kmeans_900.csv")
