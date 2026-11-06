# COVID / non-COVID chest X-ray, kmeans: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, XRAY_DIR, FIRST_ID, END_ID.

def kmeans_segmentation(flatten: np.ndarray, k: int = 2, max_iters: int = 40,
                        seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k‑means clustering on a 1‑D array of shape (N,)."""
    if seed is not None:
        np.random.seed(seed)
    x = flatten.astype(float)
    #centroids = np.random.choice(x, size=k, replace=False)
    centroids = x.min() + (x.max() - x.min()) * np.random.rand(k)
    
    labels = np.zeros_like(x, dtype=int)
    for _ in range(max_iters):
        dist = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dist, axis=1)
        new_centroids = np.array([
            x[labels == i].mean() if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def KMEANS1D_segmentations(image_path, mask_path,resize_shape = (150, 150),fuzzy = 2.0,\
                          sigma = None,param=1e11,iterations=40,seed=RUN_SEED,\
                          save_output=None, r=None):
    
    # read image and mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    if resize_shape is None:
        resize_shape=img.shape
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask {mask_path}")
    # resize
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    GT=np.where(mask >= 1,1,0)
    N = resize_shape[0] * resize_shape[1]
    # build flattened views (raw intensity and histogram equalisation)
    view_raw = img.reshape(-1)
    k_labels, k_centroids = kmeans_segmentation(view_raw, k=2, max_iters=iterations, seed=seed)
    k_labels = k_labels.reshape(resize_shape)

    result1 = (k_labels == 0).astype(np.uint8)
    result2 = (k_labels == 1).astype(np.uint8)
    iou1 = iou_score(result1, GT)
    iou2 = iou_score(result2, GT)


    iou=(iou1,iou2)
    index=np.argmax(iou)
    result=(result1,result2)

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

    
    # print(iou1, iou2)
    # fig, axs = plt.subplots()
    # axs.imshow(GT, cmap='Greys',  interpolation='nearest')
    # axs.set_title('Orignal_segmentation')
    # axs.axis('off')
    # plt.show()    
    # fig, axs = plt.subplots()
    # axs.imshow(result1, cmap='Greys',  interpolation='nearest')
    # axs.set_title('Orignal_segmentation')
    # axs.axis('off')
    # plt.show()    
    # fig, axs = plt.subplots()
    # axs.imshow(result2, cmap='Greys',  interpolation='nearest')
    # axs.set_title('Orignal_segmentation')
    # axs.axis('off')
    # plt.show()    
    

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
    KMEANS1D_segmentations(image_path=XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp",
                          mask_path=XRAY_DIR+'masks/CXR_Mask_' + str(i)+'.bmp',
                          resize_shape = (150, 150),
                          fuzzy= 1.09,
                          sigma= 10000.0,
                          param= 1e11,
                          iterations= 40,
                          seed= RUN_SEED,
                          save_output= None,
                          r=r1)

for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
    r1[v].append(sum(r1[v])/len(r1[v]))

r1['confusion_matrix'].append('Null')
print("*"*20+"RESULT 1"+"*"*20)
df1 = pd.DataFrame(r1)
df1.to_csv("./1D_NON_COVID/KMEANS_COVID.csv")
