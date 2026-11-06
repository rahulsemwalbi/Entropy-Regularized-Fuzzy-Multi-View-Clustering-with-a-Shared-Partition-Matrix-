# COVID / non-COVID chest X-ray, SPMVKFCM: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, RUN_LAMBDA, RUN_SIGMA, XRAY_DIR, FIRST_ID, END_ID, make_adapter.

class MVKFCM1D:
    """
    1‑D Multi‑View Kernel Fuzzy C‑Means clustering for image segmentation.

    This implementation matches the update rules derived from the multi‑view
    fuzzy C‑means objective with a Gaussian kernel and entropy regularisation.
    It accepts a feature matrix x of shape (V, N) where each row is a
    flattened view of length N and returns membership degrees for C
    clusters.
    """
    def __init__(self, x: np.ndarray, C: int,
                 fuzzy: float = 2.0,
                 sigma: float | None = None,
                 param: float = 1e11,
                 iterations: int = 40,
                 seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.x = np.asarray(x, dtype=np.float64)
        self.V, self.N = self.x.shape
        self.C = C
        self.fuzzy = fuzzy
        self.iterations = iterations
        # Estimate sigma if not provided
        if sigma is None:
            sigma = 0.2 * np.std(self.x)
        self.sigma = float(sigma)
        self.param = param
        # Initialize weights uniformly across views
        self.w = np.full(self.V, 1.0 / self.V)
        # Initialize cluster centres randomly within data range per view
        min_vals = self.x.min(axis=1)
        max_vals = self.x.max(axis=1)
        # self.v = np.vstack([
        #     min_vals + (max_vals - min_vals) * np.random.rand(self.V)
        #     for _ in range(C)
        # ]).astype(float)
        self.v = min_vals + (max_vals - min_vals) * np.random.rand(C, self.V)
        # Initialize memberships randomly and normalise per pixel
        u = np.random.rand(C, self.N)
        u /= u.sum(axis=0, keepdims=True)
        self.u = u
        self.loss_margin: list[float] = []

    def _kernel_diff(self) -> np.ndarray:
        """Return 1 - Gaussian similarity tensor of shape (C, V, N)."""
        diff_list: list[np.ndarray] = []
        for i in range(self.C):
            sq = (self.x - self.v[i][:, None])**2
            diff_list.append(1.0 - np.exp(-sq / (self.sigma ** 2)))
        return np.stack(diff_list, axis=0)

    def _kernel_sim(self) -> np.ndarray:
        """Return Gaussian similarity tensor of shape (C, V, N)."""
        sim_list: list[np.ndarray] = []
        for i in range(self.C):
            sq = (self.x - self.v[i][:, None])**2
            sim_list.append(np.exp(-sq / (self.sigma ** 2)))
        return np.stack(sim_list, axis=0)

    def update_weights(self) -> None:
        """Update view weights using current centres and memberships."""
        diff = self._kernel_diff()  # (C, V, N)
        u_fuzzy = self.u ** self.fuzzy  # (C, N)
        cost = 2.0 * np.sum(diff * u_fuzzy[:, None, :], axis=(0, 2))  # (V,)
        z = -cost / self.param
        z -= z.max()
        numer = np.exp(z)
        self.w = numer / (numer.sum() + 1e-12)

    def update_memberships(self) -> None:
        """Update memberships using current centres and weights."""
        diff = self._kernel_diff()  # (C, V, N)
        d = np.sum(diff * self.w[None, :, None], axis=1)  # (C, N)
        d = np.maximum(d, 1e-12)
        exponent = 1.0 / (self.fuzzy - 1.0)
        d_expanded = d[:, None, :]
        ratio = (d_expanded / d[None, :, :]) ** exponent  # (C, C, N)
        inv_sum = ratio.sum(axis=1)  # (C, N)
        u_new = 1.0 / inv_sum
        u_new /= (u_new.sum(axis=0, keepdims=True) + 1e-12)
        self.u = u_new

    def update_centres(self) -> None:
        """Update cluster centres using current memberships and kernel similarities."""
        sim = self._kernel_sim()  # (C, V, N)
        u_fuzzy = self.u ** self.fuzzy  # (C, N)
        numer = np.sum(sim * self.x[None, :, :] * u_fuzzy[:, None, :], axis=2)  # (C, V)
        denom = np.sum(sim * u_fuzzy[:, None, :], axis=2)  # (C, V)
        denom = np.maximum(denom, 1e-12)
        self.v = numer / denom

    def optimize(self) -> None:
        """Run the alternating optimisation steps."""
        prev_loss = None
        for _ in range(self.iterations):
            self.update_weights()
            self.update_memberships()
            self.update_centres()
            # compute loss
            diff = self._kernel_diff()
            cost = np.sum(self.w[None, :, None] * (self.u ** self.fuzzy)[:, None, :] * 2.0 * diff)
            entropy = self.param * np.sum(self.w * np.log(self.w + 1e-12))
            loss = cost + entropy
            self.loss_margin.append(float(loss))
            if prev_loss is not None and abs(loss - prev_loss) < 1e-6:
                break
            prev_loss = loss

    def get_labels(self) -> np.ndarray:
        """Return hard labels per sample as argmax memberships."""
        return np.argmax(self.u, axis=0)

MVKFCM1D = make_adapter(MVKFCM1D, kernel=True)   # optimisation: SPMVKFCM (spmvfcm/methods.py)

def MKVFCM1D_segmentations(image_path, mask_path,resize_shape = (150, 150),fuzzy = 2.0,\
                          sigma = None,param=1e11,iterations=40,seed=RUN_SEED,\
                          save_output=None, r1=None):
    """
    Run MVKFCM1D and k‑means on a single image and compare IoU.

    Parameters
    ----------
    image_path, mask_path : str
        Paths to grayscale image and binary mask.
    resize_shape : tuple, optional
        Target size for resizing (default (150,150)).
    fuzzy, sigma, param, iterations, seed : see MVKFCM1D.__init__.
    save_output : str or None, optional
        Directory to save segmentation results.  If None, results are not saved.
    """
    # read image and mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if resize_shape is None:
        resize_shape=img.shape
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask {mask_path}")
    # resize
    
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    GT=np.where(mask >= 1,1,0)

    
    N = resize_shape[0] * resize_shape[1]
    # build flattened views (raw intensity and histogram equalisation)
    # view_raw =img.reshape(-1)
    # hist_eq = cv2.equalizeHist(img)
    # view_hist = hist_eq.reshape(-1)

    # clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(4,4))
    # view_clahe = clahe.apply(img).reshape(-1)
    
    # 4. Gamma Correction (try 0.5 and 2.0)
    #gamma1 = 0.1
    #gamma2 = 2.2
    # view_gamma1 = np.power(img/255.0, gamma1)
    # view_gamma1 = (view_gamma1 * 255).astype(np.uint8).reshape(-1)
    gamma2 = 2.2
    view_gamma2 = np.power(img/255.0, gamma2)
    view_gamma2 = (view_gamma2 * 255).astype(np.uint8).reshape(-1)
    
    # 5. Negative Image
    #view_negative = (255 - img).reshape(-1)
    gamma2 = 2.2
    view_gamma2 = np.power(img/255.0, gamma2)
    view_gamma2 = (view_gamma2 * 255).astype(np.uint8).reshape(-1)
    r = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    
    # Morphological gradients
    dil = cv2.dilate(img, kernel)
    ero = cv2.erode(img, kernel)
    morph_grad = cv2.subtract(dil, ero).reshape(-1)         # edge-strength
    internal_grad = cv2.subtract(img, ero).reshape(-1)      # inside edges
    external_grad = cv2.subtract(dil, img).reshape(-1)      # outside edges

    #LL, (LH, HL, HH) = pywt.dwt2(img.astype(np.float32), 'haar')
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))

    # Black-hat transform
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).ravel()
   
    #x_data = np.stack([view_raw, view_hist,view_clahe,view_gamma1,view_gamma2,view_negative], axis=0)
    x_data = np.stack([view_gamma2,blackhat, morph_grad,external_grad], axis=0)
    # MVKFCM
    mvk = MVKFCM1D(x_data, C=2, fuzzy=fuzzy, sigma=sigma, param=param,
                   iterations=iterations, seed=seed)
    mvk.optimize()

    
    mvk_labels = mvk.get_labels().reshape(resize_shape)
    
    
    result1 = (mvk_labels == 0).astype(np.uint8)
    result2 = (mvk_labels == 1).astype(np.uint8)
    iou1 = iou_score(result1, GT)
    iou2 = iou_score(result2, GT)
    #print(iou1)
    #print(iou2)

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
    MKVFCM1D_segmentations(image_path=XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp",
                          mask_path=XRAY_DIR+'masks/CXR_Mask_' + str(i)+'.bmp',
                          resize_shape = (150, 150),
                          fuzzy= 2,
                          sigma= RUN_SIGMA,
                          param= RUN_LAMBDA,
                          iterations= 40,
                          seed= RUN_SEED,
                          save_output= None,
                          r1=r1)
for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
    r1[v].append(sum(r1[v])/len(r1[v]))

r1['confusion_matrix'].append('Null')
print("*"*20+"RESULT 1"+"*"*20)
df1 = pd.DataFrame(r1)
df1.to_csv("./1D_NON_COVID/MVKFC_COVID.csv")
