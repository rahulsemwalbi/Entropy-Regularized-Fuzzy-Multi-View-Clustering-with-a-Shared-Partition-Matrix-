# COVID / non-COVID chest X-ray, SPMVFCM: segments the images FIRST_ID ... END_ID - 1 and writes the CSV of
# per-image measures.
# Set by spmvfcm/runner.py: RUN_SEED, RUN_LAMBDA, XRAY_DIR, FIRST_ID, END_ID, make_adapter.

class MVJFCM1D:
    """
    1‑D multi‑view Fuzzy C‑Means clustering with view weights and
    Euclidean distance.  This implementation follows the given
    objective and update equations exactly.
    """
    def __init__(self, x: np.ndarray, C: int,
                 fuzzy: float = 2.0,
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
        self.param = param
        # initialize weights uniformly
        self.w = np.full(self.V, 1.0 / self.V)
        # initialize centres randomly within data range per view
        min_vals = self.x.min(axis=1)
        max_vals = self.x.max(axis=1)
        # self.v = np.vstack([
        #     min_vals + (max_vals - min_vals) * np.random.rand(self.V)
        #     for _ in range(C)
        # ]).astype(float)
        self.v = min_vals + (max_vals - min_vals) * np.random.rand(C, self.V)
        # initialize memberships randomly
        u = np.random.rand(C, self.N)
        u /= u.sum(axis=0, keepdims=True)
        self.u = u
        self.loss_margin: list[float] = []

    def update_weights(self) -> None:
        """Update view weights based on current centres and memberships."""
        # compute squared differences per cluster/view/sample
        diff = np.stack([
            (self.x - self.v[i][:, None])**2
            for i in range(self.C)
        ], axis=0)  # (C, V, N)
        # cost_j = Σ_i Σ_n diff[i,j,n] * u_i(n)^m
        u_fuzzy = (self.u**self.fuzzy)  # (C,N)
        cost = np.sum(diff * u_fuzzy[:, None, :], axis=(0, 2))  # (V,)
        # update w with softmax
        z = -cost / self.param
        z -= z.max()
        numer = np.exp(z)
        self.w = numer / (numer.sum() + 1e-12)

    def update_memberships(self) -> None:
        """Update membership matrix based on current centres and weights."""
        # compute distance per cluster/sample: d_i(n) = Σ_j w_j * ||x_j(n) - v_i(j)||^2
        diff = np.stack([
            (self.x - self.v[i][:, None])**2
            for i in range(self.C)
        ], axis=0)  # (C,V,N)
        d = np.sum(diff * self.w[None, :, None], axis=1)  # (C,N)
        d = np.maximum(d, 1e-12)
        exponent = 1.0 / (self.fuzzy - 1.0)
        d_expand = d[:, None, :]  # (C,1,N)
        ratio = (d_expand / d[None, :, :]) ** exponent  # (C,C,N)
        inv_sum = ratio.sum(axis=1)  # (C,N)
        u_new = 1.0 / inv_sum
        u_new /= (u_new.sum(axis=0, keepdims=True) + 1e-12)
        self.u = u_new

    def update_centres(self) -> None:
        """Update cluster centres based on current memberships."""
        u_fuzzy = (self.u**self.fuzzy)  # (C,N)
        numer = np.sum(u_fuzzy[:, None, :] * self.x[None, :, :], axis=2)  # (C,V)
        denom = np.sum(u_fuzzy, axis=1)[:, None]  # (C,1)
        denom = np.maximum(denom, 1e-12)
        self.v = numer / denom

    def optimize(self) -> None:
        """Run the alternating optimisation for the specified iterations."""
        prev_loss = None
        for _ in range(self.iterations):
            self.update_weights()
            self.update_memberships()
            self.update_centres()
            # compute current objective
            diff = np.stack([
                (self.x - self.v[i][:, None])**2
                for i in range(self.C)
            ], axis=0)
            cost = np.sum(self.w[None, :, None] * (self.u**self.fuzzy)[:, None, :] * diff)
            entropy = self.param * np.sum(self.w * np.log(self.w + 1e-12))
            loss = cost + entropy
            self.loss_margin.append(float(loss))
            if prev_loss is not None and abs(loss - prev_loss) < 1e-6:
                break
            prev_loss = loss

    def get_labels(self) -> np.ndarray:
        """Return hard labels (0..C-1) for each sample."""
        return np.argmax(self.u, axis=0)

MVJFCM1D = make_adapter(MVJFCM1D, kernel=False)  # optimisation: SPMVFCM (spmvfcm/methods.py)

def run_comparison(image_path,mask_path,
                   resize_shape = (150, 150),
                   fuzzy = 2.0,
                   param=1e11,
                   iterations = 40,
                   seed= RUN_SEED,
                   save_output = None, r=None):

    
    # read and resize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {image_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask {mask_path}")
    if resize_shape is None:
        resize_shape=img.shape
        
    img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    GT=np.where(mask >= 1,1,0)

    
    N = resize_shape[0] * resize_shape[1]
    # build flattened views (raw intensity and histogram equalisation)
    # view_raw =img.reshape(-1)
    # hist_eq = cv2.equalizeHist(img)
    # view_hist = hist_eq.reshape(-1)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # view_clahe = clahe.apply(img).reshape(-1)
    
    # # 4. Gamma Correction (try 0.5 and 2.0)
    # gamma1 = 0.5
    # gamma2 = 2.0
    # view_gamma1 = np.power(img/255.0, gamma1)
    # view_gamma1 = (view_gamma1 * 255).astype(np.uint8).reshape(-1)
    
    # view_gamma2 = np.power(img/255.0, gamma2)
    # view_gamma2 = (view_gamma2 * 255).astype(np.uint8).reshape(-1)
    
    # # 5. Negative Image
    # view_negative = (255 - img).reshape(-1)

    
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
   
   
    x_data = np.stack([view_gamma2,blackhat, morph_grad,external_grad], axis=0)


    
    #x_data = np.stack([view_raw, view_hist,view_clahe,view_gamma1,view_gamma2,view_negative], axis=0)
    # MVKFCM
    mvj = MVJFCM1D(x_data, C=2, fuzzy=fuzzy, param=param,
                   iterations=iterations, seed=seed)
    mvj.optimize()
    mvj_labels = mvj.get_labels().reshape(resize_shape)
   
    
    result1 = (mvj_labels == 0).astype(np.uint8)
    result2 = (mvj_labels == 1).astype(np.uint8)
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
    run_comparison(image_path=XRAY_DIR+"images/CXR_Images_"+str(i)+".bmp",
                   mask_path=XRAY_DIR+'masks/CXR_Mask_' + str(i)+'.bmp',
                   resize_shape = (150, 150),
                   fuzzy=2, param=RUN_LAMBDA,
                   iterations=40, seed=RUN_SEED,
                   save_output=None,r=r1)
for v in ['IoU','Dice','MSE','RMSE','PSNR','accuracy','precision','recall','F1']:
    r1[v].append(sum(r1[v])/len(r1[v]))

r1['confusion_matrix'].append('Null')
print("*"*20+"RESULT 1"+"*"*20)
df1 = pd.DataFrame(r1)
df1.to_csv("./1D_NON_COVID/MVJFC_COVID.csv")
