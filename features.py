import cv2
import numpy as np


def calculate_basic_features_cv(block_float):
    """Calculate mean, variance, standard deviation and total sum."""
    mean_val, std_val = cv2.meanStdDev(block_float)
    mean_val = float(mean_val[0][0])
    std_dev_val = float(std_val[0][0])
    var_val = std_dev_val ** 2
    sum_val = mean_val * block_float.size
    return mean_val, var_val, std_dev_val, sum_val


def calculate_stats_cv(block_float):
    """Calculate horizontal and vertical variances and standard deviations."""
    row_means = cv2.reduce(block_float, dim=1, rtype=cv2.REDUCE_AVG)
    row_stds = np.sqrt(cv2.reduce((block_float - row_means) ** 2, 1, cv2.REDUCE_AVG))
    col_means = cv2.reduce(block_float, dim=0, rtype=cv2.REDUCE_AVG)
    col_stds = np.sqrt(cv2.reduce((block_float - col_means) ** 2, 0, cv2.REDUCE_AVG))
    row_vars = row_stds ** 2
    col_vars = col_stds ** 2
    vH = float(np.mean(row_vars))
    vV = float(np.mean(col_vars))
    dH = float(np.mean(row_stds))
    dV = float(np.mean(col_stds))
    return vH, vV, dV, dH


def calculate_gradients_sobel_cv(block_float):
    """Calculate Sobel gradients, magnitude, direction and ratio."""
    Gh = cv2.Sobel(block_float, cv2.CV_32F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    Gv = cv2.Sobel(block_float, cv2.CV_32F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    mGv = float(np.mean(np.abs(Gv)))
    mGh = float(np.mean(np.abs(Gh)))
    mag = cv2.magnitude(Gv, Gh)
    direction = cv2.phase(Gh, Gv, angleInDegrees=True)
    razao_grad = mGh / (mGv + 1e-6)
    return mGv, mGh, float(np.mean(mag)), float(np.mean(direction)), float(razao_grad)


def calculate_gradients_prewitt_cv(block_float):
    """Calculate Prewitt gradients, magnitude, direction and ratio."""
    kernel_gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    Gh = cv2.filter2D(block_float, -1, kernel_gx, borderType=cv2.BORDER_REPLICATE)
    Gv = cv2.filter2D(block_float, -1, kernel_gy, borderType=cv2.BORDER_REPLICATE)
    mGv = float(np.mean(np.abs(Gv)))
    mGh = float(np.mean(np.abs(Gh)))
    mag = cv2.magnitude(Gv, Gh)
    direction = cv2.phase(Gh, Gv, angleInDegrees=True)
    razao_grad = mGh / (mGv + 1e-6)
    return mGv, mGh, float(np.mean(mag)), float(np.mean(direction)), float(razao_grad)


def calculate_contrast_features_cv(block_orig):
    """Compute min, max and range (peak-to-peak) using cv2.minMaxLoc."""
    blk_min, blk_max, _, _ = cv2.minMaxLoc(block_orig)
    blk_range = blk_max - blk_min
    return float(blk_min), float(blk_max), float(blk_range)


def calculate_laplacian_cv(block_float):
    """
    Compute the variance of the Laplacian, a sharpness (blur) indicator.
    Higher values = sharp; lower values = blurred.
    """
    laplacian_var = float(cv2.Laplacian(block_float, cv2.CV_32F, ksize=1).var())
    return laplacian_var


def calculate_entropy_cv(block_orig):
    """
    Compute Shannon entropy using cv2.calcHist.
    Measures block complexity / texture.
    """
    is_10bit = np.max(block_orig) > 255

    if is_10bit:
        hist_size = 1024
        hist_range = [0, 1024]
        block_calc = block_orig.astype(np.float32)
    else:
        hist_size = 256
        hist_range = [0, 256]
        block_calc = block_orig.astype(np.uint8)

    hist = cv2.calcHist([block_calc], [0], None, [hist_size], hist_range)
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    probabilities = hist[hist > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def fwht_1d(a):
    """In-place 1D Fast Walsh–Hadamard Transform (length must be a power of 2)."""
    h = 1
    n = a.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a


def fwht_2d(mat):
    """Apply 2D Hadamard transform in-place and return the result as float32."""
    mat = mat.astype(np.float32)
    for r in range(mat.shape[0]):
        fwht_1d(mat[r, :])
    for c in range(mat.shape[1]):
        fwht_1d(mat[:, c])
    return mat


def calculate_hadamard_block_features(block):
    """Calculate requested Hadamard features for the block and return values."""
    H = fwht_2d(block.copy())

    dc = float(H[0, 0])
    energy_total = float(np.sum(H ** 2))
    energy_ac = float(energy_total - dc ** 2)
    max_coef = float(np.max(H))
    min_coef = float(np.min(H))

    # Valores nos cantos
    top_left = float(H[0, 0])
    top_right = float(H[0, -1])
    bottom_left = float(H[-1, 0])
    bottom_right = float(H[-1, -1])

    return dc, energy_total, energy_ac, max_coef, min_coef, top_left, top_right, bottom_left, bottom_right


def extract_block(Y, x, y, block_w, block_h):
    """Extract the block from the Y matrix according to (x, y, w, h)."""
    if y + block_h > Y.shape[0] or x + block_w > Y.shape[1]:
        return None
    block = Y[y:y + block_h, x:x + block_w]
    if block.size == 0:
        return None
    return block
