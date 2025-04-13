import cv2
import numpy as np
from skimage import io, color, feature
from skimage.measure import shannon_entropy
from scipy.stats import entropy

# --- Detail Metrics ---

def edge_density(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    density = np.sum(edges > 0) / edges.size
    return density

def laplacian_variance(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

def image_entropy(image_path):
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = img.astype("float32") / 255.0
    gray = color.rgb2gray(img)
    return shannon_entropy(gray)

# --- Color Metrics ---

def colorfulness(image_path):
    img = cv2.imread(image_path)
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def _histogram_spread(channel, bins=32):
    hist, _ = np.histogram(channel, bins=bins, range=(0, 1), density=False)
    non_zero_bins = np.count_nonzero(hist)
    return non_zero_bins / bins

def _histogram_entropy(channel, bins=32):
    hist, _ = np.histogram(channel, bins=bins, range=(0, 1), density=True)
    return entropy(hist + 1e-10)

def color_histogram_diversity(image_path, color_space='hsv', bins=32):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    if color_space == 'hsv':
        converted = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    elif color_space == 'lab':
        converted = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2Lab) / 255.0
    else:
        raise ValueError("Choose 'hsv' or 'lab'.")
    diversity_metrics = []
    for i in range(3):
        channel = converted[:, :, i].flatten()
        spread = _histogram_spread(channel, bins)
        ent = _histogram_entropy(channel, bins)
        diversity_metrics.append((spread, ent))
    avg_spread = np.mean([m[0] for m in diversity_metrics])
    avg_entropy = np.mean([m[1] for m in diversity_metrics])
    return {
        "avg_spread": avg_spread,
        "avg_entropy": avg_entropy
    }

# --- Temperature Metrics ---

def temperature_analysis(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel = image[:, :, 0].astype(np.int16)
    blue_channel = image[:, :, 2].astype(np.int16)
    diff = red_channel - blue_channel
    temp_mean = np.mean(diff)
    temp_stddev = np.std(diff)
    return {
        "temp_mean": temp_mean,
        "temp_stddev": temp_stddev
    }

# --- Contrast Metrics ---

def grayscale_contrast_analysis(image_path):
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = img.astype("float32") / 255.0
    gray = color.rgb2gray(img)
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return {
        "mean_gray": mean_val,
        "std_gray": std_val
    }

# --- Texture Metrics (LBP) ---

def compute_lbp_features(image_path, radius=1, n_points=8):
    img = io.imread(image_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    gray = color.rgb2gray(img)
    lbp = feature.local_binary_pattern(gray, P=n_points, R=radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# --- Final Feature Extraction ---

def feature_extraction(image_path):
    density = edge_density(image_path)
    variance = laplacian_variance(image_path)
    entropy_val = image_entropy(image_path)
    colorful = colorfulness(image_path)
    color_histogram = color_histogram_diversity(image_path)
    temperature = temperature_analysis(image_path)
    grayscale_contrast = grayscale_contrast_analysis(image_path)
    lbp_features = compute_lbp_features(image_path)

    result = [
        density,
        variance,
        entropy_val,
        colorful,
        color_histogram['avg_spread'],
        color_histogram['avg_entropy'],
        temperature['temp_mean'],
        temperature['temp_stddev'],
        grayscale_contrast['mean_gray'],
        grayscale_contrast['std_gray']
    ]
    result += lbp_features.tolist()
    return result

image_path = rf'dataset/mery/danbooru_866364_32a36dadb2476488304e227fbc9be19e.png'
print(feature_extraction(image_path))