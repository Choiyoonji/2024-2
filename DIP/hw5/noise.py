import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2


# Gaussian Noise Addition
def add_gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image + noise, 0, 255)


# Mean Filter
def mean_filter(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    denoised_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            denoised_image[i, j] = np.mean(padded_image[i:i+kernel_size, j:j+kernel_size])
    return denoised_image


# Median Filter
def median_filter(image, kernel_size=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    denoised_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            denoised_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])
    return denoised_image


# Gaussian Filter
def gaussian_filter(image, kernel_size=3, sigma=1.0):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    denoised_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            denoised_image[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    return denoised_image


# Bilateral Filter
def bilateral_filter(image, kernel_size=5, sigma_color=25, sigma_space=3):
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    denoised_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            center_pixel = image[i, j]
            weights = []
            values = []
            for di in range(-pad_size, pad_size+1):
                for dj in range(-pad_size, pad_size+1):
                    neighbor_pixel = padded_image[i+pad_size+di, j+pad_size+dj]
                    spatial_weight = np.exp(-(di**2 + dj**2) / (2 * sigma_space**2))
                    color_weight = np.exp(-((neighbor_pixel - center_pixel)**2) / (2 * sigma_color**2))
                    weight = spatial_weight * color_weight
                    weights.append(weight)
                    values.append(neighbor_pixel)
            weights = np.array(weights)
            values = np.array(values)
            denoised_image[i, j] = np.sum(weights * values) / np.sum(weights)
    return denoised_image


# Non-Local Means Filter
def nlm_filter(image, patch_size=3, window_size=7, h=20):
    pad_size = patch_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    denoised_image = np.zeros_like(image)
    half_window = window_size // 2
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            center_patch = padded_image[i:i+patch_size, j:j+patch_size]
            weights = []
            values = []
            for m in range(max(i-half_window, 0), min(i+half_window+1, image.shape[0])):
                for n in range(max(j-half_window, 0), min(j+half_window+1, image.shape[1])):
                    current_patch = padded_image[m:m+patch_size, n:n+patch_size]
                    diff = (current_patch - center_patch)**2
                    weight = np.exp(-np.sum(diff) / h**2)
                    weights.append(weight)
                    values.append(image[m, n])
            weights = np.array(weights)
            values = np.array(values)
            weights /= np.sum(weights)
            denoised_image[i, j] = np.sum(values * weights)
    return np.clip(denoised_image, 0, 255)


# Wavelet Denoising
def wavelet_denoise(image, wavelet='db1', level=2, thresholding='soft'):
    def soft_threshold(coeff, threshold):
        return np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_thresh = [coeffs[0]] + [
        tuple(soft_threshold(detail, threshold) for detail in coeff)
        for coeff in coeffs[1:]
    ]
    return np.clip(pywt.waverec2(coeffs_thresh, wavelet), 0, 255)


# Load PNG Image
def load_png_image(filepath):
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(image, dtype=np.float32)


# Main
filepath = "dgu_gray.png"  # Update this to your PNG file path
image = cv2.imread(filepath, 0).astype(np.float32)

# Add Gaussian Noise
noisy_image = add_gaussian_noise(image, mean=0, std=20)

# Apply Filters
denoised_mean = mean_filter(noisy_image)
denoised_median = median_filter(noisy_image)
denoised_gaussian = gaussian_filter(noisy_image)
denoised_bilateral = bilateral_filter(noisy_image)
denoised_nlm = nlm_filter(noisy_image)
denoised_wavelet = wavelet_denoise(noisy_image)

# Plot
plt.figure(figsize=(15, 10))
filters = [
    ("Original Noisy Image", noisy_image),
    ("Mean Filter", denoised_mean),
    ("Median Filter", denoised_median),
    ("Gaussian Filter", denoised_gaussian),
    ("Bilateral Filter", denoised_bilateral),
    ("Non-Local Means Filter", denoised_nlm),
    ("Wavelet Denoising", denoised_wavelet),
]
for i, (title, img) in enumerate(filters, 1):
    plt.subplot(2, 4, i)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
