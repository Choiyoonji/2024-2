import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_histogram_equalization_y_channel(image, s=0.8):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    Y = ycbcr_image[:, :, 0]
    
    equalized_Y = cv2.equalizeHist(Y)
    
    luminance_ratio = (equalized_Y / (Y + 1e-6))**s
    adjusted_channels = [
        np.clip(image[:, :, c] * luminance_ratio, 0, 255).astype('uint8') for c in range(3)
    ]
    
    output_image = cv2.merge(adjusted_channels)
    
    return output_image

color_image = cv2.imread('./dgu_night_color.png')


s_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for s in s_values:
    processed = apply_histogram_equalization_y_channel(color_image, s)
    results.append((s, processed))

plt.figure(figsize=(15, 10))

for i, (s, processed) in enumerate(results):
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title(f's = {s}')
    plt.axis('off')

plt.subplot(2, 4, len(s_values) + 1)
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 4, len(s_values) + 2)
plt.imshow(cv2.cvtColor(results[3][1], cv2.COLOR_BGR2RGB))
plt.title('Best Image (s = 0.8)')
plt.axis('off')

plt.tight_layout()
plt.show()
