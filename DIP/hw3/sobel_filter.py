import cv2
import numpy as np
import matplotlib.pyplot as plt

image =  cv2.imread('dgu_gray.png', 0)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])

height, width = image.shape

gradient_x = np.zeros_like(image)
gradient_y = np.zeros_like(image)
edge_magnitude = np.zeros_like(image, dtype=np.float32)

for i in range(1, height-1):
    for j in range(1, width-1):
        gx = np.sum(np.multiply(sobel_x, image[i-1:i+2, j-1:j+2]))
        gy = np.sum(np.multiply(sobel_y, image[i-1:i+2, j-1:j+2]))
        
        gradient_x[i, j] = gx
        gradient_y[i, j] = gy
        
        edge_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

edge_magnitude = np.clip(edge_magnitude / np.max(edge_magnitude) * 255, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 7))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(gradient_x, cmap='gray')
plt.title('Gradient X')

plt.subplot(1, 3, 3)
plt.imshow(gradient_y, cmap='gray')
plt.title('Gradient Y')

plt.figure()
plt.imshow(edge_magnitude, cmap='gray')
plt.title('Sobel Edge Detection')

plt.show()
