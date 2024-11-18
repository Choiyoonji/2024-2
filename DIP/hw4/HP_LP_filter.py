import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft


img = cv2.imread('dgu_gray.png', 0)

height, width = img.shape

k = 50
h = np.ones((k,k))

h_lp = np.pad(h, ((height//2-k//2, width//2-k//2), (height//2-k//2, width//2-k//2)), 'constant')
h_hp = np.ones((height, width)) - h_lp

f_img = fft.fft2(img)
fshift_img = fft.fftshift(f_img)

mag_img = 20 * np.log(np.abs(fshift_img))
phs_img = np.angle(fshift_img)

fshift_img_lp = np.abs(fshift_img) * h_lp
fshift_img_hp = np.abs(fshift_img) * h_hp

recon_img_lp = fshift_img_lp * np.exp(1j * phs_img)
recon_img_lp = np.minimum(np.abs(np.real(fft.ifft2(fft.fftshift(recon_img_lp)))), 255)
recon_img_hp = fshift_img_hp * np.exp(1j * phs_img)
recon_img_hp = np.minimum(np.abs(np.real(fft.ifft2(fft.fftshift(recon_img_hp)))), 255)


# cv2.imshow(mag_img)
# cv2.imshow(phs_img)
# cv2.imshow(recon_img_lp)
# cv2.imshow(recon_img_hp)
plt.subplot(2, 2, 1)
plt.imshow(mag_img, cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(phs_img, cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(recon_img_lp, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(recon_img_hp, cmap='gray')

# cv2.waitKey()
plt.show()
