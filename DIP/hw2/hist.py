import numpy as np
import cv2
from matplotlib import pyplot as plt

low_light_image = cv2.imread('dgu_night.png', 0)

# 히스토그램 계산
hist, bins = np.histogram(low_light_image.flatten(), 256, [0, 256])

# 누적 히스토그램 계산
cdf = hist.cumsum()

# 누적 히스토그램의 최소값을 이용하여 정규화
cdf_min = cdf[cdf > 0].min()
cdf_normalized = (cdf - cdf_min) / (cdf.max() - cdf_min) * 255
cdf_normalized = cdf_normalized.astype(np.uint8)

# 이미지의 각 픽셀 값에 대해 매핑된 값 적용
equalized_image = cdf_normalized[low_light_image]

images = np.vstack((low_light_image, equalized_image))
cv2.imwrite('dgu_light.png', images)
cv2.imshow('Result Image', images)
cv2.waitKey()