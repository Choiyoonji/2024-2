import cv2
import numpy as np


def transform(img, angle):  # forward transformation
    height, width = img.shape
    result = np.zeros((height, width), np.uint8)  # result image

    affine = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                       [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                       [0, 0, 1]])  # Affine transformation matrix

    for x in range(width):
        for y in range(height):
            p = affine.dot(np.array([x, y, 1]))

            xp = int(p[0])
            yp = int(p[1])

            if 0 <= yp < height and 0 <= xp < width:
                result[yp, xp] = img[y, x]
    return result


in_image = cv2.imread('dgu_gray.png', 0)  # img2numpy

out_image = transform(in_image, 20)

cv2.imshow('Input Image', in_image)
cv2.imshow('Result Image', out_image)

cv2.imwrite('dgu_gray_rotate.png', out_image)  # save result img
cv2.waitKey()
