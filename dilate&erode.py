import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('D:/opencv_lib/image/coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
img_dilate = cv2.dilate(img, kernel)   # 腐蚀和膨胀操作可以对二值图像，灰度图以及彩色图像使用
img_erode = cv2.erode(img, kernel)

plt.subplot(121),plt.imshow(img_dilate),plt.title("img_dilate")
plt.subplot(122),plt.imshow(img_erode),plt.title("img_erode")
plt.show()