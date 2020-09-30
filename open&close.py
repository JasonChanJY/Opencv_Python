import cv2
import numpy as np
from matplotlib import pyplot as plt


# 开闭操作
# img = cv2.imread('D:/opencv_lib/image/geometry.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   # MORPH_ELLIPSE为椭圆形
# #dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# dst = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("binary", thresh)
# cv2.imshow("open", dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 其他形态学操作
img = cv2.imread('D:/opencv_lib/image/geometry.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   # MORPH_ELLIPSE为椭圆形

dst1 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)  # 顶帽：原图-开操作
dst2 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)  # 黑帽：闭操作-原图
dst3 = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)  # 基本梯度：膨胀-腐蚀

d_img = cv2.dilate(gray, kernel)
e_img = cv2.erode(gray, kernel)
dst4 = cv2.subtract(d_img, gray)  # internal gradient
dst5 = cv2.subtract(gray, e_img)  # external gradient

plt.subplot(231), plt.imshow(dst1), plt.title("TOPHAT")
plt.subplot(232), plt.imshow(dst2), plt.title("BLACKHAT")
plt.subplot(233), plt.imshow(dst3), plt.title("GRADIENT")
plt.subplot(234), plt.imshow(dst4), plt.title("INTERNAL GRADIENT")
plt.subplot(235), plt.imshow(dst5), plt.title("EXTERNAL GRADIENT")
plt.show()
