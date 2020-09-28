import cv2
import numpy as np
from matplotlib import pyplot as plt


# 金字塔操作需要输入图像长宽相同
# 降采样
def pyramid_down(image):
    level = 3              # 3层金字塔
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv2.pyrDown(temp)
        pyramid_images.append(dst)
        temp = dst.copy()
        cv2.imshow("pyramid_down"+str(i), dst)
    return pyramid_images


# 高斯金字塔
def Laplacian_pyramid(image):
    pyramid_images = pyramid_down(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:            # 最底层金字塔
            dst = cv2.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            Laplacian_image = cv2.subtract(image, dst)
            cv2.imshow("Laplacian_pyramid"+str(i), Laplacian_image)
        else:
            dst = cv2.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            Laplacian_image = cv2.subtract(pyramid_images[i-1], dst)
            cv2.imshow("Laplacian_pyramid" + str(i), Laplacian_image)


img = cv2.imread(r"D:\opencv_lib\image\lena.jpg")
Laplacian_pyramid(img)
cv2.waitKey()
cv2.destroyAllWindows()