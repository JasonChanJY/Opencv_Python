import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread(r"D:\opencv_lib\image\abbeyroad1.jpg", 0)
#二值化前先转为灰度图，ret返回的是阈值(若选择的是otsu或triangle等方法，返回的阈值是算法计算的值，函数设置的阈值将不起效)
#最后一个参数表示二值化，并选择阈值分割算法
ret, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
print("ret: ", ret)

cv2.imshow("targe", threshold)
cv2.waitKey()
cv2.destroyAllWindows()