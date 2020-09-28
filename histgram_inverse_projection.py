import cv2
import numpy as np
from matplotlib import pyplot as plt


target = cv2.imread("D:/opencv_lib/image/abbeyroad1.jpg")
template = cv2.imread("D:/opencv_lib/image/template.png")
target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
#计算模板直方图
roihist = cv2.calcHist([template_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) #180为H的范围，256为S、V的范围，与实际定义有所区别
#模板归一化
cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
dst = cv2.calcBackProject([target_hsv], [0, 1], roihist, [0, 180, 0, 256], 1)

cv2.imshow("target", dst)
cv2.waitKey()
cv2.destroyAllWindows()