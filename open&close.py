import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/opencv_lib/image/geometry.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))   # MORPH_ELLIPSE为椭圆形
#dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
dst = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("binary", thresh)
cv2.imshow("open", dst)
cv2.waitKey()
cv2.destroyAllWindows()