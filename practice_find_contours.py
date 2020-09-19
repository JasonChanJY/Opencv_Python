import cv2
import numpy as np

###轮廓框
# img = cv2.imread(r'D:\opencv_lib\image\hammer.png', 0)
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
# for c in contours:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(color, (x,y), (x+w,y+h), (255, 0, 0), 2)
#
#     rect = cv2.minAreaRect(c)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(color, [box], 0, (0, 0, 255), 2)
#
#     (x, y), radius = cv2.minEnclosingCircle(c)
#     center = (int(x), int(y))
#     radius = int(radius)
#     color = cv2.circle(color, center, radius, (255, 255, 0), 2)
#
# cv2.imshow('lena.jpg', color)
# cv2.waitKey()
# cv2.destroyAllWindows()

###线检测，圆检测
img = cv2.imread(r'D:\opencv_lib\image\thunder.jpg', 0)
img = cv2.resize(img, (300, 300))
edge = cv2.Canny(img, 50, 120)
minlength = 50
maxLineGap = 10
lines = cv2.HoughLinesP(edge, 1, np.pi/180, 50, minlength, maxLineGap)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for x1, y1, x2, y2 in lines[0]:
    cv2.line(color, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("edge", edge)
cv2.imshow("lines", color)
cv2.waitKey()
cv2.destroyAllWindows()



