import cv2
import numpy as np


# 轮廓框
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


# 线检测
# img = cv2.imread(r'D:\opencv_lib\image\thunder.jpg', 0)
# img = cv2.resize(img, (300, 300))
# # img = np.zeros((200, 200), dtype=np.uint8)  #用方块测试
# # img[50:150, 50:150] = 255
# edge = cv2.Canny(img, 100, 200)
# minlength = 5
# maxLineGap = 5
# lines = cv2.HoughLinesP(edge, 1, np.pi/180, 10, minlength, maxLineGap)
# color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(color, (x1, y1), (x2, y2), (255, 0, 0), 2)

# cv2.imshow("edge", edge)
# cv2.imshow("lines", color)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 圆检测
# coins = cv2.imread("D:\opencv_lib\image\coins.jpg")
# gray_coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
# gray_coins = cv2.medianBlur(gray_coins, 5)
# circles = cv2.HoughCircles(gray_coins, cv2.HOUGH_GRADIENT, 1, 120,
#                            param1=100, param2=70, minRadius=0, maxRadius=0,)
# circles = np.uint16(np.around(circles))
#
# for i in circles[0, :]:
#     cv2.circle(coins, (i[0], i[1]), i[2], (255, 0, 0), 2)
#     cv2.circle(coins, (i[0], i[1]), 2, (0, 255, 255), 2)
#
# cv2.imshow("coins", coins)
# cv2.waitKey()
# cv2.destroyAllWindows()



