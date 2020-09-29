import cv2
import numpy as np


# 轮廓检测
# img = cv2.imread('D:/opencv_lib/image/coins.jpg', 0)
# color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# mask = cv2.Canny(img, 50, 100)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for i, contour in enumerate(contours):
#     cv2.drawContours(color, contours, i, (255, 0, 0), -1)
#
# cv2.imshow("dst", color)
# cv2.waitKey()
# cv2.destroyAllWindows()


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


# 对象测量
img = cv2.imread('D:/opencv_lib/image/geometry.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("hammer", gray)
# cv2.waitKey()
gray = cv2.GaussianBlur(gray, (7, 7), 0)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
contours, hireachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)  # 计算轮廓面积
    print(area)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    mm = cv2.moments(contour)
    cx = np.int(mm["m10"]/(mm["m00"]+0.001))    # 利用原点矩计算重心坐标，加上0.001防止分母为0
    cy = np.int(mm["m01"]/(mm["m00"]+0.001))
    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
    approxCurve = cv2.approxPolyDP(contour, 4, True)  # 多边形拟合函数，approxCurve.shape[0]中存储的是拟合图像所需的曲线数
    if approxCurve.shape[0] == 4:  # 检测矩形
        cv2.drawContours(img, contours, i, (255, 0, 0), 2)
    if approxCurve.shape[0] == 3:  # 检测三角
        cv2.drawContours(img, contours, i, (0, 255, 0), 2)
    if approxCurve.shape[0] > 4:
        cv2.drawContours(img, contours, i, (255, 255, 0), 2)


cv2.imshow("hammer", img)
cv2.waitKey()
cv2.destroyAllWindows()




