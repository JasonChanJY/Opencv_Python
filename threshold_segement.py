import cv2
import numpy as np
from matplotlib import pyplot as plt

###GrabCut
# coins = cv2.imread("D:\opencv_lib\image\coins.jpg")  #转为灰度图时，运行会报错
# mask = np.zeros(coins.shape[:2], np.uint8)
# bgdModel = np.zeros((1,65), np.float64)
# fgdModel = np.zeros((1,65), np.float64)
#
# rect = (50, 200, 700, 300)
# cv2.grabCut(coins, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
# coins = coins*mask2[:, :, np.newaxis]
#
# plt.subplot(121), plt.imshow(coins)
# plt.title("grabcut"), plt.xticks([]), plt.yticks([])
# plt.show()

###分水岭算法
coins = cv2.imread("D:\opencv_lib\image\coins.jpg")
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

ret_1, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
plt.subplot(321)
plt.imshow(thresh)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
plt.subplot(322)
plt.imshow(sure_bg)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret_2, sure_fg = cv2.threshold(dist_transform, 0.06*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
plt.subplot(323)
plt.imshow(sure_fg)
unknown = np.subtract(sure_bg, sure_fg)
plt.subplot(324)
plt.imshow(unknown)
ret_3, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(coins, markers)
coins[markers==-1] = [0, 255, 0]

plt.subplot(325)
plt.imshow(coins)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()