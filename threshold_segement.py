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
coins = cv2.imread("D:/opencv_lib/image/coins1.jpg")
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)
ret_1, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
plt.subplot(321), plt.imshow(thresh), plt.title("threshold")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
sure_bg = cv2.dilate(opening, kernel, iterations=2)
plt.subplot(322), plt.imshow(sure_bg), plt.title("background")  # 蓝色区域为背景

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # cv2.DIST_L2表示欧几里得距离， cv2.DIST_L1表示棋盘格距离， 5为掩模大小
ret_2, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
plt.subplot(323), plt.imshow(sure_fg), plt.title("frontground")  # 黄色区域为前景

unknown = np.subtract(sure_bg, sure_fg)
plt.subplot(324), plt.imshow(unknown), plt.title("unknown area")

ret_3, markers = cv2.connectedComponents(sure_fg)  # markers 表示不同的连通区域，用不同颜色表示。如有两个连通区域，
markers = markers + 1                              # 则markers中有三种值，分别为0、1、2，其中0表示背景
markers[unknown==255] = 0
plt.subplot(325), plt.imshow(markers), plt.title("markers")
markers = cv2.watershed(coins, markers)  # markers=-1 表示边界，markers=1 表示背景（不为0的原因应该是由于前面对markers进行了加1操作），
coins[markers==-1] = [0, 255, 0]         # markers=2 表示第一个硬币，markers=3 表示第二个硬币。。。

plt.show()
cv2.imshow("dist", coins)
cv2.waitKey()
cv2.destroyAllWindows()
