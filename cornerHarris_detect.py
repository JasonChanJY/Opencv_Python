import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


# Harris角点检测
# img = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\chess_board.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# img[dst > 0.01*dst.max()] = [0, 0, 255]
# while (True):
#     cv2.imshow("corner", img)
#     if cv2.waitKey(1000) & 0xff == ord(" "):
#         break

# cv2.destroyAllWindows()


# SIFT/SURF特征提取
# def fd(algorithm):
#     if algorithm == "SIFT":
#         return cv2.xfeatures2d.SIFT_create()
#     if algorithm == "SURF":
#         value = input("Please input the Hessian value: ")
#         value = float(value)
#         return cv2.xfeatures2d.SURF_create(value)
#
#
# img_path = r'D:\Pycharm_project\OpencvLib\Image\Varese.jpg'
# img = cv2.imread(img_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# algorithm = input("Please input which method you want to use(SIFT or SURF): ")
# algorithm = algorithm.upper()
# fd = fd(algorithm)
# keypoints, descriptor = fd.detectAndCompute(gray, None)
#
# img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
#                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=[255, 0, 255])
#
# cv2.imshow("outcome", img)
# cv2.waitKey()
# cv2.destroyAllWindows()


# ORB特征匹配
img1 = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\abbey_road1.jpg", 0)
img2 = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\abbey_road.jpg", 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(des1, des2, k=2)
# matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
plt.imshow(img3), plt.show()
