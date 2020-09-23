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
# img1 = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\abbey_road1.jpg", 0)
# img2 = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\abbey_road.jpg", 0)
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.knnMatch(des1, des2, k=2)
# # matches = sorted(matches, key=lambda x: x.distance)
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
# plt.imshow(img3), plt.show()


# flnn匹配
# img1 = cv2.imread(r"D:\opencv_lib\image\abbeyroad1.jpg", 0)
# img2 = cv2.imread(r"D:\opencv_lib\image\abbeyroad.jpg", 0)
# img1 = cv2.resize(img1, (200, 200))
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN_INDEX_KDTREE = 0
# indexparams = dict(algorithm= FLANN_INDEX_KDTREE, trees = 5)
# searchparams = dict(checks = 50)

# flann = cv2.FlannBasedMatcher(indexparams, searchparams)
# matches = flann.knnMatch(des1, des1, k = 2)
# matchMask = [[0, 0] for i in range(len(matches))]

# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchMask[i] = [1, 0]

# drawparams = dict(matchColor = (0, 255, 0), singlePointColor = (255, 0, 0),
#                   matchesMask = matchMask, flags = 0)
# resultimage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **drawparams)

# # matches = sorted(matches, key=lambda x: x.distance)
# plt.imshow(resultimage), plt.show()


# flann单应性
img1 = cv2.imread(r"D:\opencv_lib\image\abbeyroad1.jpg", 0)
img2 = cv2.imread(r"D:\opencv_lib\image\abbeyroad.jpg", 0)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
indexparams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchparams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexparams, searchparams)
matches = flann.knnMatch(des1, des1, k=2)

min_match_count = 100
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
print(len(kp2))
if len(good) > min_match_count:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good[:1152]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good[:1152]]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print("not enough matches are found - %d/%d", (len(good), min_match_count))
    matchMask = None

drawparams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                  matchesMask=matchMask, flags=2)
resultimage = cv2.drawMatches(img1, kp1, img2, kp2, good[:1152], None, **drawparams)
plt.imshow(resultimage, "gray"), plt.show()
