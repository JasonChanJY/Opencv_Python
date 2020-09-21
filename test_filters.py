import cv2
import numpy as np
import filters


img = cv2.imread(r"D:\Pycharm_project\OpencvLib\Image\gray_lena.jpg")
img_sharpen = filters.EmbossFilter()
img_sharpen.apply(img, img)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
