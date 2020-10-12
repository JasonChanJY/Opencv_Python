import cv2
import numpy as np
from matplotlib import pyplot as plt

target = cv2.imread(r"D:\opencv_lib\image\abbeyroad1.jpg")
template = cv2.imread(r"D:\opencv_lib\image\template.png")

w, h = template.shape[:2]
#方法可以选cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED
#选择前两种时使用max_loc, 选择最后一种时用min_loc
result = cv2.matchTemplate(target, template, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
cv2.rectangle(target, (max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h), [255, 0, 0],
              2)
cv2.imshow("target", target)

cv2.waitKey()
cv2.destroyAllWindows()
