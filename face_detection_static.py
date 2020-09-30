import cv2


def detect(img):
    face_cascade = cv2.CascadeClassifier('D:/pycharmfile/venv/Lib/site-packages/cv2/data'
                                         '/haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('D:/pycharmfile/venv/Lib/site-packages/cv2/data'
                                         '/haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 索引y与x反过来的原因在于第一个值代表y坐标，第二个值代表x坐标
        roi_img = img[y:y+h, x:x+w]
        # 最后一个参数限制检测范围，避免假阳性的出现
        eyes = eyes_cascade.detectMultiScale(roi_img, 1.1, 5, 0, (20, 20))
        for (x1, y1, w1, h1) in eyes:
            img = cv2.rectangle(img, (x1+x, y1+y), (x1+x+w1, y1+y+h1), (255, 255, 0), 1)

    cv2.imshow("result", img)


# img = cv2.imread("D:/opencv_lib/image/face.jpg")
# detect(img)

# 摄像机读取
capture = cv2.VideoCapture(0)
cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
while (True):
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    detect(frame)
    c = cv2.waitKey(10)
    if c == 32:
        break
 
