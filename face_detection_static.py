import cv2


def detect(filename):
    face_cascade = cv2.CascadeClassifier(r'D:\Pycharm_project\OpencvLib\venv\Lib'
                                         r'\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier(r'D:\Pycharm_project\OpencvLib\venv\Lib'
                                         r'\site-packages\cv2\data\haarcascade_eye.xml')
    img = cv2.imread(filename, 0)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 索引y与x反过来的原因在于第一个值代表y坐标，第二个值代表x坐标
        roi_img = img[y:y+h, x:x+w]
        # 最后一个参数限制检测范围，避免假阳性的出现
        eyes = eyes_cascade.detectMultiScale(roi_img, 1.3, 5, 0, (20, 20))
        for (x1, y1, w1, h1) in eyes:
            img = cv2.rectangle(img, (x1+x, y1+y), (x1+x+w1, y1+y+h1), (255, 255, 0), 1)

    cv2.imshow("face", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


filename = r"D:\Pycharm_project\OpencvLib\Image\gray_lena.jpg"
detect(filename)
