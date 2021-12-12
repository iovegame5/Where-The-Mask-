
import cv2

webcam = cv2.VideoCapture(0)
face_cascade = 'E:\Desktop\PSIT\project\haarcascade_frontalface_default.xml'
count = 0
while True:
    success, bgr_image = webcam.read()
    img_org = bgr_image.copy()
    bw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(bw_image)
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (100, 255, 0), 5)
        cv2.imwrite('E:/Desktop/PSIT/project/masked/masked_%d.jpg' %(count), img_org[x:x+w,y:y+h])
        count+= 1
    cv2.imshow("Faces found", bgr_image)
    cv2.waitKey(1)
    cv2.destroyAllWindows