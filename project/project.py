import cv2
from pygame import mixer
import pygame
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

mixer.init()
mixer.music.load('sound.mp3')

   

webcam = cv2.VideoCapture(0)

face_cascade = 'model/haarcascade_frontalface_default.xml'
model = load_model('model/mask.h5')
np.set_printoptions(suppress=True) #ทำให้ค่า predict เป็นทศนิยม
while True:
    success, bgr_image = webcam.read() 
    #เปลี่ยนรูปเป็นขาวดำเพื่อให้ cascade อ่านค่าได้
    bw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    face_classifier = cv2.CascadeClassifier(face_cascade)
    faces = face_classifier.detectMultiScale(bw_image)
    cv2.putText(bgr_image, 'Press "Q" to exit', (350, 475), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2, cv2.LINE_4)
    for face in faces:
        x, y, w, h = face
        crop_rgb_image = Image.fromarray(rgb_image[y:y+h,x:x+w])

    
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        
        image = crop_rgb_image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        
        data[0] = normalized_image_array
        prediction = model.predict(data)
        # print(prediction)
        # prediction[0][0] = masked, [0][1] = nonmasked
        masked = prediction[0][0] > prediction[0][1]
        unmasked = prediction[0][0] < prediction[0][1]
        if masked:
            cv2.putText(bgr_image, 'Masked', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0),2)
            cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
        elif unmasked:
            cv2.putText(bgr_image, 'Unmasked', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255),2)
            cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 0, 255), 5)
            cv2.putText(bgr_image, 'wear a mask!!', (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255),2, cv2.LINE_4)
            if not mixer.music.get_busy():
                mixer.music.play()
                
                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("mask detection", bgr_image)
    cv2.waitKey(1)
webcam.release()
cv2.destroyAllWindows()
