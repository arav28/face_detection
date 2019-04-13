import numpy as np
import cv2
video = cv2.VideoCapture(0)

while True:
    count=0
    (ret, frame) = video.read()
    if not ret:
        break 
    face_cascade = cv2.CascadeClassifier('/home/aruve/haarcascade_frontalface_default.xml')
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        count+=1
    cv2.putText(frame, "Number of faces detected: " + str(len(faces)), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.7,  (255,255,255), 1)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cv2.destroyAllWindows()
