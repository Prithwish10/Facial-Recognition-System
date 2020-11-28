import cv2
import numpy as np
import sqlite3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('Trained_DataSet/trainingData.yml')

cam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

fontScale = 1;

colour = (0, 255, 0)

thickness = 2

def getProfile(Id):

    connect = sqlite3.connect("FaceDataBase.db")

    cmd = "SELECT * FROM Faces WHERE ID ="+ str(Id)

    c = connect.execute(cmd)

    profile = None

    for row in c:
        profile = row

    connect.close()
    
    return profile
    
while(True):

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ID, conf = recognizer.predict(gray[y:y+h, x:x+w])

        profile = getProfile(ID)

        if profile != None:
            cv2.putText(img, profile[1], (x, y+h+20), font, fontScale, colour, thickness, cv2.LINE_AA)
            cv2.putText(img, profile[2], (x, y+h+50), font, fontScale, colour, thickness, cv2.LINE_AA)
            cv2.putText(img, profile[3], (x, y+h+80), font, fontScale, colour, thickness, cv2.LINE_AA)
            
    cv2.imshow("Face_Detected",img)

    if(cv2.waitKey(1) == ord('q')):
        break;

cam.release()

cv2.destroyAllWindows()

        
