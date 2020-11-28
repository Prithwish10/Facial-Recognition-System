import cv2
import numpy as np
import sqlite3

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

def bioData(Id, name):

    connect = sqlite3.connect("FaceDataBase.db")

    cmd = "SELECT * FROM Faces WHERE ID = "+ str(Id)

    c = connect.execute(cmd)

    isRecordPresent = 0

    for row in c:
        isRecordPresent = 1

    if isRecordPresent == 1:
        cmd = "UPDATE Faces SET Name = "+ str(name)+ " WHERE ID =" + str(Id)

    else:
        cmd = "INSERT INTO Faces(ID, Name) Values(" +str(Id)+","+str(name)+")"

    connect.execute(cmd)

    connect.commit()

    connect.close()

id = input("Enter your id :")
name = input("Enter your name :")

bioData(id, name)

sample = 0

while(True):

    ret,img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:

        sample = sample + 1

        cv2.imwrite("DATASET/User." + str(id) + "." + str(sample) + ".jpg", gray[y:y+h,x:x+w])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.waitKey(100)

    cv2.imshow('FACE_DETECTED', img)

    cv2.waitKey(1)

    if(sample > 30):
        break

cam.release()
cv2.destroyAllWindows()
