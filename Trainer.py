import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'DATASET'

def getImageWithID(path):

    ImagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for ImagePath in ImagePaths:

        faceImg = Image.open(ImagePath).convert('L')

        faceNp = np.array(faceImg, 'uint8')

        ID = int(os.path.split(ImagePath)[-1].split('.')[1])

        faces.append(faceNp)

        IDs.append(ID)

        cv2.imshow('Training Images', faceNp)

        cv2.waitKey(10)

    return faces, np.array(IDs)

faces, Ids = getImageWithID(path)

recognizer.train(faces, Ids)

recognizer.save('Trained_DataSet/trainingData.yml')

cv2.destroyAllWindows()

