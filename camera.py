
import cv2
from model import FEM
import numpy as np

# detecing faces
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# loading the model from model.py 
model = FEM('model.json', 'model_weights_3.h5')
# font for the predicted expression
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    # initializing the videocamera
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, fr = self.video.read()
        # converting frame to a grayscale frame/image for the model
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        # detecing faces    
        faces = face.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_exp(roi[np.newaxis, :, :, np.newaxis])
            # writing the predicted expression on top of the face
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            # drawing a rectangle on the detected face
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
        # encoding the frame after the prediction and displaying in a jpg foramt
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()