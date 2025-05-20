import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('best_model.keras')

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is None or len(faces) == 0:
        return None, frame

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    
    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    prediction = classifier.predict(roi, verbose=0)[0]
    label = emotion_labels[prediction.argmax()]
    return label, frame
