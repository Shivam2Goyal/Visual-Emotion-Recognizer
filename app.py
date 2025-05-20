from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import base64
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load face detector and emotion model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('best_model.keras')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# Global camera state
camera_on = False

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

# Video frame generator for streaming
def generate_frames():
    global camera_on
    cap = cv2.VideoCapture(0)

    while True:
        if not camera_on:
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi, verbose=0)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera')
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    return ('', 204)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_data = data['image'].split(',')[1]  # Remove base64 header
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Image decoding failed'}), 400

    emotion, _ = detect_emotion(img)
    if emotion is None:
        emotion = "No face detected"

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
