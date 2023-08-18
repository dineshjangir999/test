from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os 

load_dotenv()
database_url = os.getenv('DATABASE_URL')
face_database = os.getenv("FACE_DATABASE")
registered_users = os.getenv("REGISTERED_USERS")
debug = os.getenv("DEBUG") 

app = Flask(__name__)
client = MongoClient(database_url)
db = client[face_database]
collection = db[registered_users]

def verify_landmarks(landmarks):
    required_landmark_points = ['nose_tip', 'chin', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    for point in required_landmark_points:
        if point not in landmarks:
            return False
    
    return True

def is_covering_nose_and_lips(landmarks):
    nose_tip = landmarks['nose_tip']
    top_lip = landmarks['top_lip']
    bottom_lip = landmarks['bottom_lip']
    
    nose_lips_region = [
        np.mean([nose_tip[3], top_lip[0]], axis=0),
        np.mean([nose_tip[2], bottom_lip[-1]], axis=0)
    ]
    
    return np.linalg.norm(nose_lips_region[1] - nose_lips_region[0]) < 10  # Adjust the threshold as needed
 
def recognize_face_data(uploaded_image):
    uploaded_face = face_recognition.load_image_file(uploaded_image)
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_face)

    if not uploaded_face_encodings:
        return {'message': 'No faces were detected in the uploaded image'}

    uploaded_face_landmarks = face_recognition.face_landmarks(uploaded_face)
    
    if not uploaded_face_landmarks:
        return {'message': 'No landmarks were detected on the face'}
     
    for landmarks in uploaded_face_landmarks: 
        if is_covering_nose_and_lips(landmarks):
            return {'message': 'Some part of face is covered.'}
        
        verify_landmarks(landmarks)
        if 'nose_tip' not in landmarks or 'top_lip' not in landmarks or 'bottom_lip' not in landmarks:
            return {'message': 'Nose and lips were not detected on the face'}
        
        if not is_eye_open(landmarks['left_eye']) or not is_eye_open(landmarks['right_eye']):
            return {'message': 'Please keep your eyes open for proper recognition'}

      
    if not uploaded_face_encodings:
        return {'message': 'Face encoding could not be generated'}
    else:
        uploaded_face_encoding = uploaded_face_encodings[0]

    for user in collection.find():
        registered_face_encoding = np.frombuffer(user['face_encoding'], dtype=np.float64)

        match_results = face_recognition.compare_faces(
            [registered_face_encoding], uploaded_face_encoding, tolerance=0.3)
        if any(match_results):
            recognized_user = user['name']
            return {'message': f'Hello, {recognized_user}!'}
    
    return False

def is_eye_open(eye_landmarks):
    eye_landmarks_np = np.array(eye_landmarks, dtype=np.float32)
    eye_aspect_ratio = (np.linalg.norm(eye_landmarks_np[1] - eye_landmarks_np[5]) +
                        np.linalg.norm(eye_landmarks_np[2] - eye_landmarks_np[4])) / (2.0 * np.linalg.norm(eye_landmarks_np[0] - eye_landmarks_np[3]))
    return eye_aspect_ratio > 0.3 # Adjust the threshold as needed

@app.route('/register', methods=['POST'])
def register_face():
    name = request.form.get('name')
    face_image = request.files.get('face_image')
    
    if not name or not face_image:
        return jsonify({'message': 'Name and face image are required'})

    new_face = face_recognition.load_image_file(face_image)
    new_face_encodings = face_recognition.face_encodings(new_face)

    if not new_face_encodings:
        return jsonify({'message': 'No face detected in the uploaded image'})

    new_face_encoding = new_face_encodings[0]
    new_face_encoding_bytes = new_face_encoding.tobytes()

    if not recognize_face_data(face_image):
        collection.insert_one({
            'name': name,
            'face_encoding': new_face_encoding_bytes
        })

        return jsonify({'message': 'Face registered successfully'})
    else:
         if recognize_face_data(face_image)['message']:
             return jsonify(recognize_face_data(face_image))
         return jsonify({'message': 'Face already registered'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    uploaded_image = request.files.get('face_image')
    
    if not uploaded_image:
        return jsonify({'message': 'No image uploaded for recognition'})

    data = recognize_face_data(uploaded_image)
    if not data:
        return jsonify({'message': 'not a recognized face'})
    return jsonify(data)

if __name__ == '__main__':
    if debug == "True":
        app.run(debug=True)
    else:
        app.run()
