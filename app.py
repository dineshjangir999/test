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


def recognize_face_data(uploaded_image):
    uploaded_face = face_recognition.load_image_file(uploaded_image)
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_face)
    for uploaded_face_encoding in uploaded_face_encodings:
        for user in collection.find():
            registered_face_encoding = np.fromstring(
                user['face_encoding'], dtype=np.float64)

            match_results = face_recognition.compare_faces(
                [registered_face_encoding], uploaded_face_encoding)
            if any(match_results):
                recognized_user = user['name']
                return jsonify({'message': recognized_user})

    return False


@app.route('/register', methods=['POST'])
def register_face():
    name = request.form['name']
    face_image = request.files['face_image']
    new_face = face_recognition.load_image_file(face_image)
    new_face_encoding = face_recognition.face_encodings(new_face)
    if not new_face_encoding:
        return jsonify({'message': 'Face Not Found'})
    elif len(new_face_encoding)> 1:
        return jsonify({'message': 'Multiple Users Found'})
    else: 
        new_face_encoding = new_face_encoding[0]

    if not recognize_face_data(face_image):
        new_face_encoding_str = new_face_encoding.tostring()
        collection.insert_one({
            'name': name,
            'face_encoding': new_face_encoding_str
        })

        return jsonify({'message': 'Face registered successfully'})
    else:
        return jsonify({'message': 'Face already registered'})


@app.route('/recognize', methods=['POST'])
def recognize_face():
    uploaded_image = request.files['face_image']
    data = recognize_face_data(uploaded_image)
    if not data:
        return jsonify({'message': 'no data available'})
    return data


if __name__ == '__main__':
    if debug == "True":
        app.run(debug=True)
    else:
        app.run()
