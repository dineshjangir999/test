from flask import Flask, request, jsonify,abort
import face_recognition
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import os 
from helper import  recognize_face_data

load_dotenv()
database_url = os.getenv('DATABASE_URL')
face_database = os.getenv("FACE_DATABASE")
registered_users = os.getenv("REGISTERED_USERS")
debug = os.getenv("DEBUG") 

app = Flask(__name__)
client = MongoClient(database_url)
db = client[face_database]
collection = db[registered_users]


@app.route('/register', methods=['POST'])
def register_face():
    name = request.form.get('name')
    face_image = request.files.get('face_image')
    
    if not name or not face_image:
        return abort(400,{'message': 'Name and face image are required'})

    new_face = face_recognition.load_image_file(face_image)
    new_face_encodings = face_recognition.face_encodings(new_face)

    if not new_face_encodings:
        return abort(404,{'message': 'No face detected in the uploaded image'})

    new_face_encoding = new_face_encodings[0]
    new_face_encoding_bytes = new_face_encoding.tobytes()

    face_result =recognize_face_data(face_image, True)
    
    if not face_result:
        collection.insert_one({
            'name': name,
            'face_encoding': new_face_encoding_bytes
        })

        return jsonify({'message': 'Face registered successfully'})
    else:
         if face_result['message']:
             return jsonify(face_result)
         return abort(409,{'message': 'Face already registered'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    uploaded_image = request.files.get('face_image')
    
    if not uploaded_image:
        return abort(400,{'message': 'No image uploaded for recognition'})

    data = recognize_face_data(uploaded_image)
    if not data:
        return abort(404,{'message': 'not a recognized face'})
    return jsonify(data)

if __name__ == '__main__':
    if debug == "True":
        app.run(debug=True)
    else:
        app.run()
