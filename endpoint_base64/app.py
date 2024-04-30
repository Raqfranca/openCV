from flask import Flask, request, jsonify
import os
import face_recognition
import cv2
import numpy as np
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_faces(image1_data, image2_data):
    # Decodificar as imagens base64
    image1_decoded = base64.b64decode(image1_data)
    image2_decoded = base64.b64decode(image2_data)

    # Converter para numpy arrays
    nparr1 = np.frombuffer(image1_decoded, np.uint8)
    nparr2 = np.frombuffer(image2_decoded, np.uint8)

    # Decodificar as imagens OpenCV
    image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # Obter os encodings faciais
    encoding1 = face_recognition.face_encodings(image1)[0]
    encoding2 = face_recognition.face_encodings(image2)[0]

    # Calcular a distância euclidiana entre os encodings
    euclidean_distance = face_recognition.face_distance([encoding1], encoding2)[0]

    return euclidean_distance

# Define as rotas da API
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json

    if 'document_image' not in data or 'selfie_image' not in data:
        return jsonify({'error': 'Missing image data'})

    document_image = data['document_image']
    selfie_image = data['selfie_image']

    if not document_image or not selfie_image:
        return jsonify({'error': 'Invalid image data'})

    # Comparar as faces nas duas imagens
    euclidean_distance_threshold = 0.55  # Limite de distância euclidiana
    euclidean_distance = compare_faces(document_image, selfie_image)

    same_person = euclidean_distance <= euclidean_distance_threshold

    return jsonify({'same_person': same_person, 'euclidean_distance': euclidean_distance.item()})

# Rota para verificar a saúde da API
@app.route('/health', methods=['GET'])
def health_check():
    return 'API is healthy'

if __name__ == '__main__':
    app.run(debug=True)
