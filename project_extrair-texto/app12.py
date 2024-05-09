from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64
import os
from PIL import Image
import pytesseract

app = Flask(__name__)

# Rota de comparar imagens
def compare_faces(image1_data, image2_data):
    try:
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
    except Exception as e:
        return str(e)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()

        if 'document_image' not in data or 'selfie_image' not in data:
            return jsonify({'error': 'Missing image data'}), 400

        document_image = data['document_image']
        selfie_image = data['selfie_image']

        # Comparar as faces nas duas imagens
        euclidean_distance_threshold = 0.55
        euclidean_distance = compare_faces(document_image, selfie_image)

        same_person = euclidean_distance <= euclidean_distance_threshold

        return jsonify({'same_person': bool(same_person), 'dist': float(euclidean_distance)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota de extrair texto
@app.route('/extract', methods=['POST'])
def extract_text():
    try:
        image_base64 = request.json['image']
        
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        img = cv2.imdecode(image_array, -1)
        
        cropped_document = crop_background(img)
        
        extracted_text = extract_text_from_image(cropped_document)
        
        return jsonify({'text': extracted_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def crop_background(image):
    try:
        height, width = image.shape[:2]
        
        # Find face locations in the image using face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        # If no face is detected, set default crop coordinates
        if len(face_locations) == 0:
            x1 = int(width * 0.10)
            y1 = int(height * 0.25)
            x2 = int(width * 0.85)
            y2 = int(height * 0.80)
        else:
            # Crop a rectangle around the first detected face
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            # Multiplication factors for each side of the rectangle
            left_factor = 3.5  # Multiplication factor for the left side
            right_factor = 8   # Multiplication factor for the right side
            # Apply different factors for each side of the rectangle
            x1 = max(0, left - int(face_width * (left_factor - 1) / 2))
            y1 = max(0, top - int(face_height * (left_factor - 1) / 2))
            x2 = min(width, right + int(face_width * (right_factor - 1) / 2))
            y2 = min(height, bottom + int(face_height * (left_factor - 1) / 2))
        
        # Calculate the size of the cropped region
        cropped_height = y2 - y1
        cropped_width = x2 - x1
        
        # Crop the document
        cropped_document = image[y1:y2, x1:x2]
        
        return cropped_document
    
    except Exception as e:
        print("Error cropping image:", e)
        return None

def extract_text_from_image(image):
    try:
        extracted_text = pytesseract.image_to_string(image, lang='por')
        
        if not extracted_text.strip():
            return "Unable to extract information from this document."
        
        return extracted_text
    
    except Exception as e:
        print("Error extracting text:", e)
        return None

# Rota para verificar a saúde da API
@app.route('/health', methods=['GET'])
def health_check():
    return 'API is healthy'

if __name__ == '__main__':
    app.run(debug=True, port=8080)