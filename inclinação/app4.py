import base64
import numpy as np
import cv2
import pytesseract
import face_recognition
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

def calculate_angle(left_eye, right_eye):
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
    angle_degrees = np.degrees(angle)
    return angle_degrees

def correct_image_orientation(image):
    # Converter a imagem para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return image

    # Pegar as coordenadas do primeiro rosto detectado
    x, y, w, h = faces[0]
    face_image = image[y:y+h, x:x+w]

    # Converter a imagem da face para tons de cinza
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Detectar pontos-chave do rosto
    face_landmarks = face_recognition.face_landmarks(gray_face)

    if len(face_landmarks) == 0:
        return image

    # Calcular a inclinação do rosto
    left_eye = face_landmarks[0]['left_eye']
    right_eye = face_landmarks[0]['right_eye']
    angle = calculate_angle(left_eye, right_eye)

    # Aplicar a rotação à imagem inteira
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return rotated_image

def crop_background(image):
    try:
        height, width = image.shape[:2]
        
        # Localizar a face na imagem 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if len(face_locations) == 0:
            x1 = int(width * 0.10)
            y1 = int(height * 0.25)
            x2 = int(width * 0.85)
            y2 = int(height * 0.80)
        else:
            # Corte se tiver face
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
          
            left_factor = 3.5  # Fator para o lado esquerdo 
            right_factor = 8   # Fator para o lado direito 
    
            x1 = max(0, left - int(face_width * (left_factor - 1) / 2))
            y1 = max(0, top - int(face_height * (left_factor - 1) / 2))
            x2 = min(width, right + int(face_width * (right_factor - 1) / 2))
            y2 = min(height, bottom + int(face_height * (left_factor - 1) / 2))
        
        # Corte da imagem
        cropped_document = image[y1:y2, x1:x2]
        
        return cropped_document
    
    except Exception as e:
        print("Error cropping image:", e)
        return None

def extract_text_from_image(image, output_path):
    try:
        # Obter as dimensões da imagem
        height, width, _ = image.shape
        
        # Definir a região de interesse (ROI) como toda a imagem
        roi = (0, 0, width, height)
        
        # Obter a região de interesse da imagem
        roi_image = image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
        # Extrair texto da região de interesse da imagem usando Tesseract OCR
        extracted_text = pytesseract.image_to_string(roi_image, lang='por')
        
        if not extracted_text.strip():
            return "Unable to extract information from this document."
        
        # Não é mais necessário salvar a imagem corrigida
        # cv2.imwrite(output_path, image)
        
        return extracted_text
    
    except Exception as e:
        print("Error extracting text:", e)
        return None

@app.route('/extract', methods=['POST'])
def extract_text():
    try:
        image_base64 = request.json['image']
        
        # Decodificar a imagem base64
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        # Decodificar a imagem para um formato OpenCV
        img = cv2.imdecode(image_array, -1)
        
        # Corrigir a inclinação da imagem
        rotated_image = correct_image_orientation(img)
        
        # Recortar o documento
        cropped_document = crop_background(rotated_image)
        
        # Extrair texto da imagem
        extracted_text = extract_text_from_image(cropped_document, "")
        
        return jsonify({'text': extracted_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port='8080')
