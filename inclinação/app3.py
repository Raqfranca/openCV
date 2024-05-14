import os
import numpy as np
import cv2
import face_recognition

from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

def calculate_angle(left_eye, right_eye):
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])
    angle_degrees = np.degrees(angle)
    return angle_degrees

@app.route('/detect_face_orientation', methods=['POST'])
def detect_face_orientation():
    # Verificar se a imagem foi enviada
    if 'image' not in request.json:
        return jsonify({'error': 'Missing image data'})

    # Decodificar a imagem base64
    try:
        image_base64 = request.json['image']
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'Failed to decode image data'})

    # Verificar se a imagem foi decodificada corretamente
    if image is None:
        return jsonify({'error': 'Failed to decode image data'})

    # Converter a imagem para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'})

    # Pegar as coordenadas do primeiro rosto detectado
    x, y, w, h = faces[0]
    face_image = image[y:y+h, x:x+w]

    # Converter a imagem da face para tons de cinza
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Detectar pontos-chave do rosto
    face_landmarks = face_recognition.face_landmarks(gray_face)

    if len(face_landmarks) == 0:
        return jsonify({'error': 'Could not detect face landmarks'})

    # Calcular a inclinação do rosto
    left_eye = face_landmarks[0]['left_eye']
    right_eye = face_landmarks[0]['right_eye']
    angle = calculate_angle(left_eye, right_eye)

    # Verificar se o rosto está inclinado
    if abs(angle) < 3:
        orientation = 'straight'
    else:
        orientation = 'tilted'

    # Aplicar a rotação à imagem inteira
    rows, cols, _ = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Salvar a imagem corrigida em uma pasta
    output_folder = 'images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'rotated_image.jpg')
    cv2.imwrite(output_path, rotated_image)

    return jsonify({'orientation': orientation, 'angle': angle, 'image_path': output_path})

if __name__ == '__main__':
    app.run(debug=True, port='8080')
