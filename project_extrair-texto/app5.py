from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import face_recognition

app = Flask(__name__)

# Criar o diretório para salvar as imagens cortadas, se não existir
save_folder = 'documentos_cortados'
os.makedirs(save_folder, exist_ok=True)

@app.route('/cortar-fundo', methods=['POST'])
def cortar_fundo():
    # Receber a imagem do documento
    imagem = request.files['imagem']
    
    # Carregar a imagem usando o OpenCV
    img = cv2.imdecode(np.frombuffer(imagem.read(), np.uint8), -1)
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar a binarização adaptativa da imagem
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Encontrar os contornos na imagem binarizada
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcular a área de cada contorno
    areas = [cv2.contourArea(contour) for contour in contours]
    
    # Encontrar o contorno com a maior área (presumindo que seja o documento)
    max_index = np.argmax(areas)
    max_contour = contours[max_index]
    
    # Encontrar os limites do retângulo que contém o documento
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Cortar a região do documento da imagem cortada
    documento_cortado = img[y:y+h, x:x+w]
    
    # Salvar a imagem cortada na pasta especificada
    filename = os.path.join(save_folder, 'documento_cortado.jpg')
    cv2.imwrite(filename, documento_cortado)
    
    # Retornar as dimensões do contorno (largura e altura) como parte da resposta JSON
    return jsonify({'documento_cortado_salvo': filename, 'width': w, 'height': h})

if __name__ == '__main__':
    app.run(debug=True, port='5005')
