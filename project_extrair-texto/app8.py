from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import face_recognition

app = Flask(__name__)

# Criar o diretório para salvar as imagens cortadas, se não existir
save_folder = 'retangulos_cortados'
os.makedirs(save_folder, exist_ok=True)

@app.route('/cortar-retangulo', methods=['POST'])
def cortar_retangulo():
    # Receber a imagem
    imagem = request.files['imagem']
    
    # Carregar a imagem usando o OpenCV
    img = cv2.imdecode(np.frombuffer(imagem.read(), np.uint8), -1)
    
    # Encontrar as dimensões da imagem
    altura, largura = img.shape[:2]
    
    # Encontrar as localizações dos rostos na imagem usando face_recognition
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(imagem_rgb)
    
    # Se nenhum rosto for detectado, definir coordenadas de corte padrão
    if len(face_locations) == 0:
        x1, y1, x2, y2 = int(largura * 0.10), int(altura * 0.25), int(largura * 0.85), int(altura * 0.80)
    else:
        # Cortar um retângulo em volta do primeiro rosto detectado
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        # Fatores de multiplicação para cada lado do retângulo
        fator_esquerda = 3.5  # Fator de multiplicação para o lado esquerdo
        fator_direita = 8   # Fator de multiplicação para o lado direito
        # Aplicar fatores diferentes para cada lado do retângulo
        x1 = max(0, left - int(face_width * (fator_esquerda - 1) / 2))
        y1 = max(0, top - int(face_height * (fator_esquerda - 1) / 2))
        x2 = min(largura, right + int(face_width * (fator_direita - 1) / 2))
        y2 = min(altura, bottom + int(face_height * (fator_esquerda - 1) / 2))
    
    # Cortar a região da imagem de acordo com as coordenadas
    retangulo_cortado = img[y1:y2, x1:x2]
    
    # Salvar a imagem cortada na pasta especificada
    filename = os.path.join(save_folder, 'retangulo_cortado.jpg')
    cv2.imwrite(filename, retangulo_cortado)
    
    return jsonify({'retangulo_cortado_salvo': filename})

if __name__ == '__main__':
    app.run(debug=True, port='5008')

