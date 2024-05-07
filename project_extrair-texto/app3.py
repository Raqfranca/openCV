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
    
    # Encontrar as dimensões da imagem
    altura, largura = img.shape[:2]
    
    # Verificar se a imagem está na horizontal ou na vertical
    orientacao = "horizontal" if largura > altura else "vertical"
    
    # Encontrar as localizações dos rostos na imagem usando face_recognition
    imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(imagem_rgb)
    
    # Se não houver rostos detectados, definir coordenadas de corte padrão
    if len(face_locations) == 0:
        x1, y1, x2, y2 = int(largura * 0.10), int(altura * 0.25), int(largura * 0.85), int(altura * 0.80)
    else:
        # Apenas cortar em volta do primeiro rosto detectado
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
    
    # Cortar a região da imagem de acordo com as coordenadas
    documento_cortado = img[y1:y2, x1:x2]
    
    # Salvar a imagem cortada na pasta especificada
    filename = os.path.join(save_folder, 'documento_cortado.jpg')
    cv2.imwrite(filename, documento_cortado)
    
    return jsonify({'documento_cortado_salvo': filename, 'orientacao': orientacao})

if __name__ == '__main__':
    app.run(debug=True)


