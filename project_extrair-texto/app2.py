from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

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
    
    # Redimensionar a imagem para 1080x1080
    img = cv2.resize(img, (1080, 1080))
    
    # Encontrar as dimensões da imagem redimensionada
    altura, largura = img.shape[:2]
    
    # Definir as coordenadas para o retângulo central
    # Vou definir um retângulo que cobre 70% da largura e altura da imagem
    # e está centralizado na imagem
    x1 = int(largura * 0.10)
    y1 = int(altura * 0.25)
    x2 = int(largura * 0.85)
    y2 = int(altura * 0.80)
    
    # Cortar a região central da imagem
    documento_cortado = img[y1:y2, x1:x2]
    
    # Salvar a imagem cortada na pasta especificada
    filename = os.path.join(save_folder, 'documento_cortado.jpg')
    cv2.imwrite(filename, documento_cortado)
    
    return jsonify({'documento_cortado_salvo': filename})

if __name__ == '__main__':
    app.run(debug=True, port='5002')




