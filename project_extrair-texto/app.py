import os
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

# Obtenha o diretório atual do script
script_dir = os.path.dirname(__file__)

@app.route('/extrair-texto', methods=['POST'])
def extrair_texto():
    imagem = request.files['imagem']
    texto_extraido = extrair_texto_da_imagem(imagem)
    return jsonify({'texto': texto_extraido})

def extrair_texto_da_imagem(imagem):
    # Carregar a imagem
    img = Image.open(imagem)
    
    # Extrair texto da imagem
    texto_extraido = pytesseract.image_to_string(img, lang='por')
    
    return texto_extraido

if __name__ == '__main__':
    app.run(debug=True)





