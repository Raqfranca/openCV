import os
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

@app.route('/text', methods=['POST'])
def extract_text():
    # Receber o arquivo de imagem
    image = request.files['image']
    
    # Extrair texto da imagem
    extracted_text = extract_text_from_image(image)
    
    # Retornar o texto extraído como resposta JSON
    return jsonify({'text': extracted_text})

def extract_text_from_image(image):
    img = Image.open(image)
    
    # Extrair texto da imagem usando Tesseract OCR
    extracted_text = pytesseract.image_to_string(img, lang='por')
    
    return extracted_text

if __name__ == '__main__':
    app.run(debug=True, port=8080)





