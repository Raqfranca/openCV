from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_text():
    # Receber a imagem do documento
    image = request.files['image']
    
    # Função que corta a imagem
    cropped_document = crop_background(image)
    
    # Função vai extrair o texto da imagem
    extracted_text = extract_text_from_image(cropped_document)
    
    #Retorno da extração do texto em json
    return jsonify({'text': extracted_text})

def crop_background(image):
    # Load the image using OpenCV
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)
    
    # Encontrar as dimensões da imagem
    height, width = img.shape[:2]
    
    # Definir as coordenadas para o retângulo central
    x1 = int(width * 0.10)
    y1 = int(height * 0.25)
    x2 = int(width * 0.85)
    y2 = int(height * 0.80)
    
    # Cortar a região central da imagem
    cropped_document = img[y1:y2, x1:x2]
    
    return cropped_document

def extract_text_from_image(image):
    # Extrair texto da imagem usando Tesseract OCR
    extracted_text = pytesseract.image_to_string(image, lang='por')
    
    #Check if the extracted text is empty or contains only whitespace characters
    if not extracted_text.strip():
        return "Unable to extract information from this document."
    
    return extracted_text

if __name__ == '__main__':
    app.run(debug=True, port='5006')