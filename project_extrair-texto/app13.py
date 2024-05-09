import os
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import pytesseract
import re

app = Flask(__name__)

@app.route('/text', methods=['POST'])
def extract_text():
    # Receber o arquivo de imagem
    image = request.files['image']
    
    # Extrair informações sobre orientação e detecção de script
    osd_info = get_orientation_and_script_detection(image)
    
    # Extrair texto e coordenadas das bounding boxes da imagem
    extracted_text, bounding_boxes = extract_text_and_boxes_from_image(image)
    
    # Salvar a imagem com as bounding boxes marcadas
    image_with_boxes_path = save_image_with_boxes(image, bounding_boxes)
    
    # Retornar todas as informações como resposta JSON
    return jsonify({
        'text': extracted_text,
        'bounding_boxes': bounding_boxes,
        'image_with_boxes_path': image_with_boxes_path,
        'osd_info': osd_info
    })

def get_orientation_and_script_detection(image):
    img = Image.open(image)
    
    # Obter informações sobre orientação e detecção de script
    osd_info = pytesseract.image_to_osd(img)
    
    return osd_info

def extract_text_and_boxes_from_image(image):
    img = Image.open(image)
    
    # Extrair texto da imagem usando Tesseract OCR
    extracted_text = pytesseract.image_to_string(img, lang='por')
    
    # Obter bounding boxes dos caracteres reconhecidos
    boxes_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Filtrar apenas as caixas delimitadoras com caracteres reconhecidos
    bounding_boxes_processed = []
    for i in range(len(boxes_data['text'])):
        if boxes_data['conf'][i] > 0:  # Considerar apenas caixas com confiança maior que 0
            char = boxes_data['text'][i]
            x, y, w, h = int(boxes_data['left'][i]), int(boxes_data['top'][i]), int(boxes_data['width'][i]), int(boxes_data['height'][i])
            bounding_boxes_processed.append({
                'char': char,
                'x_min': x,
                'y_min': y,
                'x_max': x + w,
                'y_max': y + h
            })
    
    return extracted_text, bounding_boxes_processed


def save_image_with_boxes(image, bounding_boxes):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    
    # Desenhar retângulos delimitando cada caractere reconhecido
    for box in bounding_boxes:
        x_min = box['x_min']
        y_min = img.height - box['y_max']  # Inverter a coordenada y
        x_max = box['x_max']
        y_max = img.height - box['y_min']  # Inverter a coordenada y
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red')
    
    # Salvar a imagem com as bounding boxes marcadas
    image_with_boxes_path = 'image_with_boxes.png'
    img.save(image_with_boxes_path)
    
    return image_with_boxes_path

if __name__ == '__main__':
    app.run(debug=True, port=8080)

