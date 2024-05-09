from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import pytesseract
import re

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Receber o arquivo de imagem
    image = request.files['image']
    
    # Extrair informações sobre orientação e detecção de scripts
    osd_info, area_of_interest = get_osd_and_area_of_interest(image)
    
    # Salvar a imagem com a área de interesse marcada
    image_with_area_of_interest_path = save_image_with_area_of_interest(image, area_of_interest)
    
    # Retornar todas as informações como resposta JSON
    return jsonify({
        'osd_info': osd_info,
        'image_with_area_of_interest_path': image_with_area_of_interest_path
    })

def get_osd_and_area_of_interest(image):
    img = Image.open(image)
    
    # Obter informações sobre orientação e detecção de scripts
    osd_info = pytesseract.image_to_osd(img)
    
    # Definir manualmente a área de interesse
    area_of_interest = (80, 10, 725, 435)  # Exemplo: área de interesse definida como um retângulo de (x_min, y_min, x_max, y_max)
    
    return osd_info, area_of_interest

def save_image_with_area_of_interest(image, area_of_interest):
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    
    # Desenhar um retângulo na área de interesse
    draw.rectangle(area_of_interest, outline='red')
    
    # Salvar a imagem com a área de interesse marcada
    image_with_area_of_interest_path = 'image_with_area_of_interest.png'
    img.save(image_with_area_of_interest_path)
    
    return image_with_area_of_interest_path

if __name__ == '__main__':
    app.run(debug=True, port=8080)

