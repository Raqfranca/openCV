import cv2
import numpy as np
import pytesseract
from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_text():
    try:
        image_base64 = request.json['image']
        
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        img = cv2.imdecode(image_array, -1)
        
        cropped_document = crop_background(img)
        
        extracted_text = extract_text_from_image(cropped_document)
        
        return jsonify({'text': extracted_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def crop_background(image):
    try:
        height, width = image.shape[:2]
        
        x1 = int(width * 0.10)
        y1 = int(height * 0.25)
        x2 = int(width * 0.85)
        y2 = int(height * 0.80)
        
        cropped_document = image[y1:y2, x1:x2]
        
        return cropped_document
    
    except Exception as e:
        print("Error cropping image:", e)
        return None

def extract_text_from_image(image):
    try:
        extracted_text = pytesseract.image_to_string(image, lang='por')
        
        if not extracted_text.strip():
            return "Unable to extract information from this document."
        
        return extracted_text
    
    except Exception as e:
        print("Error extracting text:", e)
        return None

if __name__ == '__main__':
    app.run(debug=True, port='5007')

