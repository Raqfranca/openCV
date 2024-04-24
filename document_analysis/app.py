from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_front_rg(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     # Equalizar o histograma para aumentar o contraste
    gray_equalized = cv2.equalizeHist(gray)

     # Detectar bordas na imagem usando Canny
    edges = cv2.Canny(gray_equalized, 30, 150)

    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Verificar se há uma região retangular no canto superior direito (onde geralmente está a foto)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > image.shape[1] / 2 and h > image.shape[0] / 2:
            return True

    # Verificar se há áreas de texto na parte inferior (onde geralmente estão as informações)
    bottom_region = gray_equalized[int(0.75 * gray_equalized.shape[0]):, :]
    _, bottom_thresh = cv2.threshold(bottom_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bottom_contours, _ = cv2.findContours(bottom_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in bottom_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0.5 * bottom_region.shape[1] and h > 0.1 * bottom_region.shape[0]:
            return True

    return False

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='Nenhum arquivo enviado')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='Nenhum arquivo selecionado')

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            is_front_rg = detect_front_rg(filepath)

            if is_front_rg:
                message = 'A imagem parece ser a frente de um RG.'
            else:
                message = 'A imagem não parece ser a frente de um RG.'

            return render_template('result.html', original_image=filename, gray_image='gray_' + filename, message=message)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
