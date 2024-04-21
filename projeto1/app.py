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

def detect_contours(image_path):
    # Carregar a imagem
    imagem = cv2.imread(image_path)

    # Converter a imagem para escala de cinza
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro de suavização (opcional)
    imagem_blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

    # Detectar as bordas na imagem usando Canny
    bordas = cv2.Canny(imagem_blur, 30, 150)

    # Encontrar os contornos na imagem
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos na imagem original
    cv2.drawContours(imagem, contornos, -1, (0, 255, 0), 3)

    # Salvar a imagem com os contornos detectados
    imagem_contornos = os.path.join(app.config['UPLOAD_FOLDER'], 'contornos.jpg')
    cv2.imwrite(imagem_contornos, imagem)

    return imagem_contornos

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

            imagem_contornos = detect_contours(filepath)

            return render_template('result.html', original_image=filename, contoured_image='contornos.jpg')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
