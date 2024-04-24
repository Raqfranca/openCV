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

def analyze_image(image_path):

    image = cv2.imread(image_path)

    # Foto  para cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aumentar o contraste
    alpha = 1.5 
    beta = 0   
    contrasted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Salvar a imagem após contraste
    contrasted_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contrasted_' + os.path.basename(image_path))
    cv2.imwrite(contrasted_image_path, contrasted_image)

    # Binarizar a imagem
    _, binary_image = cv2.threshold(contrasted_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Contar pixels brancos e escuros
    white_pixels = cv2.countNonZero(binary_image)
    black_pixels = binary_image.size - white_pixels

    return white_pixels, black_pixels, contrasted_image_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        if 'front_file' not in request.files or 'back_file' not in request.files:
            return render_template('index.html', message='Frente e parte de trás do documento são necessárias')

        front_file = request.files['front_file']
        back_file = request.files['back_file']

        if front_file.filename == '' or back_file.filename == '':
            return render_template('index.html', message='Frente e parte de trás do documento são necessárias')

        # Verifica se ambos os arquivos são permitidos
        if front_file and back_file and allowed_file(front_file.filename) and allowed_file(back_file.filename):
            front_filename = front_file.filename
            back_filename = back_file.filename

            front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_filename)
            back_filepath = os.path.join(app.config['UPLOAD_FOLDER'], back_filename)

            front_file.save(front_filepath)
            back_file.save(back_filepath)

            white_pixels_front, black_pixels_front, contrasted_image_path_front = analyze_image(front_filepath)
            white_pixels_back, black_pixels_back, contrasted_image_path_back = analyze_image(back_filepath)

            # Verifica a diferença na quantidade de pixels entre as fotos
            if black_pixels_front > black_pixels_back:
                message = 'Usuário informou corretamente a foto da frente.'
            else:
                message = 'Usuário não informou corretamente a foto da frente e de trás.'

            return render_template('result.html', 
                                   original_front_image=front_filename, 
                                   original_back_image=back_filename, 
                                   contrasted_front_image=os.path.basename(contrasted_image_path_front),
                                   contrasted_back_image=os.path.basename(contrasted_image_path_back),
                                   white_pixels_front=white_pixels_front, 
                                   black_pixels_front=black_pixels_front,
                                   white_pixels_back=white_pixels_back, 
                                   black_pixels_back=black_pixels_back,
                                   message=message)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)



