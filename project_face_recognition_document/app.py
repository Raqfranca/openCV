from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import face_recognition

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_image(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) > 0:
        return True  # Rosto detectado
    else:
        return False  # Nenhum rosto detectado

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'front_file' not in request.files or 'back_file' not in request.files:
            return render_template('index.html', message='Frente e parte de trás do documento são necessárias')

        front_file = request.files['front_file']
        back_file = request.files['back_file']

        if front_file.filename == '' or back_file.filename == '':
            return render_template('index.html', message='Frente e parte de trás do documento são necessárias')

        if front_file and allowed_file(front_file.filename) and back_file and allowed_file(back_file.filename):
            front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_file.filename)
            back_filepath = os.path.join(app.config['UPLOAD_FOLDER'], back_file.filename)

            front_file.save(front_filepath)
            back_file.save(back_filepath)

            front_has_face = analyze_image(front_filepath)
            back_has_face = analyze_image(back_filepath)

            return render_template('result.html', front_has_face=front_has_face, back_has_face=back_has_face)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)




