from flask import Flask, render_template, request, send_from_directory
import os
import face_recognition
import cv2

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
    num_faces = len(face_locations)
    if num_faces > 0:
        return True, num_faces, face_locations  # Rosto detectado
    else:
        return False, num_faces, []  # Nenhum rosto detectado

def draw_face_rectangles(image_path, face_locations):
    image = cv2.imread(image_path)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'front_file' not in request.files or 'back_file' not in request.files or 'selfie_file' not in request.files:
            return render_template('index.html', message='Falta dados')

        front_file = request.files['front_file']
        back_file = request.files['back_file']
        selfie_file = request.files['selfie_file']

        if front_file.filename == '' or back_file.filename == '' or selfie_file.filename == '':
            return render_template('index.html', message='Falta dados')

        if front_file and allowed_file(front_file.filename) and back_file and allowed_file(back_file.filename) and selfie_file and allowed_file(selfie_file.filename):
            front_filepath = os.path.join(app.config['UPLOAD_FOLDER'], front_file.filename)
            back_filepath = os.path.join(app.config['UPLOAD_FOLDER'], back_file.filename)
            selfie_filepath = os.path.join(app.config['UPLOAD_FOLDER'], selfie_file.filename)

            front_file.save(front_filepath)
            back_file.save(back_filepath)
            selfie_file.save(selfie_filepath)

            front_has_face, num_front_faces, front_face_locations = analyze_image(front_filepath)
            back_has_face, num_back_faces, back_face_locations = analyze_image(back_filepath)
            selfie_has_face, num_selfie_faces, selfie_face_locations = analyze_image(selfie_filepath)

            if front_has_face:
                front_image_with_rectangles = draw_face_rectangles(front_filepath, front_face_locations)
                cv2.imwrite(front_filepath, front_image_with_rectangles)
                
            if back_has_face:
                back_image_with_rectangles = draw_face_rectangles(back_filepath, back_face_locations)
                cv2.imwrite(back_filepath, back_image_with_rectangles)

            if selfie_has_face:
                selfie_image_with_rectangles = draw_face_rectangles(selfie_filepath, selfie_face_locations)
                cv2.imwrite(selfie_filepath, selfie_image_with_rectangles)

            return render_template('result.html', 
                       original_front_image=front_file.filename, 
                       original_back_image=back_file.filename,
                       original_selfie_image=selfie_file.filename,
                       front_has_face=front_has_face, 
                       num_front_faces=num_front_faces,
                       back_has_face=back_has_face,
                       num_back_faces=num_back_faces,
                       selfie_has_face=selfie_has_face,
                       num_selfie_faces=num_selfie_faces)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)



