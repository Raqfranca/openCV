from flask import Flask, request, jsonify
import os
import face_recognition

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_faces(image1_path, image2_path):
    # Carregar as imagens
    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    # Obter os encodings faciais
    encoding1 = face_recognition.face_encodings(image1)[0]
    encoding2 = face_recognition.face_encodings(image2)[0]

    # Calcular a distância euclidiana entre os encodings
    euclidean_distance = face_recognition.face_distance([encoding1], encoding2)[0]

    return euclidean_distance

# Define as rotas da API
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'document_file' not in request.files or 'selfie_file' not in request.files:
        return jsonify({'error': 'No file part'})

    document_file = request.files['document_file']
    selfie_file = request.files['selfie_file']

    if document_file.filename == '' or selfie_file.filename == '' or not allowed_file(document_file.filename) or not allowed_file(selfie_file.filename):
        return jsonify({'error': 'Invalid files'})

    document_filepath = os.path.join(app.config['UPLOAD_FOLDER'], document_file.filename)
    selfie_filepath = os.path.join(app.config['UPLOAD_FOLDER'], selfie_file.filename)

    document_file.save(document_filepath)
    selfie_file.save(selfie_filepath)

    # Comparar as faces nas duas imagens
    euclidean_distance_threshold = 0.55  # Limite de distância euclidiana
    euclidean_distance = compare_faces(document_filepath, selfie_filepath)

    same_person = euclidean_distance <= euclidean_distance_threshold

    # Convertendo para tipo primitivo do Python
    same_person = bool(same_person)

    return jsonify({'same_person': same_person, 'euclidean_distance': euclidean_distance.item()})

# Rota para verificar a saúde da API
@app.route('/health', methods=['GET'])
def health_check():
    return 'API is healthy'

if __name__ == '__main__':
    app.run(debug=True)
