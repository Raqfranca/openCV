from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Carregando o modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obtendo as camadas de saída
output_layers = net.getUnconnectedOutLayersNames()

# Função para detecção de objetos e marcação na imagem
def detect_objects(image):
    img_with_objects = image.copy()
    
    # Pré-processamento da imagem
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detecção de objetos
    detected_objects = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = str(classes[class_id])
                detected_objects.append(label)
                
                # Obtendo coordenadas do objeto detectado
                center_x = int(detection[0] * img_with_objects.shape[1])
                center_y = int(detection[1] * img_with_objects.shape[0])
                w = int(detection[2] * img_with_objects.shape[1])
                h = int(detection[3] * img_with_objects.shape[0])
                
                # Desenhando um retângulo em torno do objeto detectado
                cv2.rectangle(img_with_objects, (center_x - w // 2, center_y - h // 2), 
                              (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                cv2.putText(img_with_objects, label, (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    
    return img_with_objects, detected_objects

@app.route('/detect_objects', methods=['POST'])
def detect_objects_endpoint():
    # Receber a imagem
    image_file = request.files['image']
    image_np = np.fromfile(image_file, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Detecção de objetos
    img_with_objects, detected_objects = detect_objects(img)
    
    # Salvar a imagem com os objetos detectados
    cv2.imwrite('detected_objects.jpg', img_with_objects)

    return jsonify({'detected_objects': detected_objects})

if __name__ == '__main__':
    app.run(debug=True)


