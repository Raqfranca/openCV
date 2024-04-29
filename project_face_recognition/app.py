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
    rbg_img = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    image = cv2.cvtColor(image_path)

    
