from keras.models import load_model
import flask
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
from decimal import *
keras.__version__


# load the model, and pass in the custom loss function
global model
model = ResNet50(weights='imagenet')

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    img = image.load_img(file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    output = decode_predictions(preds, top=3)[0]
    labels = ""
    getcontext().prec = 1
    for label in output:
        labels += "\nClass: " + label[1] + \
            ", Prob: " + str(label[2])
    return labels


# instantiate flask
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    output = None
    file_path = None
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')
