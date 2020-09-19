from models import crnn_model, text_model
import os
import numpy as np

import json
import base64
from wtforms import FieldList
from flask_wtf.file import FileField
from flask_wtf import FlaskForm
from flask import Flask, request, jsonify, render_template
import sys


# declare constants
HOST = '0.0.0.0'
PORT = 8881

# initialize flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# models
net_text = text_model.TEXT_NET()
net_ocr = crnn_model.CRNN_NET()


@app.route('/predict_infer', methods=['GET', 'POST'])
def predict_infer():
    results = ["good!"]
    if request.method == 'POST':
        print(request.form)
        print(request.files)
        predict_type = request.form['predict_type']
        f = request.files['fileupdate']
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], f.filename))
        print(predict_type)
        results.append(f.filename)
    return json.dumps(results)


if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
