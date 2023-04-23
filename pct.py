#!/usr/bin/env python3
# By C3n7ral051nt4g3ncy, aka OSINT Tactical
# PCT People Count Tool uses AI (YOLOv5)

import os
import cv2
import numpy as np
import torch
from flask import Flask, request, redirect, flash
from werkzeug.utils import secure_filename
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from flask import render_template_string


def get_html_template():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>People AI Count Tool</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                margin: 0;
                padding: 0;
                font-family: sans-serif;
                background-color: black;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 50px;
            }
            h1 {
                color: #66FF00;
                text-align: center;
                margin-top: 0;
            }
            h2 {
                color: white;
                text-align: center;
                margin-top: 0;
            }
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #FFFFFF;
                border-radius: 5px;
                padding: 30px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            }
            input[type=file] {
                margin-bottom: 20px;
            }
            input[type=submit] {
                background-color: black;
                color: #FFFFFF;
                font-size: 1.2em;
                border-radius: 5px;
                padding: 10px 20px;
                border: none;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            input[type=submit]:hover {
                background-color: black;
                color: #66FF00;
            }
            .result-box {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                height: 200px;
                background-color: black;
                margin-top: 30px;
                border-radius: 5px;
            }
            .result-text {
                color: #00FF00;
                font-size: 6em;
                font-weight: bold;
            }
            .result {
                color: white;
                text-align: center;
                font-size: 1.5em;
                margin-top: 30px;
            }
            .result-value {
                color: #66FF00;
                font-weight: bold;
            }
            .loader {
                border: 16px solid #f3f3f3; 
                border-top: 16px solid #2F4F4F;
                border-radius: 50%;
                width: 80px;
                height: 80px;
                animation: spin 2s linear infinite;
                margin-top: 30px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            footer {
                background-color: #FFFFFF;
                padding: 20px;
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #999999;
                border-top: 1px solid #E5E5E5;
            }
            footer p {
                margin: 0;
            }
            footer img {
                height: 50px;
                width: 50px;
                margin-right: 20px;
                vertical-align: middle;
            }
        </style>
    </head>
    <body>

        <div class="container">
            <header>
                <center><img src="https://raw.githubusercontent.com/C3n7ral051nt4g3ncy/PCT/master/PCT_Tool_logo.png">
                <h1>People Count Tool </h1>
                <h2>This tool uses YOLOv5 Artificial Intelligence</h2>
                <h2>Automating the count of people can help with OSINT investigations</h2>
            </header>
            <form method="post" enctype="multipart/form-data">
                <label for="file-upload">Upload a photo (JPG | JPEG | PNG | GIF ):</label>
                <p>⚠️ For full Tool efficiency, make sure image is good quality and people are clear<p/>
                <input type="file" id="file-upload" name="file">
                <input type="submit" value="Count People">
            </form>
            {% if result is not none %}
                <div class="result">
                    The number of people in the photo is: <span class="result-value">{{ result }}</span>
                </div>
            {% endif %}
        </div>
        <footer>
            <p>To contribute to the tool, head to the <a href="https://github.com/C3n7ral051nt4g3ncy/PCT"
            "target="_blank">PCT Github Repository</a>
            <p>This tool was made by <a href="https://twitter.com/OSINT_Tactical"target="_blank">@OSINT_Tactical</a> 
            (also known as C3n7ral051nt4g3ncy 😁 ) using Python, Flask, and YOLOv5 🔥.</p>
            <p>To contact me, you will find all details on my <a href="https://github.com/C3n7ral051nt4g3ncy" 
            target="_blank">GitHub</a>
            <p>YOLOv5 🚀 is the world's most loved vision AI, representing Ultralytics open-source research into future 
            vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours 
            of research and development</p>
            <br>
            <center><img src="https://avatars.githubusercontent.com/u/104733166?v=4"><img src="
            https://user-images.githubusercontent.com/104733166/199998394-7ac894c9-4e99-44e2-8627-7f98bca1c82c.png">
        </footer>
    </body>
    </html>
    
    '''


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = max(img1_shape) / max(img0_shape)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0] / img1_shape[1], ratio_pad[1][0] / img1_shape[0]
        pad = ratio_pad[0][1], ratio_pad[1][1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain[0]
    coords[:, [1, 3]] /= gain[1]
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)


    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


# People counting function
def load_yolo():
    model = attempt_load("yolov5s.pt")
    model.conf = 0.25
    model.iou = 0.45
    model.classes = [0]
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def process_image(img, model):
    img_size = 640  # Default image size for YOLOv5
    img = letterbox(img, img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def get_objects(img, model):
    img = torch.from_numpy(img).to(model.device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, model.conf, model.iou, model.classes)

    return pred[0]


def count_people(image_path):
    model = load_yolo()
    img = cv2.imread(image_path)
    img_processed = process_image(img, model)
    objects = get_objects(img_processed, model)

    return len(objects)


# Flask
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit file size to 10MB
app.secret_key = 'your_secret_key'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            people_count = count_people(file_path)
            result = people_count

    return render_template_string(get_html_template(), result=result)


if __name__ == '__main__':
    app.run(debug=True)