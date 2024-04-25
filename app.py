from flask import Flask, request, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from keras.models import load_model  # type: ignore
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = 'abc'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model('model/model.h5')


@app.route('/predict')
def predict_image():
    # Check if uploaded filename exists in session
    if 'uploaded_filename' not in session:
        flash('No uploaded file found')
        return redirect(url_for('upload_file'))

    uploaded_filename = session['uploaded_filename']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    if os.path.isfile(image_path):
        image_files = [uploaded_filename]
    image = cv2.imread(image_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    result = model.predict(input_img)
    result = int(result)
    if result == 0:
        message = "There is no tumor"
    elif result == 1:
        message = "There is a tumor"
    else:
        message = "Error occured cant decide"
    return render_template('predict.html', result=result, message=message, image_files=image_files)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            session['uploaded_filename'] = filename
            return redirect('predict')
    return render_template('upload.html')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
