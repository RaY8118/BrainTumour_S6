import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from flask import Flask, render_template
app = Flask(__name__)
app.secret_key = 'abc'


model_path = 'model.h5'
model = load_model(model_path)

@app.route('/predict')
def predict_image():
    # Load the model
   

    # Read the image
    image_path = 'uploads/pred5.jpg'
    image = cv2.imread(image_path)

    # Convert the image to PIL format
    img = Image.fromarray(image)

    # Resize the image
    img = img.resize((64, 64))

    # Convert the image to numpy array
    img = np.array(img)

    # Expand the dimensions of the image
    input_img = np.expand_dims(img, axis=0)

    # Predict using the model
    result = model.predict(input_img)

    return str(result)

if __name__ == '__main__':
    app.run(debug = True)