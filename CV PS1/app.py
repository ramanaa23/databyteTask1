# import libraries 

from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
# Create flask instance 
app = Flask(__name__)
model = tf.keras.models.load_model("flower_classifier_model.h5") # Load model 

class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']  # Class names in my dataset 

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST': # Handle POST request (form submission)
        img = request.files['image'] # Get the uploaded image file from the form
        if img:
            img_path = os.path.join('static', img.filename) # Save the uploaded image to the static/ directory
            img.save(img_path)
            # Load the image in the same format as the model expects
            image = load_img(img_path, target_size=(150, 150))
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            # Predict the class using the model
            prediction = model.predict(img_array)
            pred_class = class_names[np.argmax(prediction)]
            # Render the result page with prediction and uploaded image
            return render_template('index.html', prediction=pred_class, img_path=img_path)
 # For GET request (initial page load), show empty prediction
    return render_template('index.html', prediction='', img_path='')
# Run
if __name__ == '__main__':
    app.run(debug=True)
