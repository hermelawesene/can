from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
from PIL import Image

# Load the model
model = pickle.load(open('BTDs1_model.pkl', 'rb'))



app = Flask(__name__)

# Define the upload folder path
UPLOAD_FOLDER = r'C:\Users\yabsi\OneDrive\Desktop\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'imagefile' key exists in the files
    if 'imagefile' not in request.files:
        return "No file part in the request"
    
    imagefile = request.files['imagefile']
    
    if imagefile.filename == '':
        return "No selected file"
    
    if imagefile:
        # Save the uploaded image temporarily
        filepath = os.path.join(UPLOAD_FOLDER, imagefile.filename)
        imagefile.save(filepath)
        
        # Load the image and preprocess it
        img = Image.open(filepath)
        img = img.resize((224, 224))  # Resize according to your model's input size
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for the model
        
        # Predict using the loaded model
        prediction_index = model.predict(img_array).argmax()  # Get the index of the highest probability
        
        # Define labels for the model's output
        classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        prediction = classes[prediction_index]  # Map index to label
        
        # Delete the temporary file after prediction
        os.remove(filepath)
        
        # Return the prediction result
        return render_template('index.html', prediction=prediction)

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)