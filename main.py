# app.py
import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imread, imsave
from PIL import Image
import io
import base64
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import LeakyReLU


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved model
model = None  # Placeholder for model path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model('./bestmodel.h5',
                custom_objects={
                    'LeakyReLU': LeakyReLU,
                    'mse': MeanSquaredError()
                }
                                               )
            #model = tf.keras.models.load_model('./bestmodel.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None

def preprocess_image(image_path):
    # Load image
    img = imread(image_path)
    
    # Check if image is grayscale (2D) and convert to RGB (3D)
    if len(img.shape) == 2:
        # Convert grayscale to RGB by duplicating the single channel
        img = np.stack([img] * 3, axis=-1)
    
    # Ensure image has 3 channels (some images might have alpha channel)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    
    # Resize image to 256x256
    img = resize(img, (256, 256), anti_aliasing=True)
    
    # Convert to LAB
    lab_img = rgb2lab(img)
    
    # Extract L channel and normalize
    l_channel = lab_img[:, :, 0]
    
    # Reshape for model input
    l_input = np.expand_dims(l_channel, axis=-1)
    l_input = np.expand_dims(l_input, axis=0)
    
    return l_input, img
def colorize_image(l_input):
    # Predict ab channels
    ab_output = model.predict(l_input)
    
    # Reshape back to image dimensions
    ab_output = ab_output[0] * 128  # Rescale from [-1,1] to [-128, 128]
    
    # Create LAB image
    l_channel = l_input[0, :, :, 0]
    lab_output = np.zeros((256, 256, 3))
    lab_output[:, :, 0] = l_channel
    lab_output[:, :, 1:] = ab_output
    
    # Convert back to RGB
    rgb_output = lab2rgb(lab_output)
    
    # Ensure values are in [0, 1] range
    rgb_output = np.clip(rgb_output, 0, 1)
    
    return rgb_output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        # Load model if not already loaded
        load_model()
        
        if model is None:
            return render_template('index.html', error='Model could not be loaded')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process image
            l_input, original_img = preprocess_image(file_path)
            
            # Colorize image
            colored_img = colorize_image(l_input)
            
            # Convert to grayscale for comparison
            grayscale_img = np.zeros_like(original_img)
            for i in range(3):
                grayscale_img[:, :, i] = original_img[:, :, 0]  # Use L channel for all RGB channels
            
            # Save result images
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"colored_{filename}")
            grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], f"gray_{filename}")
            
            # Convert to uint8 and save
            colored_img_uint8 = (colored_img * 255).astype(np.uint8)
            grayscale_img_uint8 = (grayscale_img * 255).astype(np.uint8)
            
            # Save using PIL to ensure correct format
            Image.fromarray(colored_img_uint8).save(result_path)
            #Image.fromarray(grayscale_img_uint8).save(grayscale_path)
            l_channel = original_img[:, :, 0]
            grayscale_img = np.stack([l_channel] * 3, axis=-1)  # Shape (256, 256, 3)
            grayscale_img_uint8 = (grayscale_img * 255).astype(np.uint8)
            Image.fromarray(grayscale_img_uint8).save(grayscale_path)
            # Convert images to base64 for embedding in HTML
            def get_image_base64(img_array):
                img_uint8 = (img_array * 255).astype(np.uint8)

                # Handle grayscale image (2D) by converting to RGB
                if img_uint8.ndim == 2:
                    img_uint8 = np.stack([img_uint8] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)

                img_pil = Image.fromarray(img_uint8)
                buffered = io.BytesIO()
                img_pil.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return img_str
            
            original_base64 = get_image_base64(grayscale_img)
            result_base64 = get_image_base64(colored_img)
            
            return render_template('index.html', 
                                  original_image=original_base64,
                                  colored_image=result_base64)
            
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')
    
    return render_template('index.html', error='Invalid file type')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0' )
