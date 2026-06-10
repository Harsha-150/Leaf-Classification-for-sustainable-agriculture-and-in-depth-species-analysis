from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)
app.secret_key = 'abcd123'

# Load the model
try:
    model = load_model("c:/Users/surya/OneDrive/Desktop/MPIP03/code/NASNetMobileleaf.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class names
class_names = ['Buzgulu', 'Dimnit', 'Nazli','Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
        'Tomato___late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

users = {}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dat'}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    """Check if the uploaded file is a valid image."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def import_and_predict(image_path, model):
    """Process the image and use the model for prediction."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.fit(image, (224, 224), Image.LANCZOS)
        img = np.asarray(image) / 255.0
        img = img[..., :3]  # Ensure RGB format
        img_reshape = np.expand_dims(img, axis=0)

        print(f"Processing image: {image_path}, Shape: {img_reshape.shape}")
        predictions = model.predict(img_reshape)
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'Unknown'
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in import_and_predict: {e}")
        return None, None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users:
            flash('Username already exists.', 'error')
            return redirect(url_for('signup'))
        
        users[username] = password
        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html')
    else:
        flash('You need to log in first.', 'error')
        return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                file_path = os.path.join('static/uploads', file.filename)
                file.save(file_path)

                predicted_class, accuracy = import_and_predict(file_path, model)
                if predicted_class is None:
                    flash('Prediction failed.', 'error')
                    return redirect(url_for('index'))

                # OpenCV grading process
                try:
                    image_cv = cv2.imread(file_path)
                    lab_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)
                    L, a, b = cv2.split(lab_image)
                    graded_L = cv2.equalizeHist(L)
                    graded_lab = cv2.merge((graded_L, a, b))
                    graded_bgr = cv2.cvtColor(graded_lab, cv2.COLOR_Lab2BGR)
                    grading_image_path = f'static/uploads/grading_{file.filename}'
                    cv2.imwrite(grading_image_path, graded_bgr)
                except Exception as e:
                    flash(f'Error in grading image processing: {str(e)}', 'error')
                    return redirect(url_for('index'))
                
                return render_template(
                    'result.html',
                    disease=predicted_class,
                    accuracy=round(accuracy * 100, 2),
                    real_image_path=f'/static/uploads/{file.filename}',
                    grading_image_path=grading_image_path
                )
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Invalid file format or file is too large.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/performance')
def performance():
    labels = ['Buzgulu', 'Dimnit', 'Nazli','Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
        'Tomato___late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
    values = [99, 100, 100, 1702,800,1273,1527,761,1417,1341,1123,299,4286]  
    return render_template('performance.html', labels=labels, values=values)


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
