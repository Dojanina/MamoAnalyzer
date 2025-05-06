import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.radiomics import extract_radiomic_features
import sqlite3
import hashlib

# Configuration
BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))
RAW_DIR      = os.path.join(BASE_DIR, 'data', 'raw', 'Dataset_BUSI_with_GT')

MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'best_fusion.h5')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'rad_scaler.gz')
DB_PATH      = os.path.join(BASE_DIR, 'users.db')

# Get class names from dataset folders
CLASS_NAMES  = sorted([
    d for d in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, d))
])   # Example: ['benign','malignant','normal']

# Flask Setup
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.secret_key = 'Kinderbueno'  # Secret key for session management
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained model and scaler
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Initialize SQLite Database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

# Initialize DB on startup
init_db()

# Frontend Routes
@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')

# Image Preprocessing Function 
def preprocess_image(img_path, mask_path=None):
    """
    Preprocesses an image and optional mask, returning:
    - x_img: (224, 224, 3) normalized image array
    - x_rad: (5,) scaled radiomic features
    """
    img = load_img(img_path, color_mode='grayscale', target_size=(224, 224))
    arr = img_to_array(img)

    #Convert grayscale to 3-channel image
    x_img = np.stack([arr[:, :, 0]] * 3, axis=-1)

    #Extract radiomic features
    if mask_path:
        rad = extract_radiomic_features(img_path, mask_path)
    else:
        rad = np.zeros((5,), dtype='float32')

    #Normalize radiomic features
    rad = scaler.transform(rad.reshape(1, -1)).reshape(-1)

    return x_img, rad

#Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files.get('image')
    if not img_file:
        return jsonify({'error': 'Image file is required'}), 400

    tmp_dir = os.path.join(BASE_DIR, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    img_path = os.path.join(tmp_dir, img_file.filename)
    img_file.save(img_path)

    mask_file = request.files.get('mask')
    mask_path = None
    if mask_file:
        mask_path = os.path.join(tmp_dir, mask_file.filename)
        mask_file.save(mask_path)

    #Preprocess image and predict
    x_img, x_rad = preprocess_image(img_path, mask_path)
    x_img = np.expand_dims(x_img, 0)
    x_rad = np.expand_dims(x_rad, 0)

    preds = model.predict([x_img, x_rad])[0]

    p_benign    = float(preds[CLASS_NAMES.index('benign')])
    p_malignant = float(preds[CLASS_NAMES.index('malignant')])
    p_normal    = float(preds[CLASS_NAMES.index('normal')])

    all_probs  = {'benign': p_benign, 'malignant': p_malignant, 'normal': p_normal}
    predicted  = max(all_probs, key=all_probs.get)
    confidence = all_probs[predicted]

    return jsonify({
        'benign'    : p_benign,
        'malignant' : p_malignant,
        'normal'    : p_normal,
        'predicted' : predicted,
        'confidence': confidence
    })

#Registration 
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    username        = data.get('username')
    confirmUsername = data.get('confirmUsername')
    email           = data.get('email')
    password        = data.get('password')
    confirmPassword = data.get('confirmPassword')

    if not all([username, confirmUsername, email, password, confirmPassword]):
        return jsonify({'error': 'All fields are required'}), 400
    if username != confirmUsername:
        return jsonify({'error': 'Usernames do not match'}), 400
    if password != confirmPassword:
        return jsonify({'error': 'Passwords do not match'}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_pw)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400

    return jsonify({'message': 'Registration successful'})

#Login Route
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, hashed_pw)
        )
        user = cursor.fetchone()

    if user:
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

#Logout 
@app.route('/logout')
def logout():
    session.clear()  # Clears user session
    return redirect('/')  # Redirects to home page

# Main Execution 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
