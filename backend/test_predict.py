import os, joblib, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.radiomics import extract_radiomic_features

# Rutas iguales a app.py
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'eff_rad_mixup.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'rad_scaler.gz')

model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
CLASS_NAMES = ['benign','malignant','normal']  # ajusta al orden real

def preprocess(img_path, mask_path=None):
    img = load_img(img_path, color_mode='grayscale', target_size=(224,224))
    arr = img_to_array(img)/255.0
    x_img = np.stack([arr[:,:,0]]*3, axis=-1)
    rad = extract_radiomic_features(img_path, mask_path) if mask_path else np.zeros(5)
    rad = scaler.transform(rad.reshape(1,-1)).reshape(-1)
    return np.expand_dims(x_img,0), np.expand_dims(rad,0)

# Pon aquí una imagen de prueba y su máscara si tienes
img_path  = 'data/raw/Dataset_BUSI_with_GT/malignant/img_001.jpeg'
mask_path = 'data/raw/Dataset_BUSI_with_GT/malignant/img_001_mask.jpeg'

X_img, X_rad = preprocess(img_path, mask_path)
preds = model.predict([X_img, X_rad])[0]
for cls, p in zip(CLASS_NAMES, preds):
    print(f"{cls}: {p:.3f}")
print("Predicción final:", CLASS_NAMES[np.argmax(preds)])
