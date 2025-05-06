import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# 1) Autenticación con Kaggle
api = KaggleApi()
api.authenticate()

# 2) Identificador del dataset y rutas
DATASET = 'aryashah2k/breast-ultrasound-images-dataset'
RAW_DIR = os.path.join('data', 'raw')

os.makedirs(RAW_DIR, exist_ok=True)

# 3) Descarga y descompresión
print('Descargando BUSI dataset desde Kaggle…')
api.dataset_download_files(DATASET, path=RAW_DIR, unzip=False)

zip_path = os.path.join(RAW_DIR, f'{DATASET.split("/")[-1]}.zip')
print(f'Descomprimiendo {zip_path}…')
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(RAW_DIR)

os.remove(zip_path)
print('Dataset listo en:', RAW_DIR)
