import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.radiomics import extract_radiomic_features

class BreastDataGenerator(Sequence):
    def __init__(self, img_paths, mask_paths, labels, batch_size=16, shuffle=True, **kwargs):
        super().__init__(**kwargs)  # allow additional args/ initialize
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.labels     = labels  # one-hot labels
        self.bs         = batch_size
        self.shuffle    = shuffle
        self.on_epoch_end()
    def __init__(self, img_paths, mask_paths, labels, batch_size=16, shuffle=True):
        super().__init__()  # Inicializar Sequence base
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.labels     = labels  # one-hot labels
        self.bs         = batch_size
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.bs))

    def on_epoch_end(self):# No of batches epochs
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.bs : (idx + 1) * self.bs]
        batch_imgs, batch_feats, batch_labels = [], [], []
        for i in batch_idx:
            # charge and process image
            img = load_img(
                self.img_paths[i], color_mode='grayscale', target_size=(224,224)
            )
            img_array = img_to_array(img) / 255.0  # (224,224,1)

            # extract radiomics features
            feats = extract_radiomic_features(
                self.img_paths[i], self.mask_paths[i]
            )  # vector (5,)

            batch_imgs.append(img_array)
            batch_feats.append(feats)
            batch_labels.append(self.labels[i])

        # Convert lists to numpy arrays
        batch_imgs   = np.array(batch_imgs, dtype=np.float32)
        batch_feats  = np.array(batch_feats, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.float32)

        return [batch_imgs, batch_feats], batch_labels
