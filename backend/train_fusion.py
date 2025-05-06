import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.utils import Sequence
from utils.radiomics import extract_radiomic_features

#Paths
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw', 'Dataset_BUSI_with_GT')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'rad_scaler.gz')
os.makedirs(MODEL_DIR, exist_ok=True)

#Detect classes
CLASSES = sorted([d for d in os.listdir(PROCESSED_DIR)
                  if os.path.isdir(os.path.join(PROCESSED_DIR, d))])
if not CLASSES:
    raise FileNotFoundError(f"No classes in {PROCESSED_DIR}")

# Debug processed folder
print("→ PROCESSED_DIR:", PROCESSED_DIR)
print("  Exists?", os.path.isdir(PROCESSED_DIR))
print("  Subfolders:", os.listdir(PROCESSED_DIR))
for cls in CLASSES:
    cls_path = os.path.join(PROCESSED_DIR, cls)
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith(('.png','.jpg','.jpeg'))]
    print(f"    {cls}: {len(imgs)} images")

# ImageDataGenerators
train_img_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
).flow_from_directory(
    PROCESSED_DIR,
    target_size=(224,224),
    classes=CLASSES,
    class_mode='categorical',
    subset='training',
    batch_size=32,
    shuffle=True
)
val_img_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
).flow_from_directory(
    PROCESSED_DIR,
    target_size=(224,224),
    classes=CLASSES,
    class_mode='categorical',
    subset='validation',
    batch_size=32,
    shuffle=False
)
print("Classes detected:", train_img_gen.class_indices)
print("# train images:", train_img_gen.samples)
print("# val images:", val_img_gen.samples)

#Radiomics: extract & scale features
rad_dict = {}
for cls in CLASSES:
    proc_dir = os.path.join(PROCESSED_DIR, cls)
    raw_dir  = os.path.join(RAW_DIR, cls)
    for fname in os.listdir(proc_dir):
        if 'mask' in fname.lower(): continue
        img_path  = os.path.join(raw_dir, fname)
        base, ext = os.path.splitext(fname)
        mask_path = os.path.join(raw_dir, f"{base}_mask{ext}")
        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask for {fname}, skipping.")
            continue
        rad_dict[fname] = extract_radiomic_features(img_path, mask_path)

# Fit scaler on training subset
train_files = [os.path.basename(p) for p in train_img_gen.filepaths if '_mask' not in p.lower()]
train_feats = np.stack([rad_dict[f] for f in train_files])
scaler = StandardScaler().fit(train_feats)
joblib.dump(scaler, SCALER_PATH)
print(f"Saved radiomics scaler to {SCALER_PATH}")
# Transform all
for fname, feats in rad_dict.items():
    rad_dict[fname] = scaler.transform(feats.reshape(1, -1))[0]

#FusionSequence
class FusionSequence(Sequence):
    def __init__(self, img_gen, rad_dict, **kwargs):
        super().__init__()
        self.img_gen = img_gen
        self.rad_dict= rad_dict
        self.bs      = img_gen.batch_size
    def __len__(self):
        return len(self.img_gen)
    def __getitem__(self, idx):
        X_img, y = self.img_gen[idx]
        start    = idx * self.bs
        end      = start + X_img.shape[0]
        idxs     = self.img_gen.index_array[start:end]
        files    = [os.path.basename(self.img_gen.filepaths[i]) for i in idxs]
        X_rad    = np.stack([self.rad_dict[f] for f in files])
        return (X_img, X_rad), y

train_seq = FusionSequence(train_img_gen, rad_dict)
val_seq   = FusionSequence(val_img_gen,   rad_dict)

# Build model
input_img = layers.Input((224,224,3), name='img_rgb')
backbone  = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_img)
backbone.trainable = True
#Unfreeze only the last N layers
for layer in backbone.layers[:-20]:
    layer.trainable = False
x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
input_rad = layers.Input((train_feats.shape[1],), name='rad_in')
rad       = layers.Dense(16, activation='relu')(input_rad)
merged    = layers.Concatenate()([x, rad])
merged    = layers.Dense(32, activation='relu')(merged)
merged    = layers.Dropout(0.3)(merged)
output    = layers.Dense(len(CLASSES), activation='softmax')(merged)
model     = Model([input_img, input_rad], output, name='EffNet_RadFusion')
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
#Callbacks & training
ckpt = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR,'best_fusion.h5'),
    monitor='val_accuracy', save_best_only=True, verbose=1
)
es   = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=20,
    callbacks=[ckpt, es]
)
print("Training finished. Models in", MODEL_DIR)
#ImageDataGenerators 
train_img_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2,
    zoom_range=[0.8, 1.2],
    shear_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
).flow_from_directory(
    PROCESSED_DIR,
    target_size=(224,224),
    classes=CLASSES,
    class_mode='categorical',
    subset='training',
    batch_size=32,
    shuffle=True
)
val_img_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
).flow_from_directory(
    PROCESSED_DIR,
    target_size=(224,224),
    classes=CLASSES,
    class_mode='categorical',
    subset='validation',
    batch_size=32,
    shuffle=False
)
#Build model
input_img = layers.Input((224,224,3), name='img_rgb')
backbone  = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_img)
backbone.trainable = False
x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
input_rad = layers.Input((train_feats.shape[1],), name='rad_in')
rad       = layers.Dense(16, activation='relu')(input_rad)
merged    = layers.Concatenate()([x, rad])
merged    = layers.Dense(32, activation='relu')(merged)
merged    = layers.Dropout(0.3)(merged)
output    = layers.Dense(len(CLASSES), activation='softmax')(merged)
model     = Model([input_img, input_rad], output, name='EffNet_RadFusion')
model.compile(
optimizer=optimizers.Adam(1e-4),
loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
metrics=['accuracy']
)
model.summary()

#Callbacks & training 
ckpt = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR,'eff_rad_fusion.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
es = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

# Calculate class weights to compensate for class imbalance
# train_img_gen.classes is an array with the label of each training sample
y_train = train_img_gen.classes
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("class_weights =", class_weights)

#Training with class_weight
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=20,
    callbacks=[ckpt, es],
    class_weight=class_weights
)