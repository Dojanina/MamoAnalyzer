import os
from PIL import Image
import numpy as np

#Directory path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

#Routes
raw_folder = os.path.join(BASE_DIR, 'data', 'raw')
processed_folder = os.path.join(BASE_DIR, 'data', 'processed')

#Parameters
target_size = (224, 224)

os.makedirs(processed_folder, exist_ok=True)

def preprocess_image(input_path, output_path, size=target_size):
   #Loads the image, crops by mask if available, resizes, normalizes and saves the image.
    try:
        #Load image in grayscale
        img = Image.open(input_path).convert('L')

        #Locate and load mask if available
        base, ext = os.path.splitext(input_path)
        mask_path = base + '_mask' + ext
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            # Binarize mask (0 = background, 255 = ROI) 
            mask = mask.point(lambda x: 255 if x > 128 else 0)
            bbox = mask.getbbox()
            if bbox:
                img = img.crop(bbox)

        #Resize
        img = img.resize(size, Image.LANCZOS)

        #Normalize
        arr = np.array(img) / 255.0
        out_img = Image.fromarray((arr * 255).astype(np.uint8))

        #Saved processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == '__main__':
    for root, _, files in os.walk(raw_folder):
        rel_path = os.path.relpath(root, raw_folder)
        if rel_path == '.':
            continue

       # Create output folder for the class
        out_dir = os.path.join(processed_folder, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        for fname in files:
            #Skip mask files
            if 'mask' in fname.lower():
                continue
            #Only process image files
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                in_path = os.path.join(root, fname)
                out_path = os.path.join(out_dir, fname)
                preprocess_image(in_path, out_path)
                print(f"Processed → {out_path}")

    print('Preprocess complete!')
