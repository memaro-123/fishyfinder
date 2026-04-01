from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from glob import glob

# Load model
model = load_model('/root/fishyfinder/final_model.h5')
class_names = ['Corbina', 'Lepord', 'Mackrel', 'barredSandBass', 
               'barredSurfPerch', 'batRay', 'garibaldi', 'halibut', 
               'jackSmelt', 'lizardFish', 'shovelnoseGuitarFish', 
               'spotfinCroaker', 'stingRay', 'yellowFinCroaker']

def classify_folder(folder_path):
    """Classify all images in a folder"""
    # Get all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob(os.path.join(folder_path, ext)))
    
    if not images:
        print("No images found!")
        return
    
    print(f"\nFound {len(images)} images")
    print("-" * 50)
    
    for img_path in images:
        # Load and predict
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        pred = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(pred)
        
        print(f"{os.path.basename(img_path):30} → {class_names[class_idx]:20} ({pred[class_idx]:.2%})")

# Use it
folder = input("Enter folder path: ").strip()
if os.path.exists(folder):
    classify_folder(folder)
else:
    print("Folder not found!")