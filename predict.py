from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the saved model
model = load_model('/root/fishyfinder/final_model.h5')
print(" Model loaded successfully!")

# Option 1: Hardcode them if you know them
['Corbina', 'Lepord', 'Mackrel', 'barredSandBass', 
               'barredSurfPerch', 'batRay', 'garibaldi', 'halibut', 
               'jackSmelt', 'lizardFish', 'shovelnoseGuitarFish', 
               'spotfinCroaker', 'stingRay', 'yellowFinCroaker']

def predict_fish(image_path):
    """
    Predict the class of a fish image
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale like during training
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    # Display results
    print(f"\n📸 Image: {image_path}")
    print(f"🎯 Predicted: {class_names[predicted_class_idx]}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    print("\n📈 Top 3 predictions:")
    for idx in top_3_idx:
        print(f"   - {class_names[idx]}: {predictions[0][idx]:.2%}")
    
    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {class_names[predicted_class_idx]}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.show()
    
    return class_names[predicted_class_idx], confidence

# Add this at the bottom of your script
if __name__ == "__main__":
    # Test on a single image
    image_path = "/root/fishyfinder/test_images/your_fish.jpg"  # CHANGE THIS to your image path
    if os.path.exists(image_path):
        result, confidence = predict_fish(image_path)
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please change the image_path variable to point to your fish image")