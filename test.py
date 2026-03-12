"""
Hand Gesture Recognition - Model Testing Script
-----------------------------------------------
This script loads the trained CNN model and predicts
hand gesture classes for input images.

Classes:
NONE, ONE, TWO, THREE, FOUR, FIVE
"""

# ======================================
# Import Required Libraries
# ======================================

from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os


# ======================================
# Step 1: Load Trained Model
# ======================================

# Load model architecture
with open("../models/model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

# Load model weights
model.load_weights("../models/model.h5")

print("✅ Model loaded successfully!")


# ======================================
# Step 2: Class Labels
# ======================================

classes = ["NONE", "ONE", "TWO", "THREE", "FOUR", "FIVE"]


# ======================================
# Step 3: Image Classification Function
# ======================================

def classify(img_path):
    """
    Predict the hand gesture in the given image.
    """

    # Load image
    test_image = image.load_img(
        img_path,
        target_size=(256, 256),
        color_mode="grayscale"
    )

    # Convert image to array
    test_image = image.img_to_array(test_image)

    # Add batch dimension
    test_image = np.expand_dims(test_image, axis=0)

    # Normalize image
    test_image = test_image / 255.0

    # Predict
    result = model.predict(test_image)

    # Get class index
    predicted_class_index = np.argmax(result)

    # Get class label
    predicted_label = classes[predicted_class_index]

    print("Image:", img_path)
    print("Prediction:", predicted_label)
    print("-------------------------")


# ======================================
# Step 4: Test Multiple Images
# ======================================

test_folder = "../dataset/test_samples"

files = []

# Collect all PNG images
for root, dirs, filenames in os.walk(test_folder):
    for file in filenames:
        if file.endswith(".png"):
            files.append(os.path.join(root, file))


# Run prediction on all images
for img in files:
    classify(img)