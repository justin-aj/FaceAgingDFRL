import insightface
import cv2
import numpy as np

# Load age/gender model
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0)

image_path= "/Users/koppisettyeashameher/Desktop/FaceAgingDFRL/data/croppedUTKFace/24_1_2_20170109213251114.jpg"

import os

image_path = "outputs/aged_face.png"
if os.path.exists(image_path):
    print("✅ Image exists!")
else:
    print("❌ Image not found. Check the path or filename.")


def estimate_age(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = model.get(img)
    if faces:
        age = faces[0]['age']
        gender = 'male' if faces[0]['gender'] == 1 else 'female'
        return age, gender
    else:
        return None, None

# Example usage:
age, gender = estimate_age("outputs/aged_face.png")
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {24}, gender: {'female'}")
