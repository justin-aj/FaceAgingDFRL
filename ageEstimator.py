import insightface
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import cv2
import numpy as np


import cv2
import numpy as np
from insightface.app import FaceAnalysis


app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)
genderage_model = app.models.get('genderage')
session = genderage_model.session

def estimate_age_from_cropped(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Failed to load image.")
        return None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (96, 96)).astype(np.float32)
    blob = np.transpose(img_resized, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]

    
    female_score = output[0][0]
    male_score = output[0][1]
    age_score = output[0][2]  

    gender = "male" if male_score > female_score else "female"
    age = age_score * 100

    return age, gender

#Testing:
image_path= "data/croppedUTKFace/40_0_0_20170104204933500.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {40}, gender: {'male'}")


image_path= "data/croppedUTKFace/64_1_0_20170110160643892.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {64}, gender: {'female'}")


image_path= "data/croppedUTKFace/13_0_2_20170103201143159.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {13}, gender: {'male'}")


image_path= "data/croppedUTKFace/30_1_1_20170110120856819.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {30}, gender: {'female'}")


image_path= "data/croppedUTKFace/75_0_3_20170111202756116.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {75}, gender: {'male'}")


image_path= "data/croppedUTKFace/12_0_4_20170103201824880.jpg"
age, gender = estimate_age_from_cropped(image_path)
print(f"Predicted age: {age}, gender: {gender}")
print(f"Actual age: {12}, gender: {'male'}")
