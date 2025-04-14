import os
from ageEstimator import estimate_age_from_cropped  # your existing function
import cv2
from deepface import DeepFace

"""Contains helper methods for evaluateModel.py"""

def parse_filename_metadata(filename):
    parts = filename.split("_")
    try:
        age = int(parts[0])
        gender = "male" if parts[1] == "0" else "female"
        return age, gender
    except Exception:
        return None, None
    


def compute_blurriness(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()



def detect_expression(img_path):
    try:
        analysis = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']  # e.g., 'happy', 'neutral', 'sad'
    except Exception as e:
        print(f"❌ Error analyzing {img_path}: {e}")
        return None



def evaluate_folder(img_folder):
    results = []
    for fname in os.listdir(img_folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(img_folder, fname)
            actual_age, actual_gender = parse_filename_metadata(fname)
            predicted_age, predicted_gender = estimate_age_from_cropped(img_path)
            blurriness = compute_blurriness(img_path)
            expression = detect_expression(img_path)

            results.append({
                "filename": fname,
                "actual_age": actual_age,
                "predicted_age": predicted_age,
                "actual_gender": actual_gender,
                "predicted_gender": predicted_gender,
                "blurriness": blurriness,
                "expression": expression
            })
    return results





