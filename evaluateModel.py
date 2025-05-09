from evaluateFolder import evaluate_folder
import csv
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image
import numpy as np

results = evaluate_folder("data/AgedImagesGeneratedByModel")  



def save_results_to_csv(results, out_path="age_gender_results.csv"):
    with open(out_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "actual_age", "predicted_age", "actual_gender", "predicted_gender"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)



# aiming for lower MAE
def compute_mae(results):
    total_error = 0
    count = 0
    for r in results:
        if r['predicted_age'] is not None and r['actual_age'] is not None:
            total_error += abs(r['predicted_age'] - r['actual_age'])
            count += 1
    return total_error / count if count > 0 else None

mean_absolute_error = compute_mae(results)
print(f"✅ Mean Absolute Error: {mean_absolute_error:.2f}%")



# aiming higher percentage
def compute_gender_preservation(results):
    correct = 0
    total = 0
    for r in results:
        if r['actual_gender'] is not None and r['predicted_gender'] is not None:
            total += 1
            if r['actual_gender'].lower() == r['predicted_gender'].lower():
                correct += 1
    return (correct / total) * 100 if total > 0 else None


gender_preservation = compute_gender_preservation(results)
print(f"✅ Gender Preservation Score: {gender_preservation:.2f}%")

# Compute average blurriness using Laplacian Variance
# higher variance= sharper image
def compute_average_blurriness(results):
    blur_vals = [r['blurriness'] for r in results if r['blurriness'] is not None]
    return sum(blur_vals) / len(blur_vals) if blur_vals else None

avg_blur = compute_average_blurriness(results)
print(f"🔍 Average Blurriness (Laplacian Var): {avg_blur:.2f}")


#compute expression preservation.
def compute_expression_preservation(original_results, aged_results):
    total = 0
    correct = 0
    for orig, aged in zip(original_results, aged_results):
        if orig['expression'] and aged['expression']:
            total += 1
            if orig['expression'].lower() == aged['expression'].lower():
                correct += 1
    return (correct / total) * 100 if total > 0 else None


""" Will implement this later:
original_results and aged_results are:
They are both lists of dictionaries, like the results you've been building using evaluate_folder(...), for original and aged images respectively.

Each entry looks like:

{
    "filename": "40_0_0_20170104204933500.jpg",
    "actual_age": 40,
    "predicted_age": 42.3,
    "actual_gender": "male",
    "predicted_gender": "male",
    "blurriness": 108.3,
    "expression": "neutral"
}
🧠 How to Create Them
Suppose you have:

Original (unaltered) images in: data/cropped_clean/

Aged/generated images in: outputs/aged/

You would run:

original_results = evaluate_folder("data/cropped_clean")
aged_results = evaluate_folder("outputs/aged")
And then:


exp_pres = compute_expression_preservation(original_results, aged_results)
print(f"Expression Preservation Score: {exp_pres:.2f}%")
🔗 Matching Logic
The code assumes that original and aged image lists are:

Same length

In corresponding order (i.e., aged[i] is the aged version of original[i])

If that’s not the case (e.g., different file names or order), you should align them using the filename, like this:



def match_results_by_filename(original_results, aged_results):
    matched = []
    aged_dict = {r['filename']: r for r in aged_results}
    for orig in original_results:
        fname = orig['filename']
        if fname in aged_dict:
            matched.append((orig, aged_dict[fname]))
    return matched
Then evaluate:

matched_pairs = match_results_by_filename(original_results, aged_results)
expression_matches = compute_expression_preservation(
    [p[0] for p in matched_pairs],  # originals
    [p[1] for p in matched_pairs]   # aged
)



exp_pres = compute_expression_preservation(orig_results, aged_results)
print(f"😊 Expression Preservation Score: {exp_pres:.2f}%")

"""

# Computing Kernel Inception Distance(KID)
# Aiming for a lower KID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.eval()

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_features(image_paths, model):
    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                feat = adaptive_avg_pool2d(output, output_size=(1, 1)).squeeze().cpu().numpy()
                features.append(feat)
        except Exception as e:
            print(f"❌ Skipping {path}: {e}")
    return np.array(features)

from sklearn.metrics.pairwise import polynomial_kernel

def compute_kid(real_features, generated_features):
    X = real_features
    Y = generated_features

    kXX = polynomial_kernel(X, X, degree=3)
    kYY = polynomial_kernel(Y, Y, degree=3)
    kXY = polynomial_kernel(X, Y, degree=3)

    m = X.shape[0]
    n = Y.shape[0]

    kid = kXX.sum() / (m * m) + kYY.sum() / (n * n) - 2 * kXY.sum() / (m * n)
    return kid

from glob import glob

real_paths = glob("data/cropped_clean/*.jpg")      
generated_paths = glob("outputs/aged/*.jpg")        

real_feats = extract_features(real_paths, inception)
gen_feats = extract_features(generated_paths, inception)

kid_score = compute_kid(real_feats, gen_feats)
print(f"📉 Kernel Inception Distance (KID): {kid_score:.4f}")




