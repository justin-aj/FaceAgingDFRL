import os
import cv2
import pandas as pd

# Paths
labels_csv_path = "data/csv/utk_labels.csv"
cropped_dir = "data/croppedUTKFace"
output_csv_path = "data/subset150/utk_train_balanced.csv"


df = pd.read_csv(labels_csv_path)

available_crops = set(os.listdir(cropped_dir))
df = df[df['filename'].isin(available_crops)].copy()

def extract_race(filename):
    try:
        return int(filename.split("_")[2])
    except:
        return None

df['race'] = df['filename'].apply(extract_race)

# Map race labels to readable strings (as per UTKFace)
race_map = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Other'
}
df['race_str'] = df['race'].map(race_map)

# Add age group
"""
Here’s how it works:

df['age'] // 10 → integer division (e.g., 23 → 2, 68 → 6)* 10 → turns it into the starting age of the decade:

2 → 20

6 → 60

This creates a new column called age_group, which contains values like:

Copy
Edit
5, 15, 25, 35, 45, 55, 65, 75
"""

df['age_group'] = (df['age'] // 10) * 10


def is_blurry(img_path, threshold=100.0):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    return cv2.Laplacian(image, cv2.CV_64F).var() < threshold


blur_checked = []
for _, row in df.iterrows():
    img_path = os.path.join(cropped_dir, row['filename'])
    if not is_blurry(img_path):
        blur_checked.append(row)

df_clean = pd.DataFrame(blur_checked)

# Sample ~30 images per race group, balancing across age groups
df_balanced = df_clean.groupby(['race_str', 'age_group']).apply(lambda x: x.sample(min(5, len(x)), random_state=42))
df_balanced = df_balanced.reset_index(drop=True)

# Save final balanced dataset for testing
df_balanced.to_csv(output_csv_path, index=False)
df_balanced.groupby("race_str").size()
