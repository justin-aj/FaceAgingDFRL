#testing if the 150 images look good or no?


import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os


csv_path = "data/csv/utk_train_balanced.csv"
img_dir = "data/croppedUTKFace"


df = pd.read_csv(csv_path)


images_per_row = 10
rows = (len(df) + images_per_row - 1) // images_per_row
fig, axes = plt.subplots(rows, images_per_row, figsize=(images_per_row * 3, rows * 3))

for i, row in df.iterrows():
    img_path = os.path.join(img_dir, row['filename'])
    try:
        img = Image.open(img_path)
        ax = axes[i // images_per_row, i % images_per_row]
        ax.imshow(img)
        ax.set_title(f"{row['age']}y\n{row['gender']}, {row['race_str']}", fontsize=8)
        ax.axis('off')
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")


for j in range(len(df), rows * images_per_row):
    axes[j // images_per_row, j % images_per_row].axis('off')

plt.tight_layout()
plt.show()
