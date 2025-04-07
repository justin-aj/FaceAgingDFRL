from facenet_pytorch import MTCNN
from PIL import Image
import os

mtcnn = MTCNN(image_size=512, margin=20, keep_all=False)

src_dir = "data/UTKFace"
dst_dir = "data/croppedUTKFace"
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.endswith(".jpg"):
        continue
    try:
        img_path = os.path.join(src_dir, fname)
        img = Image.open(img_path).convert("RGB")
        box, prob = mtcnn.detect(img)

        if prob is None or prob[0] < 0.95:
            print(f"❌ Low confidence for {fname}, skipping")
            continue

        # Check face box size
        x1, y1, x2, y2 = box[0]
        face_width = x2 - x1
        face_height = y2 - y1
        if face_width < 80 or face_height < 80:
            print(f"❌ Small face in {fname}, skipping")
            continue

        # Save high-confidence, large-enough faces
        mtcnn(img, save_path=os.path.join(dst_dir, fname))

    except Exception as e:
        print(f"❌ Failed to process {fname}: {e}")

