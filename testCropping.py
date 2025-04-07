from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt

mtcnn = MTCNN(image_size=512, margin=20)

img = Image.open("data/UTKFace/1_0_0_20161219140623097.jpg").convert("RGB")
cropped = mtcnn(img)

# Show original and cropped side-by-side
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[0].set_title("Original")
axs[1].imshow(cropped.permute(1, 2, 0))  # CHW to HWC
axs[1].set_title("Cropped")
plt.show()
