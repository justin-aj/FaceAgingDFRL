import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

"""
This class is what the training loop will use to:

Load your cropped face images from data/croppedUTKFace/

Load the text prompt from utk_train_balanced_with_prompts.csv

Apply transforms (resize, normalize, convert to tensor)

Return:

python
Copy
Edit
(image_tensor, prompt_text)
Exactly what Stable Diffusion needs during fine-tuning.
"""

class PromptDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        """
        Args:
            csv_file (str): Path to the CSV file with 'filename' and 'prompt' columns.
            img_dir (str): Directory with pre-cropped, 512x512 face images.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        prompt = self.data_frame.iloc[idx]['prompt']

        return image, prompt
