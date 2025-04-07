import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PromptDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        """
        Args:
            csv_file (str): Path to the CSV file with 'filename' and 'prompt' columns.
            img_dir (str): Directory with pre-cropped, 512x512 face images.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # Only essential transforms (no resizing, because images are already 512x512)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Convert [0,1] → [-1,1]
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load prompt
        prompt = self.data_frame.iloc[idx]['prompt']

        return image, prompt
