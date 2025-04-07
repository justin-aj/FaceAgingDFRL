from torch.utils.data import DataLoader

from prompt_dataset import PromptDataset

print(PromptDataset)


dataset = PromptDataset(
    csv_file="data/csv/utk_train_balanced_with_prompts.csv",
    img_dir="data/UTKFace/cropped_clean"
)

# Example: fetch a single item
img, prompt = dataset[0]
print(prompt)  # e.g., "a photo of a 45 year old person"

# Load into DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


for images, prompts in dataloader:
    print(images.shape)   # torch.Size([4, 3, 512, 512])
    print(prompts)        # list of 4 text prompts
    break

