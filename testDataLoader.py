from torch.utils.data import DataLoader

from prompt_dataset import PromptDataset

print(PromptDataset)


dataset = PromptDataset(
    csv_file="data/csv/utk_train_balanced_with_prompts.csv",
    img_dir="data/croppedUTKFace"
)


img, prompt = dataset[0]
print(prompt) 


dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


for images, prompts in dataloader:
    print(images.shape)   
    print(prompts)       
    break

