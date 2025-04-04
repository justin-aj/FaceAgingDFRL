import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
image_size = 128  # Higher resolution
num_steps = 100
batch_size = 32
epochs = 20  # More training
learning_rate = 0.0002

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced U-Net style Diffusion Model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x = self.dec1(x2)
        x = self.dec2(x)
        return x

# Dataset Preparation
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Simulated dataset
train_data = datasets.FakeData(transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training Loop
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Generate Face Using Trained Model
model.eval()
with torch.no_grad():
    noise = torch.randn(1, 1, image_size, image_size).to(device)
    generated_face = model(noise).cpu().squeeze().numpy()

# Plot
plt.imshow(generated_face, cmap='gray')
plt.title('Enhanced Generated Face Using Diffusion Model')
plt.show()
