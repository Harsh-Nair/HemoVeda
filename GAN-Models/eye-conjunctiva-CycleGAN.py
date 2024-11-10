import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image
import itertools
import os
import time
import gc
import datetime
import numpy as np

data_dir = Path('/path/to/your/main/folder/directory')
anemic_dir = data_dir / 'Anemic'
healthy_dir = data_dir / 'Healthy'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# Custom Dataset
class EyeDataset(Dataset):
    def __init__(self, anemic_dir, healthy_dir, transform=None):
        self.anemic_images = list(Path(anemic_dir).glob("*.png")) + list(Path(anemic_dir).glob("*.jpg"))
        self.healthy_images = list(Path(healthy_dir).glob("*.png")) + list(Path(healthy_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return min(len(self.anemic_images), len(self.healthy_images))

    def __getitem__(self, idx):
        anemic_img = Image.open(self.anemic_images[idx]).convert("RGB")
        healthy_img = Image.open(self.healthy_images[idx]).convert("RGB")

        if self.transform:
            anemic_img = self.transform(anemic_img)
            healthy_img = self.transform(healthy_img)

        return {"Anemic": anemic_img, "Healthy": healthy_img}



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(0.15),
    transforms.RandomVerticalFlip(0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = EyeDataset(anemic_dir, healthy_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# Define Generator and Discriminator
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # Residual blocks
        for _ in range(9):
            model += [ResidualBlock()]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        # Output layer
        model += [nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Initialize models
G_A2B = Generator().to(device)
G_B2A = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Loss and Optimizers
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 500



# Timer Setup
start_time = time.time()
max_duration = 11 * 60 * 60 + 50 * 60  # 11 hours 50 minutes in seconds

for epoch in range(num_epochs):
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_duration:
        print("Maximum training time reached. Stopping training...")
        
        torch.save(G_A2B.state_dict(), "/path/to/save/G_A2B_last_epoch.pth")
        torch.save(G_B2A.state_dict(), "/path/to/save/G_B2A_last_epoch.pth")
        torch.save(D_A.state_dict(), "/path/to/save/D_A_last_epoch.pth")
        torch.save(D_B.state_dict(), "/path/to/save/D_B_last_epoch.pth")
        break

    for i, batch in enumerate(dataloader):
        real_A = batch['Anemic'].to(device)
        real_B = batch['Healthy'].to(device)

        # Train Generators
        optimizer_G.zero_grad()
        fake_B = G_A2B(real_A)
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)).to(device))
        fake_A = G_B2A(real_B)
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)).to(device))

        # Cycle consistency loss
        recov_A = G_B2A(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_A2B(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator A
        optimizer_D_A.zero_grad()
        loss_real_A = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)).to(device))
        loss_fake_A = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)).to(device))
        loss_D_A = (loss_real_A + loss_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # Train Discriminator B
        optimizer_D_B.zero_grad()
        loss_real_B = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)).to(device))
        loss_fake_B = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)).to(device))
        loss_D_B = (loss_real_B + loss_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss G: {loss_G.item()} - Loss D_A: {loss_D_A.item()} - Loss D_B: {loss_D_B.item()}")


# Generate synthetic images
os.makedirs("/path/to/save/synthetic_images", exist_ok=True)
for i, sample in enumerate(dataloader):
    real_A = sample['Anemic'].to(device)
    with torch.no_grad():
        fake_B = G_A2B(real_A)
    synthetic_image = transforms.ToPILImage()(fake_B.squeeze(0).cpu().detach())
    synthetic_image.save(f"/path/to/save/synthetic_images/Anemic_synthetic_{i+1}.png")
    if i >= 10:  # Save only a few examples for inspection
        break
print("Synthetic image generation completed.")