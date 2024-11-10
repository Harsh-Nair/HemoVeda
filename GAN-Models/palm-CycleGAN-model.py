import os
import time
import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset
class PalmDataset(Dataset):
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

# DataLoader
anemic_dir = Path('/path/to/your/anemic/data')
healthy_dir = Path('/path/to/your/healthy/data')
dataset = PalmDataset(anemic_dir, healthy_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# Define Generator using ResNet blocks
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        assert n_blocks >= 0
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Residual blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
    
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
# Initialize models
G_A2B = ResNetGenerator().to(device)
G_B2A = ResNetGenerator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# Loss and Optimizers
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))


num_epochs = 100
start_time = time.time()
max_duration = 12 * 60 * 60  # 12 hours in seconds

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        if time.time() - start_time > max_duration:
            print("Training stopped after 11 hours.")
            torch.save(G_A2B.state_dict(), "G_A2B.pth")
            torch.save(G_B2A.state_dict(), "G_B2A.pth")
            torch.save(D_A.state_dict(), "D_A.pth")
            torch.save(D_B.state_dict(), "D_B.pth")
            exit()

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
    print(f"Epoch [{epoch+1}/{num_epochs}] completed.")

# Saving model after training
torch.save(G_A2B.state_dict(), "G_A2B_final.pth")
torch.save(G_B2A.state_dict(), "G_B2A_final.pth")
torch.save(D_A.state_dict(), "D_A_final.pth")
torch.save(D_B.state_dict(), "D_B_final.pth")


# Generate synthetic images
os.makedirs("/kaggle/working/synthetic_images", exist_ok=True)
for i, sample in enumerate(dataloader):
    real_A = sample['Anemic'].to(device)
    with torch.no_grad():
        fake_B = G_A2B(real_A)
    synthetic_image = transforms.ToPILImage()(fake_B.squeeze(0).cpu().detach())
    synthetic_image.save(f"/path/to/your/synthetic_images/Anemic_synthetic_{i+1}.png")
    if i >= 10:
        break
print("Synthetic image generation completed.")