import torch
import torch.nn as nn

class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )

        self.layer1_t = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer2_t = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.layer3_t = nn.Sequential(
            nn.ConvTranspose2d(128, 16, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )
        self.layer4_t = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def decode(self, x):
        x = self.layer1_t(x)
        x = self.layer2_t(x)
        x = self.layer3_t(x)
        x = self.layer4_t(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out
    

autoencoder = ConvEncoder()




