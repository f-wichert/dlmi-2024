import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)  # Shape: (B, 64, H, W)
        self.enc2 = conv_block(64, 128)  # Shape: (B, 128, H/2, W/2)
        self.enc3 = conv_block(128, 256)  # Shape: (B, 256, H/4, W/4)
        self.enc4 = conv_block(256, 512)  # Shape: (B, 512, H/8, W/8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)  # Shape: (B, 1024, H/16, W/16)

        self.upconv4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )  # Shape: (B, 512, H/8, W/8)
        self.dec4 = conv_block(1024, 512)  # Shape: (B, 512, H/8, W/8)
        self.upconv3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )  # Shape: (B, 256, H/4, W/4)
        self.dec3 = conv_block(512, 256)  # Shape: (B, 256, H/4, W/4)
        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )  # Shape: (B, 128, H/2, W/2)
        self.dec2 = conv_block(256, 128)  # Shape: (B, 128, H/2, W/2)
        self.upconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )  # Shape: (B, 64, H, W)
        self.dec1 = conv_block(128, 64)  # Shape: (B, 64, H, W)

        self.final = nn.Conv2d(
            64, out_channels, kernel_size=1
        )  # Shape: (B, out_channels, H, W)

    def forward(self, x):
        enc1 = self.enc1(x)  # (B, 64, H, W)
        enc2 = self.enc2(self.pool(enc1))  # (B, 128, H/2, W/2)
        enc3 = self.enc3(self.pool(enc2))  # (B, 256, H/4, W/4)
        enc4 = self.enc4(self.pool(enc3))  # (B, 512, H/8, W/8)

        bottleneck = self.bottleneck(self.pool(enc4))  # (B, 1024, H/16, W/16)

        dec4 = self.dec4(
            torch.cat((self.upconv4(bottleneck), enc4), dim=1)
        )  # (B, 512, H/8, W/8)
        dec3 = self.dec3(
            torch.cat((self.upconv3(dec4), enc3), dim=1)
        )  # (B, 256, H/4, W/4)
        dec2 = self.dec2(
            torch.cat((self.upconv2(dec3), enc2), dim=1)
        )  # (B, 128, H/2, W/2)
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))  # (B, 64, H, W)

        return self.final(dec1)  # (B, out_channels, H, W)

