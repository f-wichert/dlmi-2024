import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=4, pool_factor=2):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels = []
        enc_out = 32

        for i in range(depth):
            enc_out = enc_out * pool_factor
            if i == 0:
                self.encoder_blocks.append(conv_block(in_channels, enc_out))
            else:
                self.encoder_blocks.append(conv_block(int(enc_out / pool_factor), enc_out))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(enc_out)

        self.bottleneck = conv_block(enc_out, enc_out * pool_factor)

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    enc_out * pool_factor if i == depth - 1 else enc_channels[i + 1],
                    enc_channels[i],
                    kernel_size=pool_factor,
                    stride=pool_factor
                )
            )
            self.decoder_blocks.append(conv_block(enc_channels[i] * 2, enc_channels[i]))

        self.final = nn.Conv2d(enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []

        for block, pool in zip(self.encoder_blocks, self.pools):
            x = block(x)
            encoder_outputs.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            enc_output = encoder_outputs[-(i + 1)]
            x = torch.cat([x, enc_output], dim=1)
            x = dec_block(x)

        return self.final(x)    # (B, out_channels, H, W)

