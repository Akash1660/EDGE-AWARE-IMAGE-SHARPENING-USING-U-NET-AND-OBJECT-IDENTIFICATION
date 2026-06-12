import torch
import torch.nn as nn


class UNetSmall(nn.Module):
    """Lightweight U-Net for edge-aware image sharpening.

    Input: 5-channel tensor (3 RGB + 1 edge map + 1 segmentation mask)
    Output: 3-channel RGB sharpened image, range [-1, 1]
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(True),
            )

        def up_block(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, 2, 2)

        # Encoder
        self.e1, self.p1 = conv_block(5, 32), nn.MaxPool2d(2)
        self.e2, self.p2 = conv_block(32, 64), nn.MaxPool2d(2)
        self.e3, self.p3 = conv_block(64, 128), nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(128, 256)

        # Decoder
        self.u3, self.d3 = up_block(256, 128), conv_block(256, 128)
        self.u2, self.d2 = up_block(128, 64), conv_block(128, 64)
        self.u1, self.d1 = up_block(64, 32), conv_block(64, 32)

        # Output
        self.out = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))

        b = self.bottleneck(self.p3(e3))

        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))

        return self.out(d1)
