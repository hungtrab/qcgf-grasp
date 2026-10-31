"""GR-ConvNet + CLIP as released in the LGD author repository.

Vendored from https://github.com/Fsoft-AIC/LGD (MIT, Copyright (c) 2024 An Vuong),
commit 501904dc5c30c8fad58d90f7e4f49b4095f11eb0, files
inference/models/lgrconvnet3.py and inference/models/grasp_model.py.

Layers, initialization, text projection, fusion and loss are unchanged. The only
interface change: forward() receives the frozen CLIP ViT-B/32 text embedding
instead of raw strings, so this project's cached embedding is reused. Upstream
detaches the visual features before adding text (`torch.clone(x).detach()`), so
the visual encoder layers receive no gradient; that behavior is kept on purpose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class GenerativeResnet(nn.Module):
    def __init__(
        self, input_channels=3, channel_size=32, dropout=True, prob=0.1, detach=True
    ):
        super().__init__()
        c = channel_size
        self.detach = detach
        self.conv1 = nn.Conv2d(input_channels, c, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.res1 = ResidualBlock(c * 4, c * 4)
        self.res2 = ResidualBlock(c * 4, c * 4)
        self.res3 = ResidualBlock(c * 4, c * 4)
        self.res4 = ResidualBlock(c * 4, c * 4)
        self.res5 = ResidualBlock(c * 4, c * 4)
        self.conv4 = nn.ConvTranspose2d(
            c * 4, c * 2, kernel_size=4, stride=2, padding=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(c * 2)
        self.conv5 = nn.ConvTranspose2d(
            c * 2, c, kernel_size=4, stride=2, padding=2, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, kernel_size=9, stride=1, padding=4)
        self.y_flatten = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 56),
            nn.GELU(),
        )
        self.pos_output = nn.Conv2d(c, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(c, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(c, 1, kernel_size=2)
        self.width_output = nn.Conv2d(c, 1, kernel_size=2)
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in, text):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
        y = self.y_flatten(text)
        y = y.unsqueeze(2).expand(-1, -1, 56).unsqueeze(1).expand(-1, 128, -1, -1)
        # Upstream detaches; detach=False is our variant that trains the encoder.
        x = (torch.clone(x).detach() if self.detach else x) + y
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        if self.dropout:
            return (
                self.pos_output(self.dropout_pos(x)),
                self.cos_output(self.dropout_cos(x)),
                self.sin_output(self.dropout_sin(x)),
                self.width_output(self.dropout_wid(x)),
            )
        return (
            self.pos_output(x),
            self.cos_output(x),
            self.sin_output(x),
            self.width_output(x),
        )

    def compute_loss(self, x, targets, text):
        pred = self(x, text)
        losses = {
            name: F.smooth_l1_loss(p, t)
            for name, p, t in zip(
                ["p_loss", "cos_loss", "sin_loss", "width_loss"], pred, targets
            )
        }
        return sum(losses.values()), losses
