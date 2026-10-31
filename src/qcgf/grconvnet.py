"""GR-ConvNet + CLIP from the LGD author repository, run on this project's queries.

Vendored from https://github.com/Fsoft-AIC/LGD (MIT, Copyright (c) 2024 An Vuong),
commit 501904dc5c30c8fad58d90f7e4f49b4095f11eb0: inference/models/lgrconvnet3.py,
inference/models/grasp_model.py, utils/data/{language_grasp,grasp_anywhere}_data.py,
utils/dataset_processing/{image,grasp}.py and inference/post_process.py.

Layers, initialization, text fusion, loss, augmentation and post-processing follow
upstream. Changes:
* forward() takes the cached frozen CLIP ViT-B/32 text embedding, not raw strings.
* Upstream detaches visual features before adding text, so its encoder never
  trains; detach=False is our variant that trains it.
* Rectangles use this project's convention (x right, y down, theta clockwise,
  416 px); upstream's angle is the negative of ours, which only flips sin(2 theta).
* 90-degree rotations use exact array rotation; zoom crops and resizes once.
* Width is decoded with the training normalizer (output_size / 2) instead of
  upstream's hard-coded 150, which does not invert its own target scaling.
* If no peak exceeds upstream's 0.2 threshold, the global maximum is returned so
  every query receives a prediction (upstream would return no grasp).
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.transform import resize
from torch import nn

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


IMAGE_SIZE = 416


def wrap(angle):
    return (np.asarray(angle) + np.pi / 2) % np.pi - np.pi / 2


def rotate90(image, grasps, k):
    """Rotate counterclockwise by k * 90 degrees, as np.rot90 on (H, W)."""
    grasps = np.asarray(grasps, np.float64).reshape(-1, 5).copy()
    size = image.shape[0]
    for _ in range(k % 4):
        x, y = grasps[:, 0].copy(), grasps[:, 1].copy()
        grasps[:, 0], grasps[:, 1] = y, size - x
        # Direction (cos t, sin t) maps to (sin t, -cos t): t' = t - pi / 2.
        grasps[:, 4] = wrap(grasps[:, 4] - np.pi / 2)
    return np.rot90(image, k % 4).copy(), grasps


def zoom(image, grasps, factor):
    """Keep the centre `factor` of the image (upstream Image.zoom semantics)."""
    size = image.shape[0]
    start = int(size * (1 - factor)) // 2
    image = image[start : size - start, start : size - start]
    grasps = np.asarray(grasps, np.float64).reshape(-1, 5).copy()
    scale = size / (size - 2 * start)
    grasps[:, :2] = (grasps[:, :2] - start) * scale
    grasps[:, 2:4] *= scale
    return image, grasps


def preprocess(image, output_size=224):
    """Resize, scale to [0, 1], subtract the scalar mean, return CHW float32."""
    image = resize(image, (output_size, output_size), preserve_range=True)
    image = image.astype(np.float32) / 255.0
    image -= image.mean()
    return image.transpose(2, 0, 1)


def draw_targets(grasps, output_size=224, image_size=IMAGE_SIZE):
    """Central-third rectangles, later grasps overwrite earlier ones (upstream draw)."""
    pos = np.zeros((output_size, output_size), np.float32)
    ang = np.zeros_like(pos)
    width = np.zeros_like(pos)
    scale = output_size / image_size
    for x, y, w, h, t in np.asarray(grasps, np.float64).reshape(-1, 5):
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        c, s = np.cos(t), np.sin(t)
        half = np.array(
            [[-w / 6, -h / 2], [w / 6, -h / 2], [w / 6, h / 2], [-w / 6, h / 2]]
        )
        xs = x + half[:, 0] * c - half[:, 1] * s
        ys = y + half[:, 0] * s + half[:, 1] * c
        rr, cc = polygon(ys, xs, pos.shape)
        pos[rr, cc] = 1.0
        ang[rr, cc] = t
        width[rr, cc] = w
    width = np.clip(width, 0.0, output_size / 2) / (output_size / 2)
    return np.stack([pos, np.cos(2 * ang), np.sin(2 * ang), width]).astype(np.float32)


def decode(pos, cos, sin, width, topk=1, output_size=224, image_size=IMAGE_SIZE):
    """Upstream post_process_output + detect_grasps, mapped back to 416 px."""
    q = gaussian(pos, 2.0, preserve_range=True)
    ang = gaussian(np.arctan2(sin, cos) / 2.0, 2.0, preserve_range=True)
    width = gaussian(width * (output_size / 2), 1.0, preserve_range=True)
    peaks = peak_local_max(q, min_distance=20, threshold_abs=0.2, num_peaks=topk)
    if not len(peaks):
        peaks = np.array([np.unravel_index(np.argmax(q), q.shape)])
    scale = image_size / output_size
    out = []
    for r, c in peaks:
        length = max(float(width[r, c]), 1e-3) * scale
        out.append([c * scale, r * scale, length, length / 2, float(ang[r, c])])
    return np.asarray(out, np.float32)
