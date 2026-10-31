"""Upstream GR-ConvNet + CLIP data pipeline on this project's queries.

Follows the LGD repository (commit in grconvnet.UPSTREAM_COMMIT):
utils/data/language_grasp_data.py, utils/data/grasp_anywhere_data.py,
utils/dataset_processing/{image,grasp}.py and inference/post_process.py.

Rectangles use this project's convention (x right, y down, theta clockwise,
416 px); upstream's angle is the negative of ours, which only flips the sign of
the sin(2 theta) channel. Documented adaptations:
* 90-degree rotations use exact array rotation; zoom crops and resizes once.
* Width is decoded with the training normalizer (output_size / 2) instead of
  upstream's hard-coded 150, which does not invert its own target scaling.
* If no peak exceeds upstream's 0.2 threshold, the global maximum is returned so
  every query receives a prediction (upstream would return no grasp).
"""

import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.transform import resize

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
