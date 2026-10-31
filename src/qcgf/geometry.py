"""Image-plane rectangles: x right, y down, theta clockwise in radians, period pi."""

import cv2
import numpy as np


def angular_error(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi / 2) % np.pi - np.pi / 2)


def corners(grasp):
    x, y, w, h, angle = np.asarray(grasp, dtype=float)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    c, s = np.cos(angle), np.sin(angle)
    return (pts @ np.array([[c, s], [-s, c]]) + [x, y]).astype(np.float32)


def rotated_iou(a, b):
    if min(a[2:4]) <= 0 or min(b[2:4]) <= 0:
        return 0.0
    pa, pb = corners(a), corners(b)
    area, _ = cv2.intersectConvexConvex(pa, pb)
    union = a[2] * a[3] + b[2] * b[3] - area
    return float(np.clip(area / max(union, 1e-9), 0, 1))


def success(prediction, ground_truth, iou_threshold=0.25, angle_threshold=np.pi / 6):
    return any(
        angular_error(prediction[4], g[4]) < angle_threshold
        and rotated_iou(prediction, g) > iou_threshold
        for g in ground_truth
    )


def make_targets(grasps, grid=56, image_size=416):
    grasps = np.asarray(grasps, np.float32).reshape(-1, 5).copy()
    grasps[:, :4] *= grid / image_size
    yy, xx = np.mgrid[:grid, :grid].astype(np.float32)
    xx += 0.5
    yy += 0.5
    positive = np.zeros((grid, grid), np.float32)
    geometry = np.zeros((4, grid, grid), np.float32)
    best = np.full((grid, grid), np.inf, np.float32)
    for x, y, w, h, t in grasps:
        dx, dy = xx - x, yy - y
        c, s = np.cos(t), np.sin(t)
        along, across = c * dx + s * dy, -s * dx + c * dy
        inside = (np.abs(along) <= max(w / 6, 0.65)) & (
            np.abs(across) <= max(h / 2, 0.65)
        )
        dist = dx * dx + dy * dy
        update = inside & (dist < best)
        positive[inside] = 1
        geometry[0, update] = np.cos(2 * t)
        geometry[1, update] = np.sin(2 * t)
        geometry[2, update] = np.clip(w / grid, 0.005, 2)
        geometry[3, update] = np.clip(h / grid, 0.005, 2)
        best[update] = dist[update]
    return positive, geometry


def decode(quality, geometry, image_size=416, topk=1):
    from scipy.ndimage import gaussian_filter, maximum_filter

    quality = gaussian_filter(np.asarray(quality), 0.7)
    peaks = quality == maximum_filter(quality, size=5)
    candidates = np.flatnonzero(peaks)
    candidates = candidates[np.argsort(quality.ravel()[candidates])[::-1]][:topk]
    out = []
    grid = quality.shape[-1]
    for index in candidates:
        y, x = np.unravel_index(index, quality.shape)
        c, s, w, h = geometry[:, y, x]
        out.append(
            [
                (x + 0.5) * image_size / grid,
                (y + 0.5) * image_size / grid,
                np.clip(w, 0.005, 2) * image_size,
                np.clip(h, 0.005, 2) * image_size,
                0.5 * np.arctan2(s, c),
            ]
        )
    return np.asarray(out, np.float32)
