import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )


class GraspField(nn.Module):
    def __init__(self, channels=64, text_dim=512, grid=56):
        super().__init__()
        self.grid = grid
        self.projections = nn.ModuleList(
            [nn.Conv2d(n, channels, 1) for n in (128, 256, 512)]
        )
        self.fuse = ConvBlock(channels * 3 + 2, channels)
        self.film = nn.Sequential(
            nn.Linear(text_dim, 128), nn.SiLU(), nn.Linear(128, channels * 2)
        )
        self.refine = ConvBlock(channels, channels)
        self.head = nn.Conv2d(channels, 5, 1)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = -2
            self.head.bias[1] = 1
            self.head.bias[3] = -1.4
            self.head.bias[4] = -2.5

    def forward(self, features, text):
        size = features[0].shape[-2:]
        z = [
            F.interpolate(layer(f), size=size, mode="bilinear", align_corners=False)
            for f, layer in zip(features, self.projections)
        ]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size[0], device=text.device),
            torch.linspace(-1, 1, size[1], device=text.device),
            indexing="ij",
        )
        coordinates = torch.stack([xx, yy])[None].expand(text.shape[0], -1, -1, -1)
        visual = self.fuse(torch.cat([*z, coordinates], 1))
        gamma, beta = self.film(text).chunk(2, dim=1)
        fused = visual * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        fused = F.interpolate(
            fused, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        raw = self.head(self.refine(fused))
        geom = torch.cat([raw[:, 1:3], F.softplus(raw[:, 3:5])], 1)
        return raw[:, 0], geom


def field_loss(
    logits,
    geometry,
    quality,
    target_geometry,
    distractors,
    method="qcgf",
    rank_weight=0.2,
):
    assert method in {"mse", "bce", "focal", "qcgf", "vision_only"}, method
    mask = quality[:, None]
    angle = F.normalize(geometry[:, :2], dim=1, eps=1e-6)
    angle_loss = (
        (1 - (angle * target_geometry[:, :2]).sum(1)) * quality
    ).sum() / quality.sum().clamp_min(1)
    size_loss = (
        F.smooth_l1_loss(geometry[:, 2:], target_geometry[:, 2:], reduction="none")
        * mask
    ).sum() / (2 * mask.sum().clamp_min(1))
    if method == "mse":
        quality_loss = F.mse_loss(logits.sigmoid(), quality)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, quality, reduction="none")
        p = logits.sigmoid()
        focal = bce
        if method != "bce":
            focal = bce * (quality * (1 - p).square() + (1 - quality) * p.square())
        pos = (focal * quality).flatten(1).sum(1) / quality.flatten(1).sum(1).clamp_min(
            1
        )
        neg = (focal * (1 - quality)).flatten(1).sum(1) / (1 - quality).flatten(1).sum(
            1
        ).clamp_min(1)
        quality_loss = (pos + neg).mean()
    rank = logits.sum() * 0
    if method in {"qcgf", "vision_only"}:
        negative = distractors.bool() & ~quality.bool()
        eligible = negative.flatten(1).any(1) & quality.bool().flatten(1).any(1)
        if eligible.any():
            flat = logits[eligible].flatten(1)
            pm = quality[eligible].bool().flatten(1)
            nm = negative[eligible].flatten(1)
            # Log-mean-exp, rather than log-sum-exp, cancels support area bias.
            positive_score = (
                flat.masked_fill(~pm, -1e4).logsumexp(1) - pm.sum(1).float().log()
            )
            negative_score = (
                flat.masked_fill(~nm, -1e4).logsumexp(1) - nm.sum(1).float().log()
            )
            rank = F.softplus(negative_score - positive_score + 0.5).mean()
    total = quality_loss + angle_loss + 5 * size_loss + rank_weight * rank
    return total, {
        "total": total.detach(),
        "quality": quality_loss.detach(),
        "angle": angle_loss.detach(),
        "size": size_loss.detach(),
        "rank": rank.detach(),
    }
