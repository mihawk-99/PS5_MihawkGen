# PS5_FrameGen - the quality measures (docs/plan.md): PSNR and SSIM of frames in
# [0, 1], the same PSNR as the lab's (framegen/src/fg_common.cpp), and the
# Laplacian pyramid loss used in training.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import torch
import torch.nn.functional as F


def psnr(a, b):
    """Per-image PSNR in dB over RGB, the lab's measure: 10 log10(1 / mse)."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp(min=1e-10)
    return -10 * torch.log10(mse)


def _gauss(window=11, sigma=1.5, device='cpu'):
    x = torch.arange(window, device=device, dtype=torch.float32) - window // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, window, window)


def ssim(a, b):
    """Per-image mean SSIM (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5), RGB channels averaged."""
    c = a.shape[1]
    w = _gauss(device=a.device).expand(c, 1, -1, -1)
    mu_a = F.conv2d(a, w, groups=c)
    mu_b = F.conv2d(b, w, groups=c)
    saa = F.conv2d(a * a, w, groups=c) - mu_a ** 2
    sbb = F.conv2d(b * b, w, groups=c) - mu_b ** 2
    sab = F.conv2d(a * b, w, groups=c) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.flatten(1).mean(1)


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


def laplacian_loss(a, b, levels=4):
    """L1 between Laplacian pyramids: the error at every spatial scale."""
    loss = 0.0
    ca, cb = a, b
    for i in range(levels):
        da, db = F.avg_pool2d(ca, 2), F.avg_pool2d(cb, 2)
        la = ca - F.interpolate(da, size=ca.shape[-2:], mode='bilinear', align_corners=False)
        lb = cb - F.interpolate(db, size=cb.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + (2 ** i) * (la - lb).abs().mean()
        ca, cb = da, db
    return loss + (ca - cb).abs().mean() * 2 ** levels
