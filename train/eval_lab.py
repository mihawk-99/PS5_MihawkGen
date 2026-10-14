# PS5_FrameGen - scores a checkpoint on frames exported by FrameGen Lab
# (FG_EXPORT_DIR, examples/framegenlab): a rendered scene the network never saw,
# with the true in-between frames. The 4K frames are reduced by area to the
# network's working size (a factor of 8, 480x270) so the scores are comparable
# to training's.
#
#   .venv/bin/python train/eval_lab.py train/runs/v1/ckpt.pt build/export-lab [--save picture.png]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import psnr, ssim  # noqa: E402
from model import Interpolator  # noqa: E402


def load_frames(folder, factor, device):
    """All the exported frames (color_*.rgba), 3 x H x W floats in [0, 1], reduced by `factor` (area)."""
    files = sorted(glob.glob(os.path.join(folder, 'color_*.rgba')))
    frames = []
    for f in files:
        raw = np.fromfile(f, dtype=np.uint8)
        size = raw.size // 4
        # the lab's size is 16:9; recover it from the pixel count
        h = int(round((size * 9 / 16) ** 0.5))
        w = size // h
        img = torch.from_numpy(raw.reshape(h, w, 4)[..., :3].copy()).to(device).permute(2, 0, 1).float() / 255
        frames.append(F.avg_pool2d(img.unsqueeze(0), factor))
    return torch.cat(frames)


def pad8(x):
    h, w = x.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    return F.pad(x, (0, pw, 0, ph), mode='replicate'), h, w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('folder')
    ap.add_argument('--save')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--size', default='288x512', help='working size HxW the frames are reduced to')
    args = ap.parse_args()
    device = args.device
    state = torch.load(args.checkpoint, map_location=device)
    model = Interpolator([tuple(b) for b in state['blocks']]).to(device)
    model.load_state_dict(state['ema'])
    model.eval()
    frames = load_frames(args.folder, 1, device)
    h0, w0 = (int(x) for x in args.size.split('x'))
    frames = F.interpolate(frames, size=(h0, w0), mode='area')
    n = frames.shape[0]
    print(f'{n} frames at {frames.shape[3]}x{frames.shape[2]}')
    rows = {2: [], 4: []}
    best = None
    for gap in (2, 4):
        for start in range(0, n - gap):
            for mid in range(start + 1, start + gap):
                t = (mid - start) / gap
                a, c, m = frames[start:start + 1], frames[start + gap:start + gap + 1], frames[mid:mid + 1]
                pa, h, w = pad8(a)
                pc, _, _ = pad8(c)
                out = model(pa, pc, torch.tensor([t], device=device))[..., :h, :w].clamp(0, 1)
                blend = a * (1 - t) + c * t
                rows[gap].append((psnr(out, m).item(), psnr(blend, m).item(), psnr(a if t < 0.5 else c, m).item(),
                                  ssim(out, m).item(), ssim(blend, m).item()))
                if gap == 2 and start == 8:
                    best = (a, out, blend, m, c)
    for gap, vals in rows.items():
        v = np.array(vals).mean(0)
        print(f'gap {gap}: model {v[0]:.2f} dB ssim {v[3]:.3f} | blend {v[1]:.2f} dB ssim {v[4]:.3f} | repeat {v[2]:.2f} dB')
    if args.save and best:
        from PIL import Image
        a, out, blend, m, c = best
        sheet = torch.cat([torch.cat([a, c], 3), torch.cat([out, m], 3), torch.cat([blend, (out - m).abs() * 4], 3)], 2)
        Image.fromarray((sheet[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(args.save)


if __name__ == '__main__':
    main()
