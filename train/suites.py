# PS5_FrameGen - scores a checkpoint on held-out suites, one for each way an
# interpolator fails (docs/plan.md, phase 8): fast motion, almost none, a HUD over a
# moving world, translucent and added layers, spinning, dollying, tilted planes,
# many occluding layers, and a scene cut (the answer is the frame nearer in time). Each suite is made of held-out scenes (a fixed seed no
# training batch uses) with the preset's changes to the scene's parameters, scored
# against blend and repeat on the same frames.
#
#   .venv/bin/python train/suites.py train/runs/v3/ckpt.pt [--clips 16]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import psnr, ssim  # noqa: E402
from model import Interpolator  # noqa: E402
from synth import PRESETS, Synth  # noqa: E402


@torch.no_grad()
def score(model, frames):
    """(model, blend, repeat) PSNR and the model's SSIM, averaged over gap 2 and gap 4 samples."""
    rows = []
    for gap, start, mid in ((2, 0, 1), (2, 2, 3), (4, 0, 1), (4, 0, 2), (4, 0, 3)):
        a, c, m = frames[:, start], frames[:, start + gap], frames[:, mid]
        t = (mid - start) / gap
        out = model(a, c, torch.full((a.shape[0],), t, device=a.device)).clamp(0, 1)
        blend = a * (1 - t) + c * t
        rows.append([psnr(out, m).mean().item(), psnr(blend, m).mean().item(), psnr(a if t < 0.5 else c, m).mean().item(),
                     ssim(out, m).mean().item()])
    return torch.tensor(rows).mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--clips', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    state = torch.load(args.checkpoint, map_location=args.device)
    model = Interpolator([tuple(b) for b in state['blocks']]).to(args.device).eval()
    model.load_state_dict(state['ema'])
    synth = Synth(args.device, bank=512)
    print(f'{"suite":<14}{"model":>8}{"blend":>8}{"repeat":>8}{"gain":>7}{"ssim":>7}')
    for name in [None] + sorted(PRESETS):
        gen = torch.Generator(device=args.device)
        gen.manual_seed(13579)
        frames = []
        for i in range(0, args.clips, 4):
            frames.append(synth.render(4, gen, preset=name).frames)
        scores = torch.tensor([score(model, f) for f in frames]).mean(0).tolist()
        print(f'{name or "general":<14}{scores[0]:8.2f}{scores[1]:8.2f}{scores[2]:8.2f}{scores[0] - scores[1]:+7.2f}{scores[3]:7.3f}')
    # A cut: the second frame is from another scene, and the truth is the nearer frame
    gen = torch.Generator(device=args.device)
    gen.manual_seed(24680)
    rows = []
    for _ in range(max(1, args.clips // 4)):
        frames = synth.render(4, gen).frames
        other = frames.roll(1, 0)
        for t in (0.25, 0.5, 0.75):
            a, c = frames[:, 0], other[:, 4]
            out = model(a, c, torch.full((a.shape[0],), t, device=a.device)).clamp(0, 1)
            nearest = a if t < 0.5 else c
            rows.append([psnr(out, nearest).mean().item(), psnr(a * (1 - t) + c * t, nearest).mean().item()])
    m = torch.tensor(rows).mean(0).tolist()
    print(f'{"cut":<14}{m[0]:8.2f}{m[1]:8.2f}{"":8}{m[0] - m[1]:+7.2f}   (against the nearer frame)')


if __name__ == '__main__':
    main()
