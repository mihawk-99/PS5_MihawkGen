# PS5_FrameGen - checks that the compute shaders reproduce the network in PyTorch
# (docs/plan.md, phase 6): the same two frames and t through both, the outputs compared.
# PyTorch is the oracle; the shaders store feature maps in fp16, so the two agree to a
# small difference, not bit for bit.
#
#   .venv/bin/python train/parity.py train/runs/v2/ckpt.pt [--device NVIDIA] [--clips 4]
#
# Needs build/neural_run (framegen/tests/neural_run.cpp, built by tools/lab.sh tests).
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from export import export  # noqa: E402
from metrics import psnr  # noqa: E402
from model import Interpolator  # noqa: E402
from synth import Synth  # noqa: E402


def to_rgba(x):
    """A 3 x H x W float frame in [0, 1] as 8-bit RGBA bytes, and the 8-bit frame back as float."""
    q = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    rgba = torch.cat([q, torch.full_like(q[:1], 255)], 0).permute(1, 2, 0).contiguous().cpu().numpy()
    return rgba, q.float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--device', default=None, help='substring of the Vulkan device name (FG_DEVICE)')
    ap.add_argument('--clips', type=int, default=4)
    ap.add_argument('--runner', default=os.path.join(ROOT, 'build', 'neural_run'))
    ap.add_argument('--weights', default=None, help='keep the exported weights here')
    ap.add_argument('--lab', default=None, help='use FrameGen Lab frames exported to this folder, reduced to 512x288, and run on the CPU')
    args = ap.parse_args()
    dev = 'cpu' if args.lab else 'cuda'
    state = torch.load(args.checkpoint, map_location=dev)
    model = Interpolator([tuple(b) for b in state['blocks']]).to(dev).eval()
    model.load_state_dict(state['ema'])
    if args.lab:
        from eval_lab import load_frames
        lab = torch.nn.functional.interpolate(load_frames(args.lab, 1, 'cpu')[:12], size=(288, 512), mode='area')
        clips = lab.unsqueeze(0).repeat(1, 1, 1, 1, 1)           # 1 clip of 12 steps
        args.clips = 1
        cases = ((2, 0, 1), (2, 3, 4), (4, 1, 2), (4, 1, 3), (4, 1, 4))
    else:
        synth = Synth('cuda', bank=192)
        gen = torch.Generator(device='cuda')
        gen.manual_seed(2024)
        clips = synth.render(args.clips, gen).frames
        cases = ((2, 0, 1), (4, 0, 1), (4, 0, 2), (4, 0, 3))
    env = dict(os.environ)
    if args.device:
        env['FG_DEVICE'] = args.device
    worst = 1e9
    with tempfile.TemporaryDirectory() as tmp:
        weights = args.weights or os.path.join(tmp, 'net.fgw')
        export(model.cpu(), weights)
        model.to(dev)
        for i in range(args.clips):
            for gap, start, mid in cases:
                a, b = clips[i, start], clips[i, start + gap]
                t = (mid - start) / gap
                ra, qa = to_rgba(a)
                rb, qb = to_rgba(b)
                h, w = ra.shape[:2]
                ra.tofile(os.path.join(tmp, 'a.rgba')); rb.tofile(os.path.join(tmp, 'b.rgba'))
                subprocess.run([args.runner, weights, os.path.join(tmp, 'a.rgba'), os.path.join(tmp, 'b.rgba'), str(w), str(h), str(t),
                                os.path.join(tmp, 'out.rgba')], check=True, env=env, stderr=subprocess.DEVNULL)
                vk = torch.from_numpy(np.fromfile(os.path.join(tmp, 'out.rgba'), np.uint8).reshape(h, w, 4)[..., :3].copy()).to(dev).permute(2, 0, 1).float() / 255
                with torch.no_grad():
                    ref = model(qa.unsqueeze(0).to(dev), qb.unsqueeze(0).to(dev), torch.tensor([t], device=dev))[0].clamp(0, 1)
                _, ref8 = to_rgba(ref)
                truth = clips[i, mid].to(dev)
                p_agree = psnr(vk.unsqueeze(0), ref8.unsqueeze(0)).item()
                worst = min(worst, p_agree)
                print(f'clip {i} gap {gap} t {t:.2f}: shaders vs PyTorch {p_agree:.1f} dB | '
                      f'vs truth: shaders {psnr(vk.unsqueeze(0), truth.unsqueeze(0)).item():.2f} PyTorch {psnr(ref8.unsqueeze(0), truth.unsqueeze(0)).item():.2f} dB')
    print(f'VERDICT {"PASS" if worst > 40 else "FAIL"}: the shaders agree with PyTorch to at least {worst:.1f} dB (limit 40)')
    return 0 if worst > 40 else 1


if __name__ == '__main__':
    sys.exit(main())
