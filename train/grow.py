# PS5_FrameGen - grows a checkpoint's blocks by convolutions: the old hidden convolutions stay first, the new ones start as the identity
# (centre tap one) before the output convolution, so the grown network starts close to the old one and training takes it on from there.
#
#   .venv/bin/python train/grow.py runs/d5/ckpt.pt runs/g0/init.pt '[[16,64,8,3],[8,64,7,2],[4,48,6,2],[2,32,4,1]]'
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os
import sys

import torch


def grow(sd, old, new, noise=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        if not k.startswith('blocks.'):
            out[k] = v.clone()
    for b, (ob, nb) in enumerate(zip(old, new)):
        assert list(ob[:2]) == list(nb[:2]) and ob[3] == nb[3] and nb[2] >= ob[2], (ob, nb)
        width = ob[1]
        for j in range(ob[2]):
            for t in ('weight', 'bias'):
                out[f'blocks.{b}.net.{2 * j}.{t}'] = sd[f'blocks.{b}.net.{2 * j}.{t}'].clone()
        for j in range(ob[2], nb[2]):
            w = torch.randn(width, width, 3, 3, generator=g) * noise
            w[torch.arange(width), torch.arange(width), 1, 1] += 1.0
            out[f'blocks.{b}.net.{2 * j}.weight'] = w
            out[f'blocks.{b}.net.{2 * j}.bias'] = torch.zeros(width)
        for t in ('weight', 'bias'):
            out[f'blocks.{b}.net.{2 * nb[2]}.{t}'] = sd[f'blocks.{b}.net.{2 * ob[2]}.{t}'].clone()
    return out


if __name__ == '__main__':
    src, dst, blocks = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    s = torch.load(src, map_location='cpu', weights_only=False)
    old = s['blocks']
    new = [tuple(b) for b in blocks]
    s['model'] = grow(s['model'], old, new)
    s['ema'] = grow(s['ema'], old, new)
    s['blocks'] = new
    s.pop('opt', None)
    s['step'] = 0
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(s, dst)
    print('grown', old, '->', new)
