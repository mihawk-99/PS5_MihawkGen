# PS5_FrameGen - writes a trained network as the weight file the generator
# loads (framegen/src/fg_neural.cpp reads it).
#
#   .venv/bin/python train/export.py train/runs/v2/ckpt.pt model.fgw
#
# Format, little-endian ("FGW1"):
#   u32 magic 0x31574746, u32 blocks, then per block: u32 scale, u32 width, u32 convs, u32 radius
#   then per convolution, in network order: u32 cin, u32 cout, u32 leaky (1) or linear (0),
#     f32 weights [og][tap][ig][j][i], f32 bias [og][4]
# with channels padded with zeros to multiples of 4: input channel ig*4+j, output
# channel og*4+i, tap = ky*3 + kx. Each (og, tap, ig) is a 4x4 matrix stored with the
# input channel as the column, as GLSL's mat4 is, so the shader multiplies it by the
# input's vec4.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Interpolator  # noqa: E402

MAGIC = 0x31574746


def pack_conv(conv, leaky):
    w = conv.weight.detach().float().cpu().numpy()          # cout x cin x 3 x 3
    b = conv.bias.detach().float().cpu().numpy()
    cout, cin = w.shape[:2]
    ci, co = (cin + 3) // 4, (cout + 3) // 4
    wp = np.zeros((co * 4, ci * 4, 3, 3), np.float32)
    wp[:cout, :cin] = w
    bp = np.zeros(co * 4, np.float32)
    bp[:cout] = b
    # [og][tap][ig][j][i] = wp[og*4+i][ig*4+j][ky][kx]
    out = np.zeros((co, 9, ci, 4, 4), np.float32)
    for og in range(co):
        for ky in range(3):
            for kx in range(3):
                for ig in range(ci):
                    block = wp[og * 4:og * 4 + 4, ig * 4:ig * 4 + 4, ky, kx]    # [i][j]
                    out[og, ky * 3 + kx, ig] = block.T                          # [j][i]
    return struct.pack('<III', cin, cout, 1 if leaky else 0) + out.tobytes() + bp.tobytes()


def export(model, path):
    blob = struct.pack('<II', MAGIC, len(model.blocks_cfg))
    for scale, width, convs, radius in model.blocks_cfg:
        blob += struct.pack('<IIII', scale, width, convs, radius)
    for block in model.blocks:
        layers = [m for m in block.net if isinstance(m, torch.nn.Conv2d)]
        for i, conv in enumerate(layers):
            blob += pack_conv(conv, leaky=i < len(layers) - 1)
    with open(path, 'wb') as f:
        f.write(blob)
    return len(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', help="a checkpoint, or 'random' for untrained weights (for timing only)")
    ap.add_argument('output')
    ap.add_argument('--raw', action='store_true', help='export the raw weights, not the EMA')
    args = ap.parse_args()
    if args.checkpoint == 'random':
        torch.manual_seed(0)
        model = Interpolator()
    else:
        state = torch.load(args.checkpoint, map_location='cpu')
        model = Interpolator([tuple(b) for b in state['blocks']])
        model.load_state_dict(state['model' if args.raw else 'ema'])
    print(f'{args.output}: {export(model, args.output):,} bytes, {model.count():,} parameters')


if __name__ == '__main__':
    main()
