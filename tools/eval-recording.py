#!/usr/bin/env python3
"""PS5_FrameGen - scores the generator's shaders (build/neural_run, the whole pipeline on the PC's Vulkan device) on the real
pictures of a present-stream recording: for real pictures k, k+1, k+2 (one for each new frame) the generator makes the one
between k and k+2 at t = 1/2 and it is scored against the real picture k+1 (PSNR, and the blend of k and k+2 for comparison).
The pictures are the game's, cropped of the screen's bars; at full recording size (set the width with
/app0/framegen-record-width.txt on the console) the final pass is exercised as it is on the console, which the 288-row
benchmark (train/bench.py) does not.

    tools/eval-recording.py REC_DIR [REC_DIR ...] [--weights FILE] [--max 12] [--save DIR]

Copyright (C) 2026 Mihawk
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
spec = importlib.util.spec_from_file_location('analyze_stream', os.path.join(here, 'analyze-stream.py'))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def sharpness(img):
    """The mean absolute brightness gradient of a picture: what a soft picture has less of."""
    g = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recs', nargs='+')
    ap.add_argument('--weights', default=os.path.join(root, 'framegen', 'models', 'network.fgw'))
    ap.add_argument('--binary', default=os.path.join(root, 'build', 'neural_run'))
    ap.add_argument('--max', type=int, default=12, help='triplets for each recording')
    ap.add_argument('--save', help='folder for a picture of each triplet: first, truth, generated, blend, error')
    args = ap.parse_args()
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    allp, allb, allsh = [], [], []
    for rec in args.recs:
        frames, rows = analyze.load(rec)
        keep, last = [], None
        for i, r in enumerate(rows):
            if not r['gen'] and r['unique'] != last:
                keep.append(i)
                last = r['unique']
        if len(keep) < 3:
            print(rec, 'too few real pictures')
            continue
        real = frames[keep]
        cols = np.where(real.max(axis=(0, 1)).max(axis=-1) > 12)[0]
        rws = np.where(real.max(axis=(0, 2)).max(axis=-1) > 12)[0]
        x0, x1 = (cols.min(), cols.max() + 1) if len(cols) else (0, real.shape[2])
        y0, y1 = (rws.min(), rws.max() + 1) if len(rws) else (0, real.shape[1])
        real = real[:, y0:y1, x0:x1]
        h, w = real.shape[1:3]
        picks = sorted(set(int(round(x)) for x in np.linspace(0, len(real) - 3, min(args.max, len(real) - 2))))
        ps, bs, shs = [], [], []
        with tempfile.TemporaryDirectory() as td:
            for j in picks:
                a, truth, c = real[j], real[j + 1], real[j + 2]
                for name, img in (('a', a), ('c', c)):
                    rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
                    rgba.tofile(os.path.join(td, name + '.rgba'))
                out = os.path.join(td, 'out.rgba')
                subprocess.run([args.binary, args.weights, os.path.join(td, 'a.rgba'), os.path.join(td, 'c.rgba'), str(w), str(h), '0.5', out],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                gen = np.fromfile(out, np.uint8).reshape(h, w, 4)[:, :, :3]
                blend = ((a.astype(np.float32) + c.astype(np.float32)) / 2).round().astype(np.uint8)
                ps.append(psnr(gen, truth))
                shs.append(sharpness(gen) / max(sharpness(truth), 1e-6))
                bs.append(psnr(blend, truth))
                if args.save:
                    err = np.abs(gen.astype(int) - truth.astype(int)).sum(-1).clip(0, 255).astype(np.uint8)
                    tile = np.concatenate([a, truth, gen, blend, np.stack([err] * 3, -1)], 1)
                    Image.fromarray(tile).save(os.path.join(args.save, '%s_%02d.png' % (os.path.basename(os.path.dirname(rec.rstrip('/'))), j)))
        print('%s: generated %.2f dB, blend %.2f dB, sharpness %.3f of the real picture (%d triplets, %dx%d)' % (rec, np.mean(ps), np.mean(bs), np.mean(shs), len(ps), w, h))
        allsh += shs
        allp += ps
        allb += bs
    if allp:
        print('all: generated %.2f dB, blend %.2f dB, sharpness %.3f (%d triplets)' % (np.mean(allp), np.mean(allb), np.mean(allsh), len(allp)))


if __name__ == '__main__':
    sys.exit(main())
