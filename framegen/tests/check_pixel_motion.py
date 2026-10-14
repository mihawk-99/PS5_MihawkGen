#!/usr/bin/env python3
# PS5_FrameGen - the pixel-art motion pass on a made-up picture (no game content): a field of random coloured tiles with a
# few sprite-like blobs, scrolled by whole pixels between two frames, and a HUD rectangle that does not change. The generated
# frame at t = 0.5 must be the first frame moved by half the scroll (exact, outside the HUD), and the HUD must stay where it is.
#
#   framegen/tests/check_pixel_motion.py BUILD/neural_run NETWORK.fgw
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import subprocess
import sys
import tempfile

import numpy as np

K = 4
W, H = 192, 112


def picture(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)      # a colour of its own for every pixel, so that a position can be told from its neighbours
    for _ in range(40):                      # sprite-like blobs: small blocks of other colours
        y, x, h, w = rng.integers(0, H - 12), rng.integers(0, W - 12), rng.integers(4, 11), rng.integers(4, 11)
        img[y:y + h, x:x + w] = rng.integers(0, 256, 3, dtype=np.uint8)
        img[y + 1:y + h - 1, x + 1:x + w - 1] = rng.integers(0, 256, 3, dtype=np.uint8)
    return img


def up(a):
    return np.ascontiguousarray(np.repeat(np.repeat(a, K, 0), K, 1))


def rgba(a):
    return np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], 2)


def main():
    runner, network = sys.argv[1], sys.argv[2]
    hud = picture(99)[:14, :60].copy()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for dx, dy in ((2, 0), (0, 4), (4, 2), (6, -2), (-8, 0)):
            a = picture(1)
            c = np.roll(a, (dy, dx), (0, 1))
            a[:14, :60] = hud                # a HUD in both frames, the same
            c[:14, :60] = hud
            rgba(up(a)).tofile(f'{tmp}/a.rgba')
            rgba(up(c)).tofile(f'{tmp}/c.rgba')
            env = dict(os.environ, FG_CRISP='5', FG_SOURCE=f'{W},{H}')
            subprocess.run([runner, network, f'{tmp}/a.rgba', f'{tmp}/c.rgba', str(W * K), str(H * K), '0.5', f'{tmp}/o.rgba', '8'],
                           check=True, capture_output=True, env=env)
            g = np.fromfile(f'{tmp}/o.rgba', np.uint8).reshape(H * K, W * K, 4)[:, :, :3].astype(int)
            ua = up(a).astype(int)
            expected = np.roll(ua, (dy * K // 2, dx * K // 2), (0, 1))
            body = (slice(24 * K, -20 * K), slice(72 * K, -24 * K))
            exact = float((np.abs(g[body] - expected[body]).max(2) == 0).mean())
            hud_ok = float((np.abs(g[:14 * K, :60 * K] - ua[:14 * K, :60 * K]).max(2) == 0).mean())
            print(f'scroll ({dx:+d},{dy:+d}): the field is the half-moved first frame in {exact * 100:.1f}% of its pixels, the HUD is as it was in {hud_ok * 100:.1f}%')
            if exact < 0.97:
                failures.append(f'scroll ({dx},{dy}): {exact * 100:.1f}% of the field exact (needs 97%)')
            if hud_ok < 0.99:
                failures.append(f'scroll ({dx},{dy}): the HUD moved ({hud_ok * 100:.1f}% unchanged, needs 99%)')
    if failures:
        print('VERDICT FAIL: ' + '; '.join(failures))
        return 1
    print('VERDICT PASS: the pixel-art motion pass moves a scroll by half a step, exactly, and leaves a HUD alone')
    return 0


if __name__ == '__main__':
    sys.exit(main())
