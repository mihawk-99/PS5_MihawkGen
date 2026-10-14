# PS5_FrameGen - scores the frames the RetroArch integration dumped (PS5_FRAMEGEN_DUMP) against
# the test core's exact scene: the frame made at j/N of the way from content frame k-1 to k must
# be the scene at k - 1 + j/N. Prints the network's PSNR beside blend and repeat, and a verdict.
#
#   python3 integrations/retroarch/check_dump.py build/fg_test_core_libretro.so build/ra-run/dump [--min-gain 3]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import ctypes
import glob
import math
import os
import sys

import numpy as np

W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('core')
    ap.add_argument('folder')
    ap.add_argument('--min-gain', type=float, default=3.0, help='the network must beat blend by this many dB')
    ap.add_argument('--cut-after', type=int, default=-1, help='the test core cut to another scene at this core frame: score the frames made across it against the nearer real frame')
    ap.add_argument('--period', type=int, default=1, help='core frames between new content frames (a game at half rate: 2)')
    args = ap.parse_args()
    core = ctypes.CDLL(os.path.abspath(args.core))
    core.fg_test_render.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_uint32)]

    def truth(t):
        buf = (ctypes.c_uint32 * (W * H))()
        core.fg_test_render(t, buf)
        a = np.frombuffer(buf, dtype=np.uint32).reshape(H, W)
        return np.stack([(a >> 16) & 255, (a >> 8) & 255, a & 255], -1).astype(np.float32) / 255

    def load(name):
        raw = np.fromfile(os.path.join(args.folder, name), dtype=np.uint8)
        return raw.reshape(H, W, 4)[..., :3].astype(np.float32) / 255

    def psnr(a, b):
        return -10 * math.log10(max(float(((a - b) ** 2).mean()), 1e-10))

    cur = {int(os.path.basename(f)[4:8]): f for f in glob.glob(os.path.join(args.folder, 'cur_*.rgba'))}
    gens = sorted(glob.glob(os.path.join(args.folder, 'gen_*_*.rgba')))
    if not cur or not gens:
        print('VERDICT FAIL: no frames were dumped')
        return 1
    # The first frames held are not content (what the driver had before it): find the first that matches
    # a core frame exactly, and the frame after it matches the next new one
    def best(img):
        n = min(range(0, 3000), key=lambda n: float(((truth(n) - img) ** 2).mean()))
        return n, psnr(truth(n), img)
    first_k = None
    for k in sorted(cur):
        n, p = best(load(f'cur_{k:04d}.rgba'))
        if p > 60:
            first_k, first_n = k, n
            break
    if first_k is None:
        print('VERDICT FAIL: no dumped frame is a frame of the scene')
        return 1
    start = first_n - first_k * args.period
    print(f'content frame {first_k} is core frame {first_n}')
    rows = []
    for f in gens:
        name = os.path.basename(f)[:-5]
        k = int(name.split('_')[1])
        t_frac = float(open(f[:-5] + '.t').read())
        if k <= first_k or k not in cur or k - 1 not in cur:
            continue
        t = truth(start + (k - 1 + t_frac) * args.period)
        a, b = load(f'cur_{k - 1:04d}.rgba'), load(f'cur_{k:04d}.rgba')
        rows.append((psnr(load(os.path.basename(f)), t), psnr(a * (1 - t_frac) + b * t_frac, t), psnr(a if t_frac < 0.5 else b, t)))
    if args.cut_after >= 0:
        # The frames made between the content frame before the cut and the one after it: the answer is the nearer
        rows = []
        for f in gens:
            k = int(os.path.basename(f)[:-5].split('_')[1])
            t_frac = float(open(f[:-5] + '.t').read())
            if k <= first_k or k not in cur or k - 1 not in cur:
                continue
            core_before = first_n + (k - 1 - first_k) * args.period
            core_after = first_n + (k - first_k) * args.period
            if not (core_before < args.cut_after <= core_after):
                continue
            a, b = load(f'cur_{k - 1:04d}.rgba'), load(f'cur_{k:04d}.rgba')
            nearest = a if t_frac < 0.5 else b
            rows.append((psnr(load(os.path.basename(f)), nearest), psnr(a * (1 - t_frac) + b * t_frac, nearest)))
        if not rows:
            print('VERDICT FAIL: no frame was made across the cut')
            return 1
        m = np.array(rows).mean(0)
        print(f'{len(rows)} frames made across the cut, against the nearer real frame: network {m[0]:.2f} dB, blend {m[1]:.2f} dB')
        ok = m[0] > m[1] + 6.0
        print(f'VERDICT {"PASS" if ok else "FAIL"}: across the cut the network is {m[0] - m[1]:+.2f} dB over blend (needs +6.0)')
        return 0 if ok else 1
    m = np.array(rows).mean(0)
    print(f'{len(rows)} generated frames: network {m[0]:.2f} dB, blend {m[1]:.2f} dB, repeat {m[2]:.2f} dB')
    ok = m[0] > m[1] + args.min_gain
    print(f'VERDICT {"PASS" if ok else "FAIL"}: the network is {m[0] - m[1]:+.2f} dB over blend (needs {args.min_gain:+.1f})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
