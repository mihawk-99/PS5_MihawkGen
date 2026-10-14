#!/usr/bin/env python3
"""PS5_FrameGen - the baseline for tools/analyze-stream.py: the same scripted scene (integrations/retroarch/fg_test_core.c, a
checkerboard floor scrolling at an exactly known speed under moving shapes) shown at 120 pictures a second from cores that
run at 60, 30 and 24 fps, by each method (neural, blend, repeat), recorded by the present-stream recorder and measured. A row
"ideal" is the same measures on pictures drawn exactly at their instants: what a perfect generator would score, and the noise
floor of the measure itself.

    tools/stream-baseline.py [--fps 60,30,24] [--methods neural,blend,repeat] [--network FILE.fgw] [--keep]

Needs build/ra-host2 (integrations/retroarch/build-host2.sh) and a Vulkan device. Prints a table in Markdown.

Copyright (C) 2026 Mihawk
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REGION = '0.1,0.82,0.9,1'   # the floor alone: the shapes and the HUD are above it
W, H = 640, 360
PICTURES = 240
METHODS = {'neural': 0, 'blend': 1, 'repeat': 2}


def build_core():
    so = os.path.join(ROOT, 'build', 'fg_test_core_libretro.so')
    src = os.path.join(ROOT, 'integrations', 'retroarch', 'fg_test_core.c')
    inc = os.path.join(ROOT, 'build', 'ra-host2', 'src', 'libretro-common', 'include')
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(['cc', '-O2', '-shared', '-fPIC', '-I', inc, '-o', so, src, '-lm'], check=True)
    return so


def write_ideal(core, folder, fps):
    """Pictures drawn exactly at their instants, with the index the recorder would write."""
    lib = ctypes.CDLL(core)
    lib.fg_test_render.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_uint32)]
    per = int(round(120.0 / fps))
    first = 100
    frames, rows = [], []
    for p in range(PICTURES):
        when = first + p / per
        buf = (ctypes.c_uint32 * (W * H))()
        lib.fg_test_render(when * 60.0 / fps, buf)
        x = np.frombuffer(buf, dtype=np.uint32).reshape(H, W)
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[:, :, 0] = (x >> 16) & 255
        rgba[:, :, 1] = (x >> 8) & 255
        rgba[:, :, 2] = x & 255
        rgba[:, :, 3] = 255
        frames.append(rgba)
        real = p % per == 0
        unique = int(round(when)) if real else int(np.ceil(when))
        rows.append(f'{p} {p} {p * 8333333} {0 if real else 1} {1.0 if real else when - (unique - 1):.5f} {unique} 0')
    os.makedirs(folder, exist_ok=True)
    np.stack(frames).tofile(os.path.join(folder, 'frames.rgba'))
    open(os.path.join(folder, 'size.txt'), 'w').write(f'{W} {H} {PICTURES}\n')
    open(os.path.join(folder, 'index.txt'), 'w').write('# i place ns generated t unique level\n' + '\n'.join(rows) + '\n')


def run_host(core, fps, method, folder, network):
    work = os.path.join(ROOT, 'build', 'stream-baseline', 'run')
    dummy = os.path.join(ROOT, 'build', 'stream-baseline', 'content.bin')
    os.makedirs(os.path.dirname(dummy), exist_ok=True)
    open(dummy, 'w').write('x\n')
    shutil.rmtree(folder, ignore_errors=True)
    env = dict(os.environ, FG_TEST_FPS=str(fps), PS5_FRAMEGEN_RECORD=folder, PS5_FRAMEGEN_RECORD_AT='400',
               PS5_FRAMEGEN_RECORD_COUNT=str(PICTURES), AUTO='true', TARGET='120', SIZE='0', METHOD=str(METHODS[method]))
    if network:
        env['NETWORK'] = os.path.abspath(network)
    subprocess.run([os.path.join(ROOT, 'integrations/retroarch/run-host-game.sh'), core, dummy, work, '', '',
                    '700', '2', '0', '1'], env=env, capture_output=True, timeout=600)
    rec = os.path.join(folder, 'rec-1')
    if not os.path.exists(os.path.join(rec, 'frames.rgba')):
        raise RuntimeError(f'no recording from {method} at {fps} fps')
    return rec


def analyze(rec):
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'analyze-stream.py'), rec, '--region', REGION],
                         capture_output=True, text=True).stdout
    def grab(pattern, default=float('nan')):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else default
    return dict(
        track=grab(r'position of generated[^\n]*median ([\d.]+)'), track90=grab(r'position of generated[^\n]*90th percentile ([\d.]+)'),
        repeats=grab(r'repeats[^\n]*: (\d+) of'), flicker=grab(r'flicker[^\n]*mean ([\d.]+)%'),
        speed=grab(r'the game moves [^\n]*speed ([\d.]+)'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fps', default='60,30,24')
    ap.add_argument('--methods', default='neural,blend,repeat')
    ap.add_argument('--network')
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    core = build_core()
    base = os.path.join(ROOT, 'build', 'stream-baseline')
    rows = []
    for fps in [int(x) for x in args.fps.split(',')]:
        ideal = os.path.join(base, f'ideal-{fps}')
        shutil.rmtree(ideal, ignore_errors=True)
        write_ideal(core, ideal, fps)
        rows.append((fps, 'ideal', analyze(ideal)))
        for method in args.methods.split(','):
            rec = run_host(core, fps, method, os.path.join(base, f'{method}-{fps}'), args.network)
            rows.append((fps, method, analyze(rec)))
            print(f'  {method} at {fps} fps done', file=sys.stderr, flush=True)
    print('| core fps | method | position error, median (frame intervals) | 90th percentile | flicker (% of picture) | repeated pictures |')
    print('| --- | --- | --- | --- | --- | --- |')
    for fps, method, r in rows:
        print(f'| {fps} | {method} | {r["track"]:.2f} | {r["track90"]:.2f} | {r["flicker"]:.2f} | {r["repeats"]:.0f} |')
    if not args.keep:
        for name in os.listdir(base):
            if name.startswith(('neural-', 'blend-', 'repeat-', 'run')):
                shutil.rmtree(os.path.join(base, name), ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
