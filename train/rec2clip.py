# PS5_FrameGen - turns a present-stream recording (tools/analyze-stream.py's input: what the display was given on the console,
# real pictures and generated ones) into test clips: the real pictures alone, cropped to the game's picture (the black bars of a
# 4:3 game on a 16:9 screen are cut), at the working height (288 rows), 64 frames to a clip, in the clip format train/real.py
# reads (a 12-byte header: width, height, count; then RGB bytes). A game's clips go to ~/retro-tests/bench3d-clips, the
# benchmark's own folder (never trained on).
#
#   train/rec2clip.py REC_DIR GAME_TAG [OUT_DIR]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('analyze_stream', os.path.join(here, '..', 'tools', 'analyze-stream.py'))
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

rec, tag = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser('~/retro-tests/bench3d-clips')
frames, rows = analyze.load(rec)
real = [i for i, r in enumerate(rows) if not r['gen']]
# one real picture for each new frame: where the count of new frames held changes
keep = []
last = None
for i in real:
    u = rows[i]['unique']
    if u != last:
        keep.append(i)
        last = u
clip = frames[keep]
if len(clip) < 12:
    print('too few real pictures', len(clip))
    sys.exit(1)
bright = clip.max(axis=(0, 1)).max(axis=-1) > 12
cols = np.where(bright)[0]
rws = np.where(clip.max(axis=(0, 2)).max(axis=-1) > 12)[0]
x0, x1 = (cols.min(), cols.max() + 1) if len(cols) else (0, clip.shape[2])
y0, y1 = (rws.min(), rws.max() + 1) if len(rws) else (0, clip.shape[1])
clip = clip[:, y0:y1, x0:x1]
h, w = clip.shape[1:3]
th = 288
tw = max(16, round(w * th / h / 2) * 2)
x = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float()
x = F.interpolate(x, size=(th, tw), mode='area').round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
os.makedirs(out, exist_ok=True)
n = 0
for k in range(0, len(x) - 11, 64):
    part = x[k:k + 64]
    if len(part) < 12:
        break
    path = os.path.join(out, '%s_w%02d.clip' % (tag, n))
    with open(path, 'wb') as fh:
        fh.write(struct.pack('3I', tw, th, len(part)))
        fh.write(part.tobytes())
    print(path, tw, th, len(part))
    n += 1
