# PS5_FrameGen - turns a folder of frame dumps (cur_NNNN.rgba and size.txt, written by the RetroArch shim with
# PS5_FRAMEGEN_DUMP_CURONLY) into a training clip: RGB bytes at the working height (288 rows), a 12-byte header
# (width, height, count). The dump is deleted. Clips go to ~/retro-tests/train-clips, outside every repository (docs/plan.md,
# "The data rule").
#
#   train/dump2clip.py DUMPDIR OUTFILE [MIN_MOTION]
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import os
import shutil
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

src, out = sys.argv[1], sys.argv[2]
min_motion = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
w, h = map(int, open(os.path.join(src, 'size.txt')).read().split())
files = sorted(glob.glob(os.path.join(src, 'cur_*.rgba')))
files = [f for f in files if os.path.getsize(f) == w * h * 4][:64]
if len(files) < 12:
    print('too few frames', len(files))
    sys.exit(1)
th = 288
tw = max(16, round(w * th / h / 2) * 2)
frames = []
for f in files:
    a = torch.from_numpy(np.fromfile(f, np.uint8).reshape(h, w, 4)[:, :, :3].copy()).permute(2, 0, 1).float()[None]
    frames.append(F.interpolate(a, size=(th, tw), mode='area')[0].round().clamp(0, 255).byte().permute(1, 2, 0).numpy())
clip = np.stack(frames)
motion = float(np.mean([np.abs(clip[i + 1].astype(int) - clip[i]).mean() / 255 for i in range(len(clip) - 1)]))
if motion < min_motion:
    print('too little motion', round(motion, 4))
    sys.exit(2)
with open(out, 'wb') as fh:
    fh.write(struct.pack('3I', tw, th, len(clip)))
    fh.write(clip.tobytes())
shutil.rmtree(src, ignore_errors=True)
print('clip', os.path.basename(out), len(clip), 'frames', tw, 'x', th, 'motion', round(motion, 4))
