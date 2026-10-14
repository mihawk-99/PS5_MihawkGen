# PS5_FrameGen - real clips for training (docs/plan.md, "The data rule"). A clip file is what
# integrations/retroarch/fg_proxy_core.c captures on this PC: a 12-byte header (width, height, count) and the frames as
# RGB bytes. They are read from a directory outside the repository (default ~/retro-tests/train-clips) and used only to
# adjust the network's weights; no frame is stored in the model or the repository.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import os
import random
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

WORKING_HEIGHT = 288
# The benchmark: clips listed in <clips folder>/bench.list (frozen on 2026-10-11 from what the earlier rule held out: whole games
# that no training step had seen) and every clip of a game that has a listed clip. Nothing else is held out, so clips captured
# later train the network and cannot change the benchmark.
_BENCH = {}


def bench_names(root):
    if root not in _BENCH:
        path = os.path.join(root, 'bench.list')
        _BENCH[root] = set(l.strip() for l in open(path) if l.strip()) if os.path.exists(path) else set()
    return _BENCH[root]


def is_held(path, root=None):
    root = root or os.path.dirname(path)
    names = bench_names(root)
    name = os.path.basename(path)[:-5]
    games = set(n.rsplit('_', 1)[0] for n in names)
    return name in names, name.rsplit('_', 1)[0] in games


class Clips:
    def __init__(self, root=None, held=False, heavy=4.0, bench3d=True):
        root = root or os.path.expanduser('~/retro-tests/train-clips')
        tags = set()
        if os.path.exists(os.path.join(root, '3d.list')):
            tags = set(l.strip() for l in open(os.path.join(root, '3d.list')) if l.strip())
        self.clips = []
        self.weights = []   # a clip's share of the batches: clips of 3D games (named in 3d.list, one tag a line) count several times
        self.new = []      # for each clip, the indices of its new frames
        self.names = []
        paths = sorted(glob.glob(os.path.join(root, '*.clip')))
        # the benchmark's extension: games captured later and set aside whole (their folder is never trained on)
        extra = os.path.expanduser('~/retro-tests/bench3d-clips')
        if held and os.path.isdir(extra) and bench3d:
            paths += sorted(glob.glob(os.path.join(extra, '*.clip')))
        for path in paths:
            listed, game_held = is_held(path, root)
            if os.path.dirname(path) == extra or os.path.basename(root) == 'bench3d-clips':
                listed = game_held = True
            if (listed if held else not game_held) is False:
                continue
            b = open(path, 'rb').read()
            w, h, n = struct.unpack('3I', b[:12])
            if n < 8 or len(b) < 12 + n * w * h * 3:
                continue
            clip = torch.from_numpy(np.frombuffer(b, np.uint8, n * w * h * 3, 12).reshape(n, h, w, 3).copy())
            # the new frames: a game that changes its picture every other frame has half as many as frames
            new = [0] + [i for i in range(1, n) if bool((clip[i] != clip[i - 1]).any())]
            if len(new) < 5:
                continue
            self.new.append(new)
            self.clips.append(clip)
            self.weights.append(heavy if os.path.basename(path)[:-5].rsplit('_', 1)[0] in tags else 1.0)
            self.names.append(os.path.basename(path)[:-5])

    def __len__(self):
        return len(self.clips)

    @staticmethod
    def prepare(frames, device):
        """Native frames (n,h,w,3) uint8 -> (n,3,288,w') floats as the generator's down pass sees a pixel-art game:
        nearest up by k, then an area average to the working height."""
        n, h, w, _ = frames.shape
        k = min(4, max(2, 1080 // h))
        x = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
        x = x.repeat_interleave(k, 2).repeat_interleave(k, 3)
        return F.interpolate(x, size=(WORKING_HEIGHT, max(1, round(w * k * WORKING_HEIGHT / (h * k)))), mode='area')

    teacher_dir = None   # a folder of teachers' pictures (train/teachers.py make): the middle picture of each pair of new frames two apart

    def teacher_frame(self, ci, j):
        """The teachers' picture for the pair new[j], new[j + 2] of clip ci (uint8 (h, w, 3)), or None."""
        if not self.teacher_dir:
            return None
        arrays = self.__dict__.setdefault('_teacher', {})
        if ci not in arrays:
            path = os.path.join(self.teacher_dir, self.names[ci] + '.npy')
            arrays[ci] = np.load(path, mmap_mode='r') if os.path.exists(path) else None
        arr = arrays[ci]
        return None if arr is None or j >= len(arr) else arr[j]

    def sample(self, n, crop, device, rng, min_motion=0.004, keep_still=0.25, gaps=(2, 4), with_teacher=False):
        """n triplets (frame0, frame1, target, t): clips at random, a gap of 2 or 4 content frames, a target between,
        a crop, flips, a swap and a colour permutation. Triplets whose frames hardly differ are mostly skipped."""
        out = [[], [], [], [], [], [], []]
        tries = 0
        while len(out[0]) < n:
            tries += 1
            if tries > 40 * n:
                break
            ci = rng.choices(range(len(self.clips)), weights=self.weights)[0]
            clip, new = self.clips[ci], self.new[ci]
            gap = rng.choice(gaps)
            if len(new) <= gap + 1:
                continue
            # triplets of new frames: what the generator is given (a game at 30 fps shows every picture twice)
            j = rng.randrange(0, len(new) - gap)
            q = rng.randint(1, gap - 1)
            i, mid, last = new[j], new[j + q], new[j + gap]
            t = q / gap
            # the frame before the pair, as far back as the pair is long (what a generator that remembers one more frame has);
            # when the pair is read the other way round it is the frame after, as far forward. None: the clip does not reach.
            swap_now = rng.random() < 0.5
            pj = (j + 2 * gap) if swap_now else (j - gap)
            pidx = new[pj] if 0 <= pj < len(new) else None
            x = self.prepare(clip[[i, mid, last] + ([pidx] if pidx is not None else [])], device)
            a, m, c = x[0], x[1], x[2]
            pf = x[3] if pidx is not None else (c if swap_now else a)
            tf = self.teacher_frame(ci, j) if (with_teacher and gap == 2) else None
            if tf is not None:
                tf = self.prepare(torch.from_numpy(np.ascontiguousarray(tf))[None], device)[0]
            else:
                tf = m
            H, W = a.shape[-2:]
            if W < crop:
                continue
            y0, x0 = rng.randint(0, H - crop), rng.randint(0, W - crop)
            sl = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
            a, m, c, pf, tf = a[sl], m[sl], c[sl], pf[sl], tf[sl]
            has_teacher = float(self.teacher_frame(ci, j) is not None) if (with_teacher and gap == 2) else 0.0
            if float((a - c).abs().mean()) < min_motion and rng.random() > keep_still:
                continue
            if swap_now:
                a, c, t = c, a, 1 - t
            if rng.random() < 0.5:
                a, m, c, pf, tf = a.flip(-1), m.flip(-1), c.flip(-1), pf.flip(-1), tf.flip(-1)
            if rng.random() < 0.5:
                a, m, c, pf, tf = a.flip(-2), m.flip(-2), c.flip(-2), pf.flip(-2), tf.flip(-2)
            perm = torch.randperm(3, device=device)
            for k, v in enumerate((a[perm], c[perm], m[perm], torch.tensor(t, device=device), pf[perm], tf[perm], torch.tensor(has_teacher, device=device))):
                out[k].append(v)
        if not out[0]:
            return None
        res = tuple(torch.stack(v) for v in out)
        return res if with_teacher else res[:5]
