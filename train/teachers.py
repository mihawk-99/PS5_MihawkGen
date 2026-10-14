# PS5_FrameGen - public interpolation models as teachers: their middle pictures of our training clips' pairs, stored for the trainer to
# distil from (train.py --teacher DIR). Only models whose code and weights are under a permissive licence are used (SOURCES.md in
# ~/retro-tests/bench-models: RIFE MIT, IFRNet MIT, EMA-VFI Apache-2.0, FILM Apache-2.0, VFIMamba Apache-2.0); AMT (CC BY-NC) is a
# measurement reference only. Nothing of these models is copied into the repository: only their outputs on our own clips are
# stored, outside it.
#
#   ~/retro-tests/bench-models/venv/bin/python train/teachers.py eval  [--models a,b,...] [--per-clip N]
#   ~/retro-tests/bench-models/venv/bin/python train/teachers.py make  NAME model,model,...   (NAME: the folder; the mean of the models)
#
# eval: each model and the mean of each pair of models on the clean games of the benchmark (set a), behind the generator's last pass.
# make: for each training clip (the games that are not held out) and each pair of new frames two apart, the mean of the models'
# pictures at t = 1/2: ~/retro-tests/teacher/NAME/<clip>.npy, uint8 (pairs, H, W, 3), pair j = frames new[j], new[j + 2].
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import itertools
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import bench_public as BP  # noqa: E402  (it takes the train/ folder off sys.path: its models have modules of their own called "model")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RT = os.path.expanduser('~/retro-tests')
DEV = 'cuda'


def load(names):
    nets = {}
    for n in names:
        net = BP.REGISTRY[n][0]()
        net.to(DEV)
        nets[n] = net
    return nets


def run(net, a, c):
    """The model's middle picture of two uint8 (H, W, 3) frames, float (1, 3, H, W) in 0..1."""
    h, w = a.shape[:2]
    ph, pw = (-h) % net.pad, (-w) % net.pad

    def prep(img):
        x = torch.from_numpy(img).to(DEV).permute(2, 0, 1).float()[None] / 255.0
        return F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=net.fp16_ok):
        y = net.fn(prep(a), prep(c))
    return y.float()[:, :, :h, :w].clamp(0, 1)


def psnr(y, truth):
    mse = ((y.mul(255).round() - truth.mul(255).round()) ** 2).mean().item()
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def do_eval(models, per_clip):
    excluded = set(l.strip() for l in open(RT + '/bench-excluded.txt'))
    nets = load(models)
    sys.path.insert(0, HERE)
    from model import gate
    scores = {}
    names = []
    for f in sorted(glob.glob(RT + '/bench-triplets/*.npz')):
        clip = os.path.basename(f)[:-4]
        g = clip.rsplit('_', 1)[0]
        if g in excluded:
            continue
        z = np.load(f)
        for i in range(min(per_clip, len(z['a']))):
            a, m, c = z['a'][i], z['m'][i], z['c'][i]
            ys = {n: run(net, a, c) for n, net in nets.items()}
            x0 = torch.from_numpy(a).to(DEV).permute(2, 0, 1).float()[None] / 255
            x1 = torch.from_numpy(c).to(DEV).permute(2, 0, 1).float()[None] / 255
            tm = torch.from_numpy(m).to(DEV).permute(2, 0, 1).float()[None] / 255
            t = torch.tensor([0.5], device=DEV)
            cands = dict(ys)
            for p, q in itertools.combinations(sorted(ys), 2):
                cands[p + '+' + q] = (ys[p] + ys[q]) / 2
            if len(ys) >= 3:
                cands['all'] = sum(ys.values()) / len(ys)
            for k, y in cands.items():
                scores.setdefault(k, {}).setdefault(g, []).append(psnr(gate(x0, x1, y, t), tm))
        names.append(clip)
    print(len(names), 'clips of', len({n.rsplit('_', 1)[0] for n in names}), 'clean games')
    for k, per in sorted(scores.items(), key=lambda kv: -np.mean([np.mean(v) for v in kv[1].values()])):
        print(f'{k:36s} {np.mean([np.mean(v) for v in per.values()]):7.3f}')


def do_make(name, models):
    nets = load(models)
    sys.path.insert(0, HERE)
    from real import is_held
    out = os.path.join(RT, 'teacher', name)
    os.makedirs(out, exist_ok=True)
    clips = sorted(glob.glob(RT + '/train-clips/*.clip'))
    done = 0
    for path in clips:
        listed, game_held = is_held(path, RT + '/train-clips')
        clip = os.path.basename(path)[:-5]
        if game_held or os.path.exists(os.path.join(out, clip + '.npy')):
            continue
        b = open(path, 'rb').read()
        w, h, n = struct.unpack('3I', b[:12])
        if n < 8 or len(b) < 12 + n * w * h * 3:
            continue
        frames = np.frombuffer(b, np.uint8, n * w * h * 3, 12).reshape(n, h, w, 3)
        new = [0] + [i for i in range(1, n) if bool((frames[i] != frames[i - 1]).any())]
        if len(new) < 5:
            continue
        ys = []
        for j in range(len(new) - 2):
            a, c = frames[new[j]], frames[new[j + 2]]
            y = sum(run(net, a, c) for net in nets.values()) / len(nets)
            ys.append(y[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
        np.save(os.path.join(out, clip + '.npy'), np.stack(ys))
        done += 1
        if done % 20 == 0:
            print(done, 'clips', clip, flush=True)
    print('made', done, 'clips in', out)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'eval':
        models = (sys.argv[sys.argv.index('--models') + 1] if '--models' in sys.argv else 'ifrnet-l,ifrnet,emavfi,rife-4.26,film').split(',')
        do_eval(models, int(sys.argv[sys.argv.index('--per-clip') + 1]) if '--per-clip' in sys.argv else 2)
    elif cmd == 'make':
        do_make(sys.argv[2], sys.argv[3].split(','))
    else:
        print(__doc__)
