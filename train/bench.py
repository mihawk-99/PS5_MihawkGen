# PS5_FrameGen - the benchmark: PSNR of the middle frame on real clips that no training step has seen
# (train/real.py, is_held: whole games held out, and the rest of a few games' scenes), full frames, 8-bit
# rounded, every unique-frame triplet (frame, the next new frame, the one after: the target is the middle one, t = 1/2).
# Per clip, then per game, then over games (macro) and over triplets (micro). Scores of "blend" and "repeat" come with it.
#
#   .venv/bin/python train/bench.py CKPT.pt [CKPT2.pt ...] [--json OUT.json] [--per-game] [--triplets 8]
#   .venv/bin/python train/bench.py --export DIR       the triplets as DIR/<clip>.npz (a, m, c: uint8 (T,H,W,3)) for
#                                                       models that are not ours (bench_public.py reads them)
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DEFAULT_BLOCKS, build  # noqa: E402
from real import Clips  # noqa: E402


PREPARE = os.environ.get('FG_PREPARE') == '1'   # the frames as the generator sees low-resolution games: enlarged by a whole number, then area-averaged to 288 rows


def prep(frame):
    if not PREPARE or frame.shape[0] >= 288:
        return frame
    x = Clips.prepare(torch.from_numpy(frame)[None], 'cpu')[0]
    return x.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()


def triplets(clips, per_clip):
    """For each held clip, up to per_clip triplets of new frames spread over it: (name, [(a, m, c) uint8 (H,W,3)])."""
    out = []
    for name, clip, new in zip(clips.names, clips.clips, clips.new):
        n = len(new) - 2
        if n < 1:
            continue
        picks = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(per_clip, n))))
        # (a, the target, c, and the frame two new frames before a, as far back as the pair is long; a itself when the clip does not reach)
        out.append((name, [(prep(clip[new[j]].numpy()), prep(clip[new[j + 1]].numpy()), prep(clip[new[j + 2]].numpy()),
                            prep(clip[new[j - 2]].numpy()) if j >= 2 else prep(clip[new[j]].numpy())) for j in picks]))
    return out


def psnr8(pred, truth):
    """PSNR in dB between two uint8 images (as a display shows them)."""
    mse = np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


DEV = os.environ.get('FG_DEVICE', 'cuda')


def load_model(path):
    st = torch.load(path, map_location=DEV)
    m = build(st).to(DEV).eval()
    m.load_state_dict(st['ema'])
    return m


def gate(x0, x1, moved, t=0.5):
    """What the generator's last pass does (framegen/shaders/nn_final.comp): model.gate."""
    from model import gate as model_gate
    return model_gate(x0, x1, moved, torch.full((x0.shape[0],), float(t), device=x0.device))


def run_model(m, a, c, p=None, device=None, gated=True, t=0.5):
    device = device or DEV
    x0 = torch.from_numpy(a).to(device).permute(2, 0, 1).float()[None] / 255.0
    x1 = torch.from_numpy(c).to(device).permute(2, 0, 1).float()[None] / 255.0
    xp = torch.from_numpy(p).to(device).permute(2, 0, 1).float()[None] / 255.0 if (p is not None and getattr(m, 'prev', False)) else None
    h, w = x0.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16        # the blocks work at 1/16 of the size
    if ph or pw:
        x0 = torch.nn.functional.pad(x0, (0, pw, 0, ph), mode='replicate')
        x1 = torch.nn.functional.pad(x1, (0, pw, 0, ph), mode='replicate')
        if xp is not None:
            xp = torch.nn.functional.pad(xp, (0, pw, 0, ph), mode='replicate')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device == 'cuda'):
        y = m(x0, x1, torch.tensor([t], device=device), prev=xp)
    y = y.float()
    if gated and not getattr(m, 'gated', False):
        y = gate(x0, x1, y, t)
    y = y[0, :, :h, :w].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--json')
    ap.add_argument('--per-game', action='store_true')
    ap.add_argument('--triplets', type=int, default=8)
    ap.add_argument('--export')
    ap.add_argument('--set', default='a', choices=['a', 'b'], help='a: the frozen benchmark (45 games); b: 3D clips set aside later (~/retro-tests/bench3d-clips)')
    args = ap.parse_args()
    held = Clips(held=True, bench3d=False) if args.set == 'a' else Clips(root=os.path.expanduser('~/retro-tests/bench3d-clips'), held=True, bench3d=False)
    sets = triplets(held, args.triplets)
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        for name, trips in sets:
            np.savez(os.path.join(args.export, name + '.npz'), a=np.stack([t[0] for t in trips]),
                     m=np.stack([t[1] for t in trips]), c=np.stack([t[2] for t in trips]), p=np.stack([t[3] for t in trips]))
        print(len(sets), 'clips,', sum(len(t) for _, t in sets), 'triplets exported to', args.export)
        return 0
    models = {'blend': None, 'repeat': None}
    for p in args.ckpts:
        models[os.path.basename(os.path.dirname(os.path.abspath(p))) or p] = load_model(p)
    scores = {k: {} for k in models}     # model -> clip -> [psnr]
    for name, trips in sets:
        for a, m, c, pr in trips:
            for k, mod in models.items():
                if k == 'blend':
                    y = ((a.astype(np.float32) + c.astype(np.float32)) / 2).round().astype(np.uint8)
                elif k == 'repeat':
                    y = a
                else:
                    y = run_model(mod, a, c, pr)
                scores[k].setdefault(name, []).append(psnr8(y, m))
    games = {}
    for k in models:
        for name, v in scores[k].items():
            games.setdefault(name.rsplit('_', 1)[0], {}).setdefault(k, []).extend(v)
    ks = list(models)
    if args.per_game:
        print('game'.ljust(22), ' '.join(k.rjust(8) for k in ks))
        for g in sorted(games):
            print(g.ljust(22), ' '.join(f'{np.mean(games[g][k]):8.2f}' for k in ks))
    n_clips = len(sets)
    n_trip = sum(len(t) for _, t in sets)
    print(f'{n_clips} unseen clips of {len(games)} games, {n_trip} triplets (middle frame, full frames)')
    result = {}
    for k in ks:
        macro = float(np.mean([np.mean(games[g][k]) for g in games]))
        micro = float(np.mean([x for v in scores[k].values() for x in v]))
        med = float(np.median([np.mean(v) for v in scores[k].values()]))
        result[k] = dict(macro=macro, micro=micro, median_clip=med)
        print(f'{k.ljust(10)} games-average {macro:6.2f} dB   triplets-average {micro:6.2f}   median clip {med:6.2f}')
    if args.json:
        json.dump(dict(result=result, games={g: {k: float(np.mean(v)) for k, v in games[g].items()} for g in games}),
                  open(args.json, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
