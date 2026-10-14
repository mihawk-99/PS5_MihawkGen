# PS5_FrameGen - the benchmark's integrity check: no frame of the benchmark's triplets (frames, truths) is in a clip a network trained on, and
# no game of the benchmark has a training clip. Hashes every frame of every training clip (blake2b of the bytes) and of every benchmark picture.
#
#   .venv/bin/python train/bench_audit.py
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import hashlib
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RT = os.path.expanduser('~/retro-tests')


def digest(x):
    return hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=12).digest()


def clip_frames(path):
    b = open(path, 'rb').read()
    w, h, n = struct.unpack('3I', b[:12])
    arr = np.frombuffer(b, np.uint8, n * w * h * 3, 12).reshape(n, h, w, 3)
    return arr


def main():
    from real import is_held
    train_games, bench_games = set(), set()
    train_hashes = {}
    for path in sorted(glob.glob(RT + '/train-clips/*.clip')):
        listed, game_held = is_held(path, RT + '/train-clips')
        name = os.path.basename(path)[:-5]
        game = name.rsplit('_', 1)[0]
        if game_held:
            bench_games.add(game)
            continue
        train_games.add(game)
        for i, f in enumerate(clip_frames(path)):
            train_hashes.setdefault(digest(f), (name, i))
    print(len(train_games), 'games trained on,', len(bench_games), 'games held out in train-clips;', len(train_hashes), 'distinct training frames')
    bad = 0
    total = 0
    sets = {'a': RT + '/bench-triplets', 'b': RT + '/bench-triplets-b'}
    for key, d in sets.items():
        games = set()
        hit = 0
        n = 0
        for f in sorted(glob.glob(d + '/*.npz')):
            games.add(os.path.basename(f)[:-4].rsplit('_', 1)[0])
            z = np.load(f)
            for k in ('a', 'm', 'c'):
                for fr in z[k]:
                    n += 1
                    if digest(fr) in train_hashes:
                        hit += 1
        overlap = games & train_games
        print(f'set {key}: {len(games)} games, {n} pictures; games also trained on: {sorted(overlap)}; pictures found in training clips: {hit}')
        bad += hit + len(overlap)
        total += n
    # a game of the benchmark is dropped from the clean benchmark when any of its pictures that is not flat is in a training clip (the same game
    # captured twice under two tags), or when it shares a title prefix (the first word of its tag) with a game trained on: a title that came in
    # several versions is not a game never seen
    base = lambda g: g.split('-')[0]
    train_bases = {base(g) for g in train_games}
    excluded = set()
    for key, d in sets.items():
        for f in sorted(glob.glob(d + '/*.npz')):
            g = os.path.basename(f)[:-4].rsplit('_', 1)[0]
            if base(g) in train_bases:
                excluded.add(g)
            z = np.load(f)
            for k in ('a', 'm', 'c'):
                for fr in z[k]:
                    if fr.std() >= 2.0 and digest(fr) in train_hashes:
                        excluded.add(g)
    open(RT + '/bench-excluded.txt', 'w').write('\n'.join(sorted(excluded)) + '\n')
    print(len(excluded), 'games leave the clean benchmark:', ', '.join(sorted(excluded)))
    print('VERDICT', 'FAIL (the benchmark as frozen is not clean)' if bad else 'PASS', f'({total} benchmark pictures checked against {len(train_hashes)} training frames); the clean benchmark is the rest')
    return 0


if __name__ == '__main__':
    sys.exit(main())
