#!/usr/bin/env python3
"""PS5_FrameGen - reads a present-stream recording (the pictures the display was given, one after another, real and
generated: integrations/retroarch/gfx/vulkan_framegen.cpp, "the present-stream recorder") and says what a person
would see when the stream runs.

    tools/analyze-stream.py REC_DIR [--out DIR] [--sheet-start N] [--sheet-count N]

REC_DIR holds frames.rgba (raw RGBA8, one picture after the other), size.txt ("width height count") and index.txt (a line
a picture: its place, the time it was recorded in ns, 1 if generated, its instant t, the new frames held so far, the
governor's level). On the console the folder is /app0/rec-N (tools/run-title.sh brings back what the script's RECORD
wrote); on a PC run it is the folder named in PS5_FRAMEGEN_RECORD.

It writes, in --out (default REC_DIR/analysis):
  report.txt   timing (the pace pictures were recorded at), the pattern of real and generated pictures, repeats, the motion
               of the picture from one picture to the next and how even it is, and the flicker score of each generated picture
  flicker.png  where generated pictures fall outside both of their neighbours' values: the heat of it, over the whole stream
  motion.png   the picture's displacement from one picture to the next, over time (a smooth scroll is a straight line; a
               game that moves at half the rate is a staircase)
  xt.png       one row of the picture, every picture one under the other (motion is a slanted streak; judder is a zigzag);
               a bar at the left is green for a generated picture and grey for a real one
  sheet.png    a contact sheet of consecutive pictures, each labelled with its place, R or G and its instant

Timings are of when each copy was recorded in the command stream (the CPU's clock), not of when the display showed it: they
show the pace the frontend ran at, and a vsync-locked run keeps them at the display's rate.

Copyright (C) 2026 Mihawk
SPDX-License-Identifier: GPL-3.0-or-later
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def load(folder):
    w, h, n = map(int, open(os.path.join(folder, 'size.txt')).read().split())
    frames = np.fromfile(os.path.join(folder, 'frames.rgba'), dtype=np.uint8)
    n = min(n, frames.size // (w * h * 4))
    frames = frames[:n * w * h * 4].reshape(n, h, w, 4)[:, :, :, :3]
    rows = []
    for line in open(os.path.join(folder, 'index.txt')):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        rows.append(dict(place=int(f[1]), ns=int(f[2]), gen=f[3] == '1', t=float(f[4]), unique=int(f[5]), level=int(f[6])))
    return frames, rows[:n]


def gray(frame):
    return frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def phase_shift(a, b):
    """The displacement (dx, dy) of b against a, in pixels, by phase correlation (windowed, sub-pixel by a parabola),
    and the peak's strength (0..1): a picture with nothing to track gives a weak one."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa, fb = np.fft.rfft2((a - a.mean()) * win), np.fft.rfft2((b - b.mean()) * win)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-6)
    r = np.fft.irfft2(cross, s=a.shape)
    y, x = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[y, x])

    def refine(m, c, p):
        d = m - 2 * c + p
        return 0.0 if abs(d) < 1e-9 else 0.5 * (m - p) / d
    dx = x + refine(r[y, (x - 1) % w], r[y, x], r[y, (x + 1) % w])
    dy = y + refine(r[(y - 1) % h, x], r[y, x], r[(y + 1) % h, x])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy), peak


def save_plot(path, series, labels, size=(900, 300)):
    img = Image.new('RGB', size, (20, 20, 24))
    d = ImageDraw.Draw(img)
    allv = [v for s in series for v in s if v == v]
    if not allv:
        img.save(path)
        return
    lo, hi = min(allv + [0.0]), max(allv + [0.0])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    m = 28
    def pt(i, v, n):
        return m + (size[0] - 2 * m) * i / max(1, n - 1), size[1] - m - (size[1] - 2 * m) * (v - lo) / (hi - lo)
    d.line([pt(0, 0.0, 2), pt(1, 0.0, 2)], fill=(70, 70, 80))
    colors = [(90, 200, 255), (255, 170, 60)]
    for k, s in enumerate(series):
        d.line([pt(i, v, len(s)) for i, v in enumerate(s)], fill=colors[k % 2], width=2)
        d.text((m + 8 + 160 * k, 6), labels[k], fill=colors[k % 2])
    d.text((4, size[1] - m), f'{lo:.1f}', fill=(160, 160, 170))
    d.text((4, m - 4), f'{hi:.1f}', fill=(160, 160, 170))
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('folder')
    ap.add_argument('--out')
    ap.add_argument('--sheet-start', type=int, default=0)
    ap.add_argument('--sheet-count', type=int, default=12)
    ap.add_argument('--row', type=float, default=0.5, help='the row of xt.png, as a fraction of the height')
    ap.add_argument('--region', help='x0,y0,x1,y1 as fractions of the picture: the motion is tracked in it alone (a scene with a fixed HUD)')
    ap.add_argument('--tol', type=int, default=12, help='the levels a generated pixel may lie outside its neighbours')
    args = ap.parse_args()
    frames, meta = load(args.folder)
    n, h, w, _ = frames.shape
    out = args.out or os.path.join(args.folder, 'analysis')
    os.makedirs(out, exist_ok=True)
    rep = []

    def say(s=''):
        rep.append(s)
        print(s)

    say(f'{n} pictures of {w}x{h} from {args.folder}')

    # ---- timing
    ns = np.array([m['ns'] for m in meta], dtype=np.float64)
    dt = np.diff(ns) / 1e6
    if len(dt):
        med = float(np.median(dt))
        say(f'pace: median {med:.2f} ms a picture ({1000.0 / med:.1f}/s), mean {dt.mean():.2f}, min {dt.min():.2f}, max {dt.max():.2f}, '
            f'{int((dt > 1.5 * med).sum())} gaps over 1.5x the median, {int((dt < 0.5 * med).sum())} bunched under half of it')
    pattern = ''.join('G' if m['gen'] else 'R' for m in meta)
    say(f'pattern: {pattern[:120]}{"..." if n > 120 else ""}')
    say(f'generated {pattern.count("G")}, real {pattern.count("R")}; new frames in the window: {meta[-1]["unique"] - meta[0]["unique"]}; '
        f'governor level {meta[0]["level"]} to {meta[-1]["level"]}')
    gaps = [b['place'] - a['place'] for a, b in zip(meta, meta[1:])]
    skipped = sum(1 for g in gaps if g != 1)
    if skipped:
        say(f'places skipped (a picture not recorded between two): {skipped}')

    # ---- change from one picture to the next, repeats and motion
    x0, y0, x1, y1 = (0.0, 0.0, 1.0, 1.0)
    if args.region:
        x0, y0, x1, y1 = map(float, args.region.split(','))
    small = [gray(f[int(y0 * h):max(int(y1 * h), int(y0 * h) + 8):2, int(x0 * w):max(int(x1 * w), int(x0 * w) + 8):2]) for f in frames]
    change = np.array([float(np.abs(small[i + 1] - small[i]).mean()) for i in range(n - 1)])
    repeats = [i for i in range(n - 1) if change[i] < 0.02]
    say(f'repeats (a picture equal to the one before it): {len(repeats)} of {n - 1}')
    disp, strength = [], []
    for i in range(n - 1):
        dx, dy, pk = phase_shift(small[i], small[i + 1])
        disp.append((2 * dx, 2 * dy))
        strength.append(pk)
    disp = np.array(disp)
    mag = np.hypot(disp[:, 0], disp[:, 1])
    strong = np.array(strength) > 0.1
    say(f'motion of the whole picture, pixels per picture: mean {mag[strong].mean() if strong.any() else 0.0:.2f} '
        f'over {int(strong.sum())} pictures with something to track')
    # The moment of each picture, in new-frame intervals: a real picture is the held frame's own (its count), a generated one
    # lies t of the way back from it. The displacement per interval must be the same from picture to picture whatever
    # their spacing: a stream that moves at half the rate alternates 0 and twice the rate; one that jitters scatters.
    when = np.array([m['unique'] if not m['gen'] else m['unique'] - 1 + m['t'] for m in meta], dtype=np.float64)
    step = np.diff(when)
    rate = np.where(step > 1e-6, mag / np.maximum(step, 1e-6), np.nan)
    ok = strong & (step > 1e-6)
    if ok.sum() > 12:
        idx = np.where(ok)[0]
        resid = []
        for k, i in enumerate(idx):
            near = [rate[j] for j in idx[max(0, k - 4):k + 5] if j != i]
            if len(near) >= 3 and np.median(near) > 0.3:
                resid.append((i, abs(rate[i] - np.median(near)) / np.median(near)))
        if resid:
            r = np.array([x[1] for x in resid])
            say(f'evenness of the motion (the displacement per new-frame interval against the pictures around it; 0 is perfectly even): '
                f'median deviation {100 * np.median(r):.0f}%, 90th percentile {100 * np.percentile(r, 90):.0f}% over {len(r)} pictures')
            kinds = {}
            for i, dev in resid:
                kinds.setdefault(pattern[i] + pattern[i + 1], []).append(dev)
            for key in sorted(kinds):
                say(f'  steps {key[0]} -> {key[1]}: {len(kinds[key])}, median deviation {100 * np.median(kinds[key]):.0f}%')
    # Where each generated picture sits between its two real neighbours. Between two consecutive real pictures a and b the game
    # moved by V (measured); a generated picture shown tau of the way from a to b should be displaced from a by V * tau. The
    # error is the distance from that, as a share of V, so it reads in frame intervals: 0.1 is a picture that sits where the game
    # would have been 0.1 of a new-frame interval earlier or later (0 for a perfect one; a repeat of a scores tau, a blend 0.5).
    reals = [i for i, m in enumerate(meta) if not m['gen']]
    errs, rows_err = [], []
    for a_, b_ in zip(reals, reals[1:]):
        if meta[b_]['unique'] - meta[a_]['unique'] != 1 or b_ - a_ < 2:
            continue
        vx, vy, pk = phase_shift(small[a_], small[b_])
        speed = float(np.hypot(vx, vy)) * 2
        if pk < 0.1 or speed < 3.0:
            continue
        for i in range(a_ + 1, b_):
            tau = meta[i]['t']
            dx, dy, pk2 = phase_shift(small[a_], small[i])
            e = float(np.hypot(2 * dx - 2 * vx * tau, 2 * dy - 2 * vy * tau)) / speed
            errs.append(e)
            rows_err.append((i, tau, e))
    if errs:
        e = np.array(errs)
        say(f'position of generated pictures between their real neighbours (error in new-frame intervals; 0 is exact): '
            f'median {np.median(e):.2f}, 90th percentile {np.percentile(e, 90):.2f}, worst {e.max():.2f}, over {len(e)} pictures')
        by = {}
        for i, tau, er in rows_err:
            by.setdefault(round(tau, 2), []).append(er)
        for tau in sorted(by):
            say(f'  at t={tau:.2f}: {len(by[tau])} pictures, median error {np.median(by[tau]):.2f}')
    save_plot(os.path.join(out, 'motion.png'), [list(disp[:, 0]), list(disp[:, 1])], ['dx per picture', 'dy per picture'])

    # ---- flicker: a generated pixel outside both neighbours' values
    heat = np.zeros((h, w), np.float32)
    scores = []
    for i in range(1, n - 1):
        if not meta[i]['gen']:
            continue
        p, g, q = (frames[i - 1].astype(np.int16), frames[i].astype(np.int16), frames[i + 1].astype(np.int16))
        lo, hi = np.minimum(p, q), np.maximum(p, q)
        over = np.maximum(np.maximum(lo - g, g - hi) - args.tol, 0).max(axis=2)
        bad = over > 0
        heat += bad
        scores.append((i, float(bad.mean()), float(over.max())))
    if scores:
        fr = np.array([s[1] for s in scores])
        say(f'flicker (generated pixels outside both neighbours by over {args.tol} levels): mean {100 * fr.mean():.2f}% of the picture, '
            f'worst {100 * fr.max():.2f}% at picture {scores[int(fr.argmax())][0]}')
        worst = sorted(scores, key=lambda s: -s[1])[:5]
        say('  worst pictures: ' + ', '.join(f'{i} ({100 * f:.1f}%)' for i, f, _ in worst))
        # sharpness: the brightness gradient of each generated picture against the mean of the real pictures either side. A soft
        # generated picture between sharp real ones makes the sharpness pulse at the real frames' rate, which shows as shimmer.
        def sharp(img):
            g = gray(img)
            return float(np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean())
        by_t = {}
        for i, r in enumerate(meta):
            if not r['gen']:
                continue
            j, k = i - 1, i + 1
            while j >= 0 and meta[j]['gen']:
                j -= 1
            while k < len(meta) and meta[k]['gen']:
                k += 1
            if j < 0 or k >= len(meta):
                continue
            by_t.setdefault(round(r['t'], 2), []).append(sharp(frames[i]) / max((sharp(frames[j]) + sharp(frames[k])) / 2, 1e-6))
        if by_t:
            say('sharpness of the generated pictures against the real ones either side (1.00 is as sharp): ' +
                ', '.join(f't={t:.2f}: {np.mean(v):.3f}' for t, v in sorted(by_t.items())))
        heat /= len(scores)
        img = (np.clip(heat * 4.0, 0, 1) * 255).astype(np.uint8)
        base = frames[len(frames) // 2].copy() // 3
        rgb = np.stack([np.maximum(base[:, :, 0], img), base[:, :, 1], base[:, :, 2]], axis=2)
        Image.fromarray(rgb).save(os.path.join(out, 'flicker.png'))
    else:
        say('flicker: no generated pictures with two neighbours in the window')

    # ---- x-t slice
    y = int(np.clip(args.row, 0, 1) * (h - 1))
    xt = np.stack([f[y] for f in frames])
    bar = np.zeros((n, 8, 3), np.uint8)
    for i, m in enumerate(meta):
        bar[i] = (60, 200, 90) if m['gen'] else (120, 120, 120)
    Image.fromarray(np.concatenate([bar, xt], axis=1)).save(os.path.join(out, 'xt.png'))

    # ---- contact sheet
    cols = 4
    tw = 320
    th = int(h * tw / w)
    start = int(np.clip(args.sheet_start, 0, max(0, n - 1)))
    cnt = min(args.sheet_count, n - start)
    rows = (cnt + cols - 1) // cols
    sheet = Image.new('RGB', (cols * tw, rows * (th + 14)), (16, 16, 20))
    d = ImageDraw.Draw(sheet)
    for k in range(cnt):
        i = start + k
        tile = Image.fromarray(frames[i]).resize((tw, th), Image.BILINEAR)
        x0, y0 = (k % cols) * tw, (k // cols) * (th + 14)
        sheet.paste(tile, (x0, y0 + 14))
        m = meta[i]
        d.text((x0 + 4, y0 + 1), f'#{i} place {m["place"]} {"G t=%.2f" % m["t"] if m["gen"] else "R"}',
               fill=(110, 230, 130) if m['gen'] else (200, 200, 200))
    sheet.save(os.path.join(out, 'sheet.png'))
    open(os.path.join(out, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print(f'\nwritten to {out}: report.txt flicker.png motion.png xt.png sheet.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
