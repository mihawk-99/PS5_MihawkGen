# PS5_FrameGen - the situations suite: controlled pictures of one situation each (scrolls, camera pans, zoom, rotation, parallax, sprites,
# occlusion, explosions, particles, fades, HUD over motion, thin lines, repeating patterns, ripples, accelerating motion, dark scenes,
# stillness), each rendered at three instants (frame a at 0, the true middle picture m at 1/2, frame c at 1), so every method's middle
# picture has an exact truth and a sweep of magnitude per situation. The textures are real frames of clips never trained on (the clean
# benchmark's), moved analytically: sampled on a 4x grid and averaged down (nearest sampling for crisp pixel art, so an even scroll stays exact).
#
#   .venv/bin/python train/situations.py make [--per 8]      writes ~/retro-tests/situations/<situation>-<level>_<k>.npz (a, m, c, p)
#   .venv/bin/python train/situations.py sheet NAME [N]       a contact sheet of one situation's first N triplets (a | truth | c)
#   bench_all.py run s METHOD ...; bench_all.py grid s METHOD ...    (set "s": one row per situation-level)
#
# Not a replacement for the real-footage benchmark: it shows where a method breaks and how fast it degrades with magnitude.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RT = os.path.expanduser('~/retro-tests')
OUT = RT + '/situations'
H, W, S = 288, 512, 4
TAUS = (0.0, 0.5, 1.0)       # the instants of the three pictures: frame a, the middle picture, frame c (training sets the middle one at random)
DEV = 'cuda'


# ---- source pictures

def pools():
    excluded = set(l.strip() for l in open(RT + '/bench-excluded.txt'))
    pix, smooth, seen = [], [], set()
    for d, dst in ((RT + '/bench-triplets', pix), (RT + '/bench-triplets-b', smooth)):
        for f in sorted(glob.glob(d + '/*.npz')):
            g = os.path.basename(f)[:-4].rsplit('_', 1)[0]
            if g in excluded:
                continue
            z = np.load(f)
            for i in (0, len(z['a']) // 2):
                im = z['a'][i]
                if im.std() < 25 or (im.max(2) < 20).mean() > 0.35:      # flat, or mostly black
                    continue
                key = (g, i)
                if key in seen:
                    continue
                seen.add(key)
                dst.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA if im.shape[1] > W else cv2.INTER_LINEAR))
    return pix, smooth


def tex(im, alpha=None):
    t = torch.from_numpy(im).to(DEV).permute(2, 0, 1).float() / 255
    a = torch.ones(1, *t.shape[1:], device=DEV) if alpha is None else torch.as_tensor(alpha, device=DEV).float()[None]
    return torch.cat([t, a], 0)


def disc_alpha(h, w, soft=0.15, hard=False):
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt(((x + .5 - w / 2) / (w / 2)) ** 2 + ((y + .5 - h / 2) / (h / 2)) ** 2)
    return (r < 1).astype(np.float32) if hard else np.clip((1 - r) / soft, 0, 1).astype(np.float32)


# ---- the renderer: layers with a pose per instant, sampled at S x S and averaged down

class Layer:
    def __init__(self, t, pose=None, mode='bilinear', periodic=False, additive=False, warp=None):
        self.t, self.mode, self.periodic, self.additive, self.warp = t, mode, periodic, additive, warp
        self.pose = pose or (lambda tau: (0.0, 0.0, 0.0, 1.0))      # (tx, ty, rotation in radians, zoom)


_grid_cache = {}


def screen_grid():
    if S not in _grid_cache:
        ys = (torch.arange(H * S, device=DEV).float() + 0.5) / S
        xs = (torch.arange(W * S, device=DEV).float() + 0.5) / S
        _grid_cache[S] = torch.meshgrid(ys, xs, indexing='ij')
    return _grid_cache[S]


def render(layers, tau, bg=0.0):
    ys, xs = screen_grid()
    out = torch.full((3, H * S, W * S), float(bg), device=DEV)
    for L in layers:
        tx, ty, th, z = L.pose(tau)
        dx, dy = xs - W / 2 - tx, ys - H / 2 - ty
        c, s = math.cos(th), math.sin(th)
        th_, tw_ = L.t.shape[-2:]
        u = (c * dx + s * dy) / z + tw_ / 2
        v = (-s * dx + c * dy) / z + th_ / 2
        if L.warp is not None:
            u, v = L.warp(u, v, tau)
        if L.periodic:
            u, v = torch.remainder(u, tw_), torch.remainder(v, th_)
        g = torch.stack([u / tw_ * 2 - 1, v / th_ * 2 - 1], -1)[None]
        smp = F.grid_sample(L.t[None], g, mode=L.mode, padding_mode='zeros', align_corners=False)[0]
        rgb, a = smp[:3], smp[3:]
        out = out + rgb * a if L.additive else out * (1 - a) + rgb * a
    return F.avg_pool2d(out[None], S)[0].clamp(0, 1)


LAST_FLOW = None      # (flow to frame a, flow to frame c) of the last shot's middle instant, (2, H, W) each, or None when it has none (effects, ripples)


def flow_gt(layers, tau_t, tau0=0.0, tau1=1.0):
    """Exact flow of the topmost layer at each pixel of the picture at tau_t to where the same surface point is at tau0 and tau1, in pixels
    (the sampling flow of model.warp: out(x) = img(x + flow))."""
    ys, xs = torch.meshgrid(torch.arange(H, device=DEV).float() + 0.5, torch.arange(W, device=DEV).float() + 0.5, indexing='ij')
    f0 = torch.zeros(2, H, W, device=DEV)
    f1 = torch.zeros(2, H, W, device=DEV)
    for L in layers:
        tx, ty, th, z = L.pose(tau_t)
        dx, dy = xs - W / 2 - tx, ys - H / 2 - ty
        c, s = math.cos(th), math.sin(th)
        th_, tw_ = L.t.shape[-2:]
        u = (c * dx + s * dy) / z + tw_ / 2
        v = (-s * dx + c * dy) / z + th_ / 2
        uw, vw = torch.remainder(u, tw_) if L.periodic else u, torch.remainder(v, th_) if L.periodic else v
        inside = (uw >= 0) & (uw < tw_) & (vw >= 0) & (vw < th_)
        ui, vi = uw.long().clamp(0, tw_ - 1), vw.long().clamp(0, th_ - 1)
        vis = inside & (L.t[3][vi, vi * 0 + ui] > 0.5) if False else inside & (L.t[3][vi, ui] > 0.5)
        for tau, out in ((tau0, f0), (tau1, f1)):
            ttx, tty, tth, tz = L.pose(tau)
            c2, s2 = math.cos(tth), math.sin(tth)
            px = tz * (c2 * (u - tw_ / 2) - s2 * (v - th_ / 2)) + W / 2 + ttx
            py = tz * (s2 * (u - tw_ / 2) + c2 * (v - th_ / 2)) + H / 2 + tty
            out[0] = torch.where(vis, px - xs, out[0])
            out[1] = torch.where(vis, py - ys, out[1])
    return f0, f1


def shot(layers, bg=0.0, taus=None, extra=None):
    global LAST_FLOW
    ims = []
    taus_ = taus or TAUS
    LAST_FLOW = None if (extra is not None or any(L.warp is not None for L in layers)) else flow_gt(layers, taus_[1], taus_[0], taus_[2])
    for tau in taus_:
        im = render(layers, tau, bg)
        if extra is not None:
            im = extra(im, tau)
        ims.append(im.mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    return ims


def lin(vx, vy):
    return lambda tau: (vx * tau, vy * tau, 0.0, 1.0)


def rnd_dir(rng):
    a = rng.uniform(0, 2 * math.pi)
    return math.cos(a), math.sin(a)


# ---- the situations: each returns a function (rng, pool_pix, pool_smooth) -> (a, m, c)

def scroll(px, axis):
    def f(rng, pix, smooth):
        t = tex(pix[rng.integers(len(pix))])
        vx = px * (axis in 'xd') * rng.choice([-1, 1]); vy = px * (axis in 'yd') * rng.choice([-1, 1])
        return shot([Layer(t, lin(vx, vy), 'nearest', periodic=True)])
    return f


def pan(px):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        dx, dy = rnd_dir(rng)
        return shot([Layer(t, lin(dx * px, dy * px * 0.4), 'bilinear', periodic=True)])
    return f


def zoom(k):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        zi = k if rng.random() < 0.5 else 1 / k
        return shot([Layer(t, lambda tau, zi=zi: (0.0, 0.0, 0.0, zi ** tau), 'bilinear', periodic=True)])
    return f


def rotate(deg):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        sg = rng.choice([-1, 1])
        return shot([Layer(t, lambda tau: (0.0, 0.0, sg * math.radians(deg) * tau, 1.0), 'bilinear', periodic=True)])
    return f


def parallax(px):
    def f(rng, pix, smooth):
        L = []
        for k, (frac, src) in enumerate(((0.25, smooth), (0.6, smooth), (1.0, smooth))):
            im = src[rng.integers(len(src))]
            a = None if k == 0 else np.clip(cv2.GaussianBlur((np.mgrid[0:H, 0:W][0] > H * (0.35 + 0.15 * k)).astype(np.float32), (0, 0), 6), 0, 1)
            L.append(Layer(tex(im, a), lin(-px * frac, 0), 'bilinear', periodic=True))
        return shot(L)
    return f


def sprite(size, px, n=1, opposite=False, rotating=False, hard=True):
    def f(rng, pix, smooth):
        bgim = smooth[rng.integers(len(smooth))]
        L = [Layer(tex(bgim))]
        for i in range(n):
            src = pix[rng.integers(len(pix))]
            h = size; w = int(size * rng.uniform(0.8, 1.6))
            y0 = rng.integers(0, H - h); x0 = rng.integers(0, W - w)
            patch = src[y0:y0 + h, x0:x0 + w]
            t = tex(patch, disc_alpha(h, w, hard=hard) if rng.random() < 0.5 else np.ones((h, w), np.float32))
            dx, dy = rnd_dir(rng)
            if opposite and i == 1:
                dx, dy = -dx0, -dy0
            else:
                dx0, dy0 = dx, dy
            cx = rng.uniform(w, W - w) - W / 2; cy = rng.uniform(h, H - h) - H / 2
            spin = math.radians(rng.choice([-1, 1]) * 20) if rotating else 0.0
            L.append(Layer(t, lambda tau, cx=cx, cy=cy, dx=dx, dy=dy, spin=spin: (cx + dx * px * tau, cy + dy * px * tau, spin * tau, 1.0), 'nearest' if hard else 'bilinear'))
        return shot(L)
    return f


def occlusion(px):
    def f(rng, pix, smooth):
        L = [Layer(tex(smooth[rng.integers(len(smooth))]))]
        for i, sgn in enumerate((1, -1)):
            src = smooth[rng.integers(len(smooth))]
            h, w = 96, 96
            patch = src[100:100 + h, 200:200 + w]
            t = tex(patch, disc_alpha(h, w, hard=True))
            y = rng.uniform(-30, 30)
            L.append(Layer(t, lambda tau, sgn=sgn, y=y: (sgn * (-px / 2 + px * tau), y + (i * 8) * 0, 0.0, 1.0), 'nearest'))
        return shot(L)
    return f


def fireball(growth):
    def f(rng, pix, smooth):
        bg = tex(smooth[rng.integers(len(smooth))])
        cx, cy = rng.uniform(150, 360), rng.uniform(90, 200)
        ys, xs = torch.meshgrid(torch.arange(H, device=DEV).float() + .5, torch.arange(W, device=DEV).float() + .5, indexing='ij')
        noise = F.interpolate(torch.rand(1, 1, 18, 32, device=DEV), (H, W), mode='bicubic', align_corners=False)[0, 0]

        def at(im, tau):
            r = 20 + growth * tau
            d = ((xs - cx) ** 2 + (ys - cy) ** 2).sqrt() / r
            core = torch.exp(-(d ** 2) * (1.5 + 2 * noise)) * (1.2 - 0.5 * tau)
            col = torch.stack([core * 1.0, core * 0.65, core * 0.25])
            return (im + col).clamp(0, 1)
        return shot([Layer(bg)], extra=at)
    return f


def particles(speed, n=200):
    def f(rng, pix, smooth):
        bg = tex(smooth[rng.integers(len(smooth))])
        p0 = torch.tensor(np.stack([rng.uniform(100, 400, n), rng.uniform(60, 220, n)], 1), device=DEV).float()
        a = rng.uniform(0, 2 * math.pi, n); sp = rng.uniform(0.2, 1.0, n) * speed
        v = torch.tensor(np.stack([np.cos(a) * sp, np.sin(a) * sp - speed * 0.3], 1), device=DEV).float()
        g = torch.tensor([0.0, speed * 0.5], device=DEV)
        sig = torch.tensor(rng.uniform(0.9, 2.2, n), device=DEV).float()
        br = torch.tensor(rng.uniform(0.5, 1.0, n), device=DEV).float()
        col = torch.tensor(rng.uniform(0.6, 1.0, (n, 3)), device=DEV).float()
        ys, xs = torch.meshgrid(torch.arange(H, device=DEV).float() + .5, torch.arange(W, device=DEV).float() + .5, indexing='ij')

        def at(im, tau):
            pos = p0 + v * tau + 0.5 * g * tau * tau
            acc = torch.zeros(3, H, W, device=DEV)
            for i in range(0, n, 50):
                d2 = (xs[None] - pos[i:i + 50, 0, None, None]) ** 2 + (ys[None] - pos[i:i + 50, 1, None, None]) ** 2
                w_ = torch.exp(-d2 / (2 * sig[i:i + 50, None, None] ** 2)) * br[i:i + 50, None, None]
                acc += torch.einsum('nhw,nc->chw', w_, col[i:i + 50])
            return (im + acc).clamp(0, 1)
        return shot([Layer(bg)], extra=at)
    return f


def smoke(growth):
    def f(rng, pix, smooth):
        bg = tex(smooth[rng.integers(len(smooth))])
        cx, cy = rng.uniform(150, 360), rng.uniform(100, 220)
        ys, xs = torch.meshgrid(torch.arange(H, device=DEV).float() + .5, torch.arange(W, device=DEV).float() + .5, indexing='ij')
        noise = F.interpolate(torch.rand(1, 1, 14, 24, device=DEV), (H, W), mode='bicubic', align_corners=False)[0, 0]

        def at(im, tau):
            r = 30 + growth * tau
            d = ((xs - cx) ** 2 + (ys - cy + 40 * tau) ** 2).sqrt() / r
            alpha = (torch.exp(-(d ** 2) * (1 + 3 * noise)) * (0.85 - 0.35 * tau)).clamp(0, 1)
            return im * (1 - alpha) + 0.35 * alpha
        return shot([Layer(bg)], extra=at)
    return f


def fade(kind, delta, alpha0=None):
    """A fade or a dissolve as a game makes it: over a gap of two frames the picture changes by a small share (delta) of the whole fade,
    from a random point of it (a whole fade inside one gap is a cut, which no method can tell from a dissolve)."""
    def f(rng, pix, smooth):
        a = tex(smooth[rng.integers(len(smooth))])
        d = delta
        a0 = rng.uniform(0, 1 - d) if alpha0 is None else alpha0
        al = lambda tau: a0 + d * tau
        if kind == 'dissolve':
            b = tex(smooth[rng.integers(len(smooth))])[:3]
            return shot([Layer(a)], extra=lambda im, tau: im * (1 - al(tau)) + b * al(tau))
        if kind == 'black':
            return shot([Layer(a)], extra=lambda im, tau: im * (1 - al(tau)))
        return shot([Layer(a)], extra=lambda im, tau: im * (1 - al(tau)) + al(tau))      # white
    return f


def hud(px, text=True):
    def f(rng, pix, smooth):
        ov = np.zeros((H, W, 4), np.uint8)
        cv2.putText(ov, 'SCORE 012340', (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ov, 'HP', (14, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(ov, (60, H - 36), (200, H - 20), (230, 60, 60, 255), -1)
        cv2.putText(ov, 'x3', (W - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 120, 255), 2, cv2.LINE_AA)
        t = torch.from_numpy(ov).to(DEV).permute(2, 0, 1).float() / 255
        dx, dy = rnd_dir(rng)
        return shot([Layer(tex(smooth[rng.integers(len(smooth))]), lin(dx * px, dy * px * .4), 'bilinear', periodic=True), Layer(t, mode='nearest')])
    return f


def ticker(px):
    def f(rng, pix, smooth):
        ov = np.zeros((H, W * 2, 4), np.uint8)
        ov[:, :, 3] = 0
        cv2.rectangle(ov, (0, H - 50), (W * 2, H - 10), (0, 0, 0, 200), -1)
        cv2.putText(ov, 'BREAKING: FRAME GENERATION ON THE PS5 RUNS AT 120 FPS - EVERYTHING MOVES AT 120 FPS - ' * 2, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA)
        t = torch.from_numpy(ov).to(DEV).permute(2, 0, 1).float() / 255
        return shot([Layer(tex(smooth[rng.integers(len(smooth))])), Layer(t, lin(-px, 0), 'nearest', periodic=True)])
    return f


def pattern(kind, px):
    def f(rng, pix, smooth):
        y, x = np.mgrid[0:H, 0:W]
        if kind == 'stripes':
            im = (((x // 3) % 2) * 200 + 30).astype(np.uint8)
        elif kind == 'checker':
            im = ((((x // 4) + (y // 4)) % 2) * 200 + 30).astype(np.uint8)
        elif kind == 'lines':       # thin one-pixel wires on a gradient
            im = np.tile((np.linspace(40, 160, W)[None] + 0 * y).astype(np.uint8), 1)
            im[::17, :] = 250
            im[:, ::29] = 250
        else:                       # noise
            im = rng.integers(0, 255, (H, W)).astype(np.uint8)
        t = tex(np.stack([im] * 3, -1))
        dx, dy = rnd_dir(rng)
        return shot([Layer(t, lin(dx * px, dy * px), 'bilinear', periodic=True)])
    return f


def ripple(px):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        amp = 6.0
        warp = lambda u, v, tau: (u + amp * torch.sin(v / 12 + tau * px), v)
        return shot([Layer(t, lin(0, 0), 'bilinear', periodic=True, warp=warp)])
    return f


def accel(px):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        dx, dy = rnd_dir(rng)
        return shot([Layer(t, lambda tau: (dx * px * tau * tau, dy * px * tau * tau, 0.0, 1.0), 'bilinear', periodic=True)])     # truth is not the midpoint
    return f


def dark(px):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        t[:3] *= 0.12
        dx, dy = rnd_dir(rng)
        return shot([Layer(t, lin(dx * px, dy * px * .4), 'bilinear', periodic=True)])
    return f


def still(delta):
    def f(rng, pix, smooth):
        t = tex(smooth[rng.integers(len(smooth))])
        return shot([Layer(t, lambda tau: (0.0, 0.0, 0.0, 1.0))], extra=lambda im, tau: (im + delta / 255 * tau).clamp(0, 1))
    return f


SITUATIONS = {}
for px in (2, 4, 8, 16, 32, 64):
    SITUATIONS[f'scrollx-{px}'] = scroll(px, 'x')
for px in (2, 8, 32):
    SITUATIONS[f'scrolly-{px}'] = scroll(px, 'y')
for px in (6, 10, 14, 22, 38, 70):      # not powers of two: a tile pattern of 8 or 16 pixels does not repeat at these
    SITUATIONS[f'scrollx-{px}'] = scroll(px, 'x')
for px in (6, 14, 22):
    SITUATIONS[f'scrolly-{px}'] = scroll(px, 'y')
SITUATIONS['scrolld-8'] = scroll(8, 'd')
for px in (2, 6, 24, 64, 128):
    SITUATIONS[f'pan-{px}'] = pan(px)
for k in (1.01, 1.05, 1.15, 1.3):
    SITUATIONS[f'zoom-{int(round((k - 1) * 100)):02d}'] = zoom(k)
for deg in (0.5, 2, 6, 15):
    SITUATIONS[f'rotate-{deg}'] = rotate(deg)
for px in (4, 16, 48):
    SITUATIONS[f'parallax-{px}'] = parallax(px)
for size, px in ((16, 8), (16, 40), (48, 8), (48, 40), (128, 8), (128, 40), (48, 100)):
    SITUATIONS[f'sprite-s{size}v{px}'] = sprite(size, px)
SITUATIONS['sprite-two-opposite'] = sprite(48, 30, n=2, opposite=True)
SITUATIONS['sprite-many'] = sprite(24, 20, n=6)
SITUATIONS['sprite-spinning'] = sprite(64, 24, rotating=True)
SITUATIONS['sprite-soft'] = sprite(48, 24, hard=False)
for px in (60, 140):
    SITUATIONS[f'occlusion-{px}'] = occlusion(px)
for g in (20, 60, 140):
    SITUATIONS[f'fireball-{g}'] = fireball(g)
for sp in (20, 60, 140):
    SITUATIONS[f'particles-{sp}'] = particles(sp)
for g in (20, 80):
    SITUATIONS[f'smoke-{g}'] = smoke(g)
for kind in ('dissolve', 'black', 'white'):
    for d in (0.05, 0.2):
        SITUATIONS[f'fade-{kind}-d{int(d * 100):02d}'] = fade(kind, d)
for px in (6, 40):
    SITUATIONS[f'hud-over-pan-{px}'] = hud(px)
for px in (8, 24):
    SITUATIONS[f'ticker-{px}'] = ticker(px)
for kind in ('stripes', 'checker', 'lines', 'noise'):
    for px in (3, 11):
        SITUATIONS[f'pattern-{kind}-{px}'] = pattern(kind, px)
for sp in (2, 6):
    SITUATIONS[f'ripple-{sp}'] = ripple(sp)
for px in (16, 64):
    SITUATIONS[f'accel-{px}'] = accel(px)
for px in (4, 24):
    SITUATIONS[f'dark-{px}'] = dark(px)
SITUATIONS['still-0'] = still(0)
SITUATIONS['still-6'] = still(6)


def random_situation(rng):
    """One situation with a magnitude drawn at random, for training (the pictures come from the training games; the held-out suite above
    takes its pictures from games never trained on). Accelerating motion is left out: two frames cannot tell its truth."""
    lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    u = rng.random()
    if u < 0.14:
        return scroll(2 * max(1, int(round(lg(1, 40)))), 'xyd'[rng.integers(3)])
    if u < 0.24:
        return pan(lg(1, 140))
    if u < 0.31:
        return zoom(1 + lg(0.005, 0.35))
    if u < 0.37:
        return rotate(lg(0.3, 18))
    if u < 0.45:
        return parallax(lg(2, 60))
    if u < 0.57:
        return sprite(int(lg(12, 140)), lg(2, 110), n=int(rng.choice([1, 1, 2, 3, 6])), opposite=False, rotating=rng.random() < 0.2, hard=rng.random() < 0.7)
    if u < 0.61:
        return occlusion(lg(40, 150))
    if u < 0.66:
        return fireball(lg(10, 150))
    if u < 0.71:
        return particles(lg(10, 150))
    if u < 0.74:
        return smoke(lg(10, 90))
    if u < 0.82:
        return fade(['dissolve', 'black', 'white'][rng.integers(3)], lg(0.02, 0.3))
    if u < 0.87:
        return hud(lg(3, 50))
    if u < 0.91:
        return ticker(lg(4, 32))
    if u < 0.94:
        return ripple(lg(1, 7))
    if u < 0.97:
        return dark(lg(2, 40))
    return still(float(rng.choice([0, 3, 8])))


def make(per):
    pix, smooth = pools()
    print(len(pix), 'pixel pictures,', len(smooth), '3D pictures', flush=True)
    os.makedirs(OUT, exist_ok=True)
    for name, fn in SITUATIONS.items():
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32) if False else sum(ord(c) * (i + 1) for i, c in enumerate(name)))
        a, m, c = [], [], []
        for _ in range(per):
            ims = fn(rng, pix, smooth)
            a.append(ims[0]); m.append(ims[1]); c.append(ims[2])
        np.savez(f'{OUT}/{name}_0.npz', a=np.stack(a), m=np.stack(m), c=np.stack(c), p=np.stack(a))
        print(name, flush=True)


def sheet(name, n=4):
    z = np.load(f'{OUT}/{name}_0.npz')
    rows = [np.concatenate([z['a'][i], z['m'][i], z['c'][i]], 1) for i in range(min(n, len(z['a'])))]
    path = f'/tmp/claude-1000/sheet-{name}.png'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(path)


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.3)     # the desktop shares this GPU: never starve it
    if sys.argv[1] == 'make':
        make(int(sys.argv[sys.argv.index('--per') + 1]) if '--per' in sys.argv else 8)
    elif sys.argv[1] == 'sheet':
        sheet(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)


class Pool:
    """Pictures of the training games as textures for random_situation: 288x512 uint8 arrays, 2D games as `pix`, 3D games (the 3d.list tags)
    as `smooth`. Taken from the clips real.Clips holds."""

    def __init__(self, clips, n=1500, seed=3):
        rng = np.random.default_rng(seed)
        self.pix, self.smooth = [], []
        w = np.array(clips.weights, np.float64)
        for _ in range(n):
            ci = rng.choice(len(clips.clips), p=w / w.sum())
            fr = clips.clips[ci][clips.new[ci][rng.integers(len(clips.new[ci]))]].numpy()
            if fr.std() < 25 or (fr.max(2) < 20).mean() > 0.35:
                continue
            (self.smooth if clips.weights[ci] > 1.0 else self.pix).append(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA if fr.shape[1] > W else cv2.INTER_LINEAR))
        if not self.smooth:
            self.smooth = self.pix
        print('situation textures:', len(self.pix), '2D pictures,', len(self.smooth), '3D pictures', flush=True)

    def batch(self, n, crop, rng, device):
        """n triplets (a, c, target, t, flow to a, flow to c, has_flow) of crop x crop: the target's instant 1/2 (half of them) or 1/4, 3/4; flips and a
        swap as real clips get. The flows are exact where the situation moves rigidly (has_flow 1), zero otherwise."""
        global S, TAUS
        S = 2
        outs = [[] for _ in range(7)]
        for _ in range(n):
            t = float(rng.choice([0.5, 0.5, 0.25, 0.75]))
            TAUS = (0.0, t, 1.0)
            ims = random_situation(rng)(rng, self.pix, self.smooth)
            y0, x0 = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
            a, m, c = [torch.from_numpy(i[y0:y0 + crop, x0:x0 + crop]).to(device).permute(2, 0, 1).float() / 255 for i in ims]
            has = LAST_FLOW is not None
            fa, fc = (LAST_FLOW[0][:, y0:y0 + crop, x0:x0 + crop], LAST_FLOW[1][:, y0:y0 + crop, x0:x0 + crop]) if has else (torch.zeros(2, crop, crop, device=device),) * 2
            if rng.random() < 0.5:
                a, c, fa, fc, t = c, a, fc, fa, 1 - t
            if rng.random() < 0.5:
                a, m, c, fa, fc = a.flip(-1), m.flip(-1), c.flip(-1), fa.flip(-1) * torch.tensor([-1.0, 1.0], device=device).view(2, 1, 1), fc.flip(-1) * torch.tensor([-1.0, 1.0], device=device).view(2, 1, 1)
            if rng.random() < 0.5:
                a, m, c, fa, fc = a.flip(-2), m.flip(-2), c.flip(-2), fa.flip(-2) * torch.tensor([1.0, -1.0], device=device).view(2, 1, 1), fc.flip(-2) * torch.tensor([1.0, -1.0], device=device).view(2, 1, 1)
            perm = torch.randperm(3, device=device)
            for k, v in enumerate((a[perm], c[perm], m[perm], torch.tensor(t, device=device), fa, fc, torch.tensor(float(has), device=device))):
                outs[k].append(v)
        TAUS = (0.0, 0.5, 1.0)
        return tuple(torch.stack(v) for v in outs)
