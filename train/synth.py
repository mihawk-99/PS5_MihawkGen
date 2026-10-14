# PS5_FrameGen - the data factory's first source: layered scenes with exact
# time. A scene is a stack of textured, masked layers (a background, shapes,
# particles, a fixed HUD) that move under their own and a shared camera's
# similarity and perspective motion; it is rendered at any time, so every clip
# has the true in-between frames at 240 Hz (five frames, t = 0, 1/4, 1/2, 3/4, 1
# of a 60 Hz game's gap of four steps). The scenes show what interpolation must
# get right: occlusion and disocclusion, transparency, additive effects, thin
# detail, a screen-fixed HUD over a moving world, scrolling surfaces, fades and
# a retro look (pixelation, 15-bit colour, ordered dither).
#
# Everything here is ours: procedural textures from noise, cells, stripes,
# gradients and drawn primitives. No game, image or dataset is used
# (docs/plan.md, "The data rule"). The ideas are the standard ones of layered
# synthetic motion (a stack of moving sprites over a background, as in the
# FlyingChairs and Sprites datasets of the optical-flow literature).
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import math

import torch
import torch._dynamo
import torch.nn.functional as F

# one compiled variant per layer index (the HUD layers take another branch)
torch._dynamo.config.cache_size_limit = 64

STEPS = 5          # frames per clip: 240 Hz steps 0..4
LAYERS = 8         # the background and seven more
HUDS = 2           # of the last layers, screen-fixed
TEX = 128          # texture size


def _fbm(n, size, gen, device, octaves=4, base=4):
    """n x 1 x size x size fractal noise in [0, 1]."""
    out = torch.zeros(n, 1, size, size, device=device)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        g = base * 2 ** o
        grid = torch.rand(n, 1, g, g, device=device, generator=gen)
        out += amp * F.interpolate(grid, size=(size, size), mode='bicubic', align_corners=False)
        total += amp
        amp *= 0.5
    out = out / total
    lo = out.amin(dim=(1, 2, 3), keepdim=True)
    hi = out.amax(dim=(1, 2, 3), keepdim=True)
    return (out - lo) / (hi - lo + 1e-6)


def _rand(shape, gen, device, lo=0.0, hi=1.0):
    return lo + (hi - lo) * torch.rand(shape, device=device, generator=gen)


def _palette(n, gen, device):
    """Two or three random colours per texture, n x 3 x 3."""
    base = torch.rand(n, 1, 3, device=device, generator=gen)
    spread = _rand((n, 3, 3), gen, device, -0.45, 0.45)
    return (base + spread).clamp(0, 1)


def _grid(size, device):
    c = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size * 2 - 1
    yy, xx = torch.meshgrid(c, c, indexing='ij')
    return xx, yy


def make_textures(n, gen, device, size=TEX):
    """n RGB textures (n x 3 x size x size) of five procedural kinds."""
    pal = _palette(n, gen, device)  # n x 3(colours) x 3(rgb)
    xx, yy = _grid(size, device)
    kinds = []
    # 0: fractal noise coloured between the palette's colours
    f1 = _fbm(n, size, gen, device, octaves=int(torch.randint(3, 6, (1,), generator=gen, device=device)))
    f2 = _fbm(n, size, gen, device, octaves=3, base=8)
    k0 = pal[:, 0].view(n, 3, 1, 1) * (1 - f1) + pal[:, 1].view(n, 3, 1, 1) * f1
    k0 = k0 * (0.75 + 0.5 * f2)
    kinds.append(k0)
    # 1: cells of random colours, with dark borders on some
    pts = int(torch.randint(10, 60, (1,), generator=gen, device=device))
    centres = torch.rand(n, pts, 2, device=device, generator=gen) * 2 - 1
    colours = torch.rand(n, pts, 3, device=device, generator=gen)
    flat = torch.stack([xx, yy], -1).view(1, -1, 2).expand(n, -1, -1)
    dist = torch.cdist(flat, centres)               # n x px x pts
    two = dist.topk(2, dim=-1, largest=False)
    cell = colours.gather(1, two.indices[..., :1].expand(-1, -1, 3))
    edge = ((two.values[..., 1] - two.values[..., 0]) < 0.02).float().unsqueeze(-1)
    cell = cell * (1 - edge * _rand((n, 1, 1), gen, device, 0.0, 0.8))
    kinds.append(cell.permute(0, 2, 1).reshape(n, 3, size, size))
    # 2: stripes, checks or grid lines at a random angle and frequency
    ang = _rand((n, 1, 1), gen, device, 0, math.pi)
    freq = _rand((n, 1, 1), gen, device, 2, 20)
    u = xx.unsqueeze(0) * torch.cos(ang) + yy.unsqueeze(0) * torch.sin(ang)
    v = -xx.unsqueeze(0) * torch.sin(ang) + yy.unsqueeze(0) * torch.cos(ang)
    stripes = (torch.sin(u * freq * math.pi) > 0).float()
    checks = ((torch.floor(u * freq) + torch.floor(v * freq)) % 2).float()
    grid = ((torch.sin(u * freq * math.pi).abs() < 0.12) | (torch.sin(v * freq * math.pi).abs() < 0.12)).float()
    choice = torch.randint(0, 3, (n, 1, 1), device=device, generator=gen)
    pat = torch.where(choice == 0, stripes, torch.where(choice == 1, checks, grid)).unsqueeze(1)
    k2 = pal[:, 0].view(n, 3, 1, 1) * (1 - pat) + pal[:, 1].view(n, 3, 1, 1) * pat
    kinds.append(k2)
    # 3: a gradient (linear or radial) with a little noise
    ang = _rand((n, 1, 1), gen, device, 0, 2 * math.pi)
    lin = (xx.unsqueeze(0) * torch.cos(ang) + yy.unsqueeze(0) * torch.sin(ang)) * 0.5 + 0.5
    rad = (xx.unsqueeze(0) ** 2 + yy.unsqueeze(0) ** 2).sqrt().clamp(0, 1)
    radial = torch.randint(0, 2, (n, 1, 1), device=device, generator=gen).float()
    g = (lin * (1 - radial) + rad * radial).unsqueeze(1)
    k3 = pal[:, 0].view(n, 3, 1, 1) * (1 - g) + pal[:, 2].view(n, 3, 1, 1) * g
    k3 = k3 + 0.04 * (f2 - 0.5)
    kinds.append(k3)
    # 4: a flat colour with high-contrast rectangles and discs (sprites, text-like marks)
    base = pal[:, 0].view(n, 3, 1, 1).expand(n, 3, size, size).clone()
    textured = (torch.rand(n, 1, 1, 1, device=device, generator=gen) < 0.6)
    base = torch.where(textured, 0.5 * base + 0.5 * k0, base)
    for _ in range(10):
        cx = _rand((n, 1, 1), gen, device, -0.9, 0.9)
        cy = _rand((n, 1, 1), gen, device, -0.9, 0.9)
        hx = _rand((n, 1, 1), gen, device, 0.04, 0.4)
        hy = _rand((n, 1, 1), gen, device, 0.04, 0.4)
        disc = torch.rand(n, 1, 1, device=device, generator=gen) < 0.4
        rect = ((xx - cx).abs() < hx) & ((yy - cy).abs() < hy)
        ell = ((xx - cx) / hx) ** 2 + ((yy - cy) / hy) ** 2 < 1
        mask = torch.where(disc, ell, rect).unsqueeze(1)
        col = torch.rand(n, 3, 1, 1, device=device, generator=gen)
        base = torch.where(mask, col.expand_as(base), base)
    kinds.append(base)
    weights = torch.tensor([0.3, 0.15, 0.2, 0.1, 0.25], device=device)   # fbm, cells, patterns, gradient, sprites
    choice = torch.multinomial(weights.expand(n, -1), 1, generator=gen).squeeze(1)
    stack = torch.stack(kinds, 0)                    # kinds x n x 3 x s x s
    out = stack[choice, torch.arange(n, device=device)]
    grain = (torch.rand(n, 1, size, size, device=device, generator=gen) - 0.5) * _rand((n, 1, 1, 1), gen, device, 0.0, 0.16)
    out = out + grain
    contrast = _rand((n, 1, 1, 1), gen, device, 0.7, 1.3)
    return ((out - 0.5) * contrast + 0.5 + _rand((n, 1, 1, 1), gen, device, -0.12, 0.12)).clamp(0, 1)


def make_masks(n, gen, device, size=TEX, hard_bg=None):
    """n x 1 x size x size alpha masks of six shapes (the first, full, is the
    background's)."""
    xx, yy = _grid(size, device)
    xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)
    softness = _rand((n, 1, 1), gen, device, 0.01, 0.2)
    hard = torch.rand(n, 1, 1, device=device, generator=gen) < 0.55
    soft = torch.where(hard, torch.full_like(softness, 0.012), softness)
    full = torch.ones(n, 1, size, size, device=device)
    ax = _rand((n, 1, 1), gen, device, 0.35, 1.0)
    ay = _rand((n, 1, 1), gen, device, 0.35, 1.0)
    ell = (1 - ((xx / ax) ** 2 + (yy / ay) ** 2).sqrt()) / soft
    rr = torch.maximum((xx.abs() - ax * 0.9) / 0.1, (yy.abs() - ay * 0.9) / 0.1)
    rect = (1 - rr.clamp(-1, 1) * 0.5 - 0.5) * 0 + ((ax - xx.abs()).clamp(min=0).minimum((ay - yy.abs()).clamp(min=0)) / soft)
    blob = (_fbm(n, size, gen, device, octaves=4, base=3).squeeze(1) - _rand((n, 1, 1), gen, device, 0.35, 0.6)) \
        * 6 * (1 - (xx ** 2 + yy ** 2).sqrt().clamp(0, 1)) / soft.clamp(min=0.05)
    thin = _rand((n, 1, 1), gen, device, 0.015, 0.06)
    ang = _rand((n, 1, 1), gen, device, 0, math.pi)
    line = (thin - (xx * torch.sin(ang) - yy * torch.cos(ang)).abs()) / (soft * 0.4 + 0.004)
    line = torch.minimum(line, (1 - (xx * torch.cos(ang) + yy * torch.sin(ang)).abs()) / 0.02)
    dots = torch.zeros(n, size, size, device=device)
    for _ in range(14):
        cx = _rand((n, 1, 1), gen, device, -0.9, 0.9)
        cy = _rand((n, 1, 1), gen, device, -0.9, 0.9)
        r = _rand((n, 1, 1), gen, device, 0.04, 0.12)
        dots = torch.maximum(dots, ((r - ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()) / 0.02).clamp(0, 1))
    shapes = [ell.clamp(0, 1), rect.clamp(0, 1), blob.clamp(0, 1), line.clamp(0, 1), dots.clamp(0, 1)]
    choice = torch.randint(0, len(shapes), (n,), device=device, generator=gen)
    stack = torch.stack(shapes, 0)
    return stack[choice, torch.arange(n, device=device)].unsqueeze(1), full


def bayer4(h, w, device):
    m = torch.tensor([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=torch.float32, device=device)
    return (m / 16.0 - 0.5).repeat((h + 3) // 4, (w + 3) // 4)[:h, :w]


def _blend(out, top, uvm, samp, grid, rgb_gain, a_gain, additive, layer):
    """Composites one layer's premultiplied RGBA sample onto the frame: over, or added. Also keeps, for the
    ground-truth flow, the layer that is visible at each pixel (opaque enough, not added) and its texture
    coordinates there."""
    b = out.shape[0]
    rgb = samp[:, :3] * rgb_gain.view(b, 1, 1, 1)
    a = samp[:, 3:4] * a_gain.view(b, 1, 1, 1)
    add = additive.view(b, 1, 1, 1)
    visible = (a > 0.5) & ~add
    top = torch.where(visible[:, 0], torch.full_like(top, layer), top)
    uvm = torch.where(visible, grid.permute(0, 3, 1, 2), uvm)
    return torch.where(add, out + rgb, out * (1 - a) + rgb), top, uvm


def _fast(p, synth, gen):
    """Large motion: the layers and the camera move two to three times faster."""
    p['v'] = p['v'] * 2.5
    p['pan'] = p['pan'] * 2.5


def _slow(p, synth, gen):
    """Small motion: a few tenths of a pixel a step."""
    p['v'] = p['v'] * 0.1
    p['pan'] = p['pan'] * 0.1


def _hud(p, synth, gen):
    """Both HUD layers on, over a world that moves."""
    p['active'][:, LAYERS - HUDS:] = True
    p['pan'] = p['pan'] + 2.0 / synth.h * 3.0


def _transparent(p, synth, gen):
    """Every foreground layer translucent, some added (glass, smoke, glow)."""
    p['opacity'][:, 1:LAYERS - HUDS] = _rand((p['v'].shape[0], LAYERS - HUDS - 1), gen, synth.device, 0.3, 0.7)
    p['additive'][:, 1:LAYERS - HUDS] = torch.rand(p['v'].shape[0], LAYERS - HUDS - 1, device=synth.device, generator=gen) < 0.4
    p['active'][:, 1:4] = True


def _rotation(p, synth, gen):
    """Spinning layers and a rolling camera."""
    p['omega'] = p['omega'] + torch.randn_like(p['omega']) * 0.05
    p['roll'] = p['roll'] + torch.randn_like(p['roll']) * 0.02


def _zoom(p, synth, gen):
    """A dolly: the camera zooms and the layers change scale."""
    p['zoom'] = p['zoom'] + torch.randn_like(p['zoom']) * 0.03
    p['ds'] = p['ds'] + torch.randn_like(p['ds']) * 0.03


def _perspective(p, synth, gen):
    """Tilted planes (floors, walls) that move through the view."""
    p['persp0'] = p['persp0'] + torch.randn_like(p['persp0']) * 0.35
    p['dpersp'] = p['dpersp'] + torch.randn_like(p['dpersp']) * 0.02


def _occlusion(p, synth, gen):
    """Many opaque layers crossing one another at speed."""
    p['active'][:, 1:LAYERS - HUDS] = True
    p['opacity'][:, 1:LAYERS - HUDS] = 1.0
    p['additive'][:, 1:LAYERS - HUDS] = False
    p['v'][:, 1:LAYERS - HUDS] = p['v'][:, 1:LAYERS - HUDS] * 2.0


def _still(p, synth, gen):
    """Almost nothing moves: only the camera drifts by a pixel or less."""
    p['v'] = p['v'] * 0.0
    p['omega'] = p['omega'] * 0.0
    p['ds'] = p['ds'] * 0.0
    p['pan'] = p['pan'] * 0.1


def _pixel(p, synth, gen):
    """A retro console's picture: layers and a rigid camera that move whole source pixels a step (2, 3 or 4
    render pixels each), no rotation, zoom or perspective; the clip is then point-sampled at that pixel size
    (retro) so that edges are hard and motion lands on the grid. Own scenes only: no game is used."""
    b, L = p['v'].shape[0], LAYERS
    dev = synth.device
    f = torch.randint(2, 5, (b,), device=dev, generator=gen)
    q = (f.float() * 2.0 / synth.h)                                   # one source pixel in world units
    n = torch.randint(-3, 4, (b, L, 2), device=dev, generator=gen).float()
    n = n * (torch.rand(b, L, 1, device=dev, generator=gen) < 0.7).float()
    p['v'] = n * q.view(b, 1, 1)
    p['pan'] = torch.randint(-3, 4, (b, 2), device=dev, generator=gen).float() * q.view(b, 1)
    p['parallax'] = torch.ones_like(p['parallax'])
    p['c0'] = torch.round(p['c0'] / q.view(b, 1, 1)) * q.view(b, 1, 1)
    for k in ('acc', 'omega', 'ds', 'persp0', 'dpersp', 'scroll', 'zoom', 'roll', 'fade', 'flicker'):
        p[k] = p[k] * 0.0
    p['pix'] = f


PRESETS = {'pixel': _pixel, 'fast': _fast, 'slow': _slow, 'hud': _hud, 'transparent': _transparent, 'rotation': _rotation, 'zoom': _zoom,
           'perspective': _perspective, 'occlusion': _occlusion, 'still': _still}


class Clips:
    """What Synth.render returns: frames, the visible layer and texture coordinates (for the ground-truth flow)."""

    def __init__(self, frames, top, uv, p):
        self.frames, self.top, self.uv, self.p = frames, top, uv, p


class Synth:
    """Renders clips on the GPU: render(batch, gen) -> frames B x STEPS x 3 x H x W in [0, 1]."""

    def __init__(self, device='cuda', height=288, width=512, bank=3072):
        self.device = device
        self.h, self.w = height, width
        self.aspect = width / height
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * 2 - 1
        xs = ((torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * 2 - 1) * self.aspect
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.xx, self.yy = xx, yy
        self.dither = bayer4(height, width, device)
        # A bank of textures and masks, a part of it renewed on every render, so a batch
        # draws its layers from thousands of them at the price of dozens
        self.bank_size = bank
        self.tex_bank = None
        self.mask_bank = None
        self.cursor = 0
        # fused kernels for the per-layer arithmetic, the generator's hot spot
        self._coords_fast = torch.compile(self._coords)
        self._blend_fast = torch.compile(_blend)

    def _refresh_bank(self, gen, count):
        dev = self.device
        if self.tex_bank is None:
            self.tex_bank = torch.empty(self.bank_size, 3, TEX, TEX, device=dev, dtype=torch.float16)
            self.mask_bank = torch.empty(self.bank_size, 1, TEX, TEX, device=dev, dtype=torch.float16)
            count = self.bank_size
        for start in range(0, count, 64):
            n = min(64, count - start)
            tex = make_textures(n, gen, dev)
            mask, _ = make_masks(n, gen, dev)
            idx = (self.cursor + torch.arange(n, device=dev)) % self.bank_size
            self.tex_bank[idx] = tex.half()
            self.mask_bank[idx] = mask.half()
            self.cursor = (self.cursor + n) % self.bank_size

    def _draw_layers(self, n, gen):
        """n textures and masks from the bank, with a random flip, transpose and colour permutation each."""
        dev = self.device
        idx = torch.randint(0, self.bank_size, (n,), device=dev, generator=gen)
        tex = self.tex_bank[idx].float()
        mask = self.mask_bank[idx].float()
        flip = torch.rand(n, device=dev, generator=gen) < 0.5
        tex = torch.where(flip.view(n, 1, 1, 1), tex.flip(-1), tex)
        mask = torch.where(flip.view(n, 1, 1, 1), mask.flip(-1), mask)
        tr = torch.rand(n, device=dev, generator=gen) < 0.5
        tex = torch.where(tr.view(n, 1, 1, 1), tex.transpose(-1, -2), tex)
        mask = torch.where(tr.view(n, 1, 1, 1), mask.transpose(-1, -2), mask)
        perm = torch.argsort(torch.rand(n, 3, device=dev, generator=gen), dim=1)
        tex = tex.gather(1, perm.view(n, 3, 1, 1).expand(-1, -1, TEX, TEX))
        inv = torch.rand(n, 1, 1, 1, device=dev, generator=gen) < 0.15
        tex = torch.where(inv, 1 - tex, tex)
        return tex, mask

    def _params(self, b, gen):
        L, dev = LAYERS, self.device
        px = 2.0 / self.h  # world units per pixel

        def speed(shape, lo=0.1, hi=10.0):
            mag = torch.exp(_rand(shape, gen, dev, math.log(lo), math.log(hi)))
            fast = (torch.rand(shape, device=dev, generator=gen) < 0.15).float()
            return mag * (1 + fast * _rand(shape, gen, dev, 0.0, 1.5))

        p = {}
        ang = _rand((b, L), gen, dev, 0, 2 * math.pi)
        mag = speed((b, L)) * px
        still = (torch.rand(b, L, device=dev, generator=gen) < 0.15).float()      # held objects
        mag = mag * (1 - still)
        p['v'] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], -1)     # b x L x 2, per step
        p['acc'] = torch.randn(b, L, 2, device=dev, generator=gen) * 0.12 * mag.unsqueeze(-1)
        p['c0'] = torch.stack([_rand((b, L), gen, dev, -0.9, 0.9) * self.aspect, _rand((b, L), gen, dev, -0.8, 0.8)], -1)
        p['theta0'] = _rand((b, L), gen, dev, -math.pi, math.pi)
        p['omega'] = torch.randn(b, L, device=dev, generator=gen) * 0.006 * (torch.rand(b, L, device=dev, generator=gen) < 0.6)
        p['s0'] = torch.exp(_rand((b, L), gen, dev, math.log(0.12), math.log(0.8)))
        p['ds'] = torch.randn(b, L, device=dev, generator=gen) * 0.004
        p['persp0'] = torch.randn(b, L, 2, device=dev, generator=gen) * 0.1 * (torch.rand(b, L, 1, device=dev, generator=gen) < 0.4)
        p['dpersp'] = torch.randn(b, L, 2, device=dev, generator=gen) * 0.008
        p['scroll'] = torch.randn(b, L, 2, device=dev, generator=gen) * 0.004 * (torch.rand(b, L, 1, device=dev, generator=gen) < 0.2)
        p['aspect_ratio'] = torch.exp(_rand((b, L), gen, dev, math.log(0.5), math.log(2.0)))
        p['opacity'] = torch.where(torch.rand(b, L, device=dev, generator=gen) < 0.3, _rand((b, L), gen, dev, 0.3, 0.85),
                                   torch.ones(b, L, device=dev))
        p['fade'] = torch.randn(b, L, device=dev, generator=gen) * 0.04 * (torch.rand(b, L, device=dev, generator=gen) < 0.2)
        p['flicker'] = (torch.rand(b, L, device=dev, generator=gen) < 0.1).float() * _rand((b, L), gen, dev, 0.0, 0.3)
        p['phase'] = _rand((b, L), gen, dev, 0, 6.28)
        p['additive'] = torch.rand(b, L, device=dev, generator=gen) < 0.12
        p['parallax'] = (torch.linspace(0.3, 1.2, L, device=dev).unsqueeze(0)
                         * _rand((b, L), gen, dev, 0.8, 1.2))
        # the camera: pan, zoom and roll shared by the world layers
        cang = _rand((b,), gen, dev, 0, 2 * math.pi)
        cmag = speed((b,), 0.05, 8.0) * px * (torch.rand(b, device=dev, generator=gen) < 0.8)
        p['pan'] = torch.stack([cmag * torch.cos(cang), cmag * torch.sin(cang)], -1)
        p['zoom'] = torch.randn(b, device=dev, generator=gen) * 0.004 * (torch.rand(b, device=dev, generator=gen) < 0.4)
        p['roll'] = torch.randn(b, device=dev, generator=gen) * 0.003 * (torch.rand(b, device=dev, generator=gen) < 0.3)
        # active layers: the background always; the rest with a random count
        count = torch.randint(1, L - HUDS, (b, 1), device=dev, generator=gen)
        idx = torch.arange(L, device=dev).unsqueeze(0)
        active = (idx <= count) | (idx == 0)
        hud_on = torch.rand(b, HUDS, device=dev, generator=gen) < 0.5
        active[:, L - HUDS:] = hud_on
        p['active'] = active
        # the HUD: fixed, in a screen corner or edge, small
        for h in range(HUDS):
            l = L - HUDS + h
            p['v'][:, l] = 0
            p['acc'][:, l] = 0
            p['omega'][:, l] = 0
            p['ds'][:, l] = 0
            p['persp0'][:, l] = 0
            p['dpersp'][:, l] = 0
            p['scroll'][:, l] = 0
            p['opacity'][:, l] = 1
            p['additive'][:, l] = False
            p['theta0'][:, l] = 0
            p['parallax'][:, l] = 0
            p['aspect_ratio'][:, l] = 1
            side = torch.randint(0, 4, (b,), device=dev, generator=gen)
            ex = torch.where(side % 2 == 0, -0.78, 0.78) * self.aspect
            ey = torch.where(side // 2 == 0, -0.88, 0.88)
            p['c0'][:, l] = torch.stack([ex * _rand((b,), gen, dev, 0.5, 1.0), ey], -1)
            p['s0'][:, l] = _rand((b,), gen, dev, 0.1, 0.25)
        p['opacity'][:, 0] = 1                                       # the background is opaque, and not added
        p['additive'][:, 0] = False
        p['hud_zoom'] = torch.zeros(b, device=dev)
        p['bg_s'] = _rand((b,), gen, dev, 1.2, 3.0)
        return p

    def _pose(self, p, tau):
        """Every layer's pose at a time: tau is b x 1 (steps). The camera pans the layers by their parallax, and
        zooms and rolls them about the screen's centre; the HUD layers keep still."""
        dev = self.device
        world = (torch.arange(LAYERS, device=dev) < LAYERS - HUDS)
        tau3 = tau.unsqueeze(-1)                                      # b x 1 x 1
        pan = p['pan'].unsqueeze(1) * p['parallax'].unsqueeze(-1) * tau3
        zoom = 1 + p['zoom'].unsqueeze(1) * tau                       # b x 1
        roll = p['roll'].unsqueeze(1) * tau
        c = p['c0'] + p['v'] * tau3 + 0.5 * p['acc'] * tau3 * tau3 - pan
        cr, sr = torch.cos(roll), torch.sin(roll)
        cam = torch.stack([zoom * (cr * c[..., 0] - sr * c[..., 1]), zoom * (sr * c[..., 0] + cr * c[..., 1])], -1)
        c = torch.where(world.view(1, LAYERS, 1), cam, c)
        theta = torch.where(world, p['theta0'] + p['omega'] * tau + roll, p['theta0'])
        s = torch.where(world, (p['s0'] * (1 + p['ds'] * tau)).clamp(min=0.03) * zoom, p['s0'])
        s = torch.cat([p['bg_s'].unsqueeze(1) * zoom, s[:, 1:]], 1)   # the background's scale
        ar = p['aspect_ratio']
        return {'c': c, 'theta': theta, 'sx': s * ar.sqrt(), 'sy': s / ar.sqrt(),
                'persp': p['persp0'] + p['dpersp'] * tau3, 'scroll': p['scroll'] * tau3}

    def _coords(self, pose, layer):
        """Texture coordinates u, v of one layer for every pixel, b x H x W x 2."""
        b = pose['theta'].shape[0]
        c, theta = pose['c'][:, layer], pose['theta'][:, layer]
        persp, scroll = pose['persp'][:, layer], pose['scroll'][:, layer]
        dx = self.xx.unsqueeze(0) - c[:, 0].view(b, 1, 1)
        dy = self.yy.unsqueeze(0) - c[:, 1].view(b, 1, 1)
        ct, st = torch.cos(theta).view(b, 1, 1), torch.sin(theta).view(b, 1, 1)
        rx = ct * dx + st * dy
        ry = -st * dx + ct * dy
        w = (1 + persp[:, 0].view(b, 1, 1) * rx + persp[:, 1].view(b, 1, 1) * ry).clamp(min=0.25)
        u = rx / pose['sx'][:, layer].view(b, 1, 1) * w + scroll[:, 0].view(b, 1, 1)
        v = ry / pose['sy'][:, layer].view(b, 1, 1) * w + scroll[:, 1].view(b, 1, 1)
        return torch.stack([u, v], -1)

    def _screen(self, pose, ids, uv):
        """Where each pixel's surface point (layer ids, texture coordinates uv) is on screen under another
        pose, in world units: the inverse of _coords (w solves w^2 - w = p.q)."""
        def g(x):
            return x.gather(1, ids.flatten(1)).view_as(ids)
        qx = (uv[:, 0] - g(pose['scroll'][..., 0])) * g(pose['sx'])
        qy = (uv[:, 1] - g(pose['scroll'][..., 1])) * g(pose['sy'])
        pq = g(pose['persp'][..., 0]) * qx + g(pose['persp'][..., 1]) * qy
        w = (1 + torch.sqrt((1 + 4 * pq).clamp(min=0.0))) / 2
        rx, ry = qx / w, qy / w
        th = g(pose['theta'])
        ct, st = torch.cos(th), torch.sin(th)
        return g(pose['c'][..., 0]) + ct * rx - st * ry, g(pose['c'][..., 1]) + st * rx + ct * ry

    @torch.no_grad()
    def flow_gt(self, clips, ks, ms, ss):
        """Exact flow from the frame at step ms[i] of clip ks[i] to step ss[i] of the same clip, in pixels: the
        displacement that takes a pixel to where its surface point is in the other frame (the sampling flow,
        out(x) = img(x + flow)). Returns N x 2 x H x W."""
        pk = {key: val[ks] for key, val in clips.p.items()}
        pose = self._pose(pk, ss.view(-1, 1).float())
        ids = clips.top[ks, ms].long()
        uv = clips.uv[ks, ms].float()
        x, y = self._screen(pose, ids, uv)
        unit = 2.0 / self.h
        return torch.stack([(x - self.xx) / unit, (y - self.yy) / unit], 1)

    @torch.no_grad()
    def render(self, b, gen, preset=None):
        """A Clips object: frames b x STEPS x 3 x H x W, and for the ground-truth flow the visible layer and its
        texture coordinates at every pixel of every step."""
        dev = self.device
        p = self._params(b, gen)
        if preset:
            PRESETS[preset](p, self, gen)
        n = b * LAYERS
        self._refresh_bank(gen, 48)
        tex, mask = self._draw_layers(n, gen)
        tex = tex.view(b, LAYERS, 3, TEX, TEX)
        mask = mask.view(b, LAYERS, 1, TEX, TEX)
        mask[:, 0] = 1.0                                             # the background fills its quad
        tex_a = torch.cat([tex * mask, mask], 2)                     # premultiplied RGBA
        frames, tops, uvs = [], [], []
        for tau in range(STEPS):
            out = torch.zeros(b, 3, self.h, self.w, device=dev)
            top = torch.zeros(b, self.h, self.w, dtype=torch.uint8, device=dev)
            uvm = torch.zeros(b, 2, self.h, self.w, dtype=torch.float32, device=dev)
            tau_t = torch.full((b, 1), float(tau), device=dev)
            pose = self._pose(p, tau_t)
            for l in range(LAYERS):
                grid = self._coords_fast(pose, l)
                samp = F.grid_sample(tex_a[:, l], grid, mode='nearest' if 'pix' in p else 'bilinear', padding_mode='reflection' if l == 0 else 'zeros',
                                     align_corners=False)
                k = (p['opacity'][:, l] * (1 + p['fade'][:, l] * tau_t[:, 0])).clamp(0, 1) * p['active'][:, l].float()
                bright = 1 + p['flicker'][:, l] * torch.sin(p['phase'][:, l] + 0.5 * tau_t[:, 0])
                out, top, uvm = self._blend_fast(out, top, uvm, samp, grid, bright * k, k, p['additive'][:, l], l)
            frames.append(out.clamp(0, 1))
            tops.append(top)
            uvs.append(uvm)
        x = torch.stack(frames, 1)                                    # b x STEPS x 3 x H x W
        return Clips(self.retro(x, gen, p.get('pix')), torch.stack(tops, 1), torch.stack(uvs, 1), p)

    def retro(self, x, gen, pix_factor=None):
        """The retro look, one setting per clip: pixelation, colour depth, dither, a global fade."""
        b, T, C, H, W = x.shape
        dev = self.device
        pix = torch.randint(0, 6, (b,), device=dev, generator=gen)  # 0..5: 1,1,2,3,4 and 2
        factors = torch.tensor([1, 1, 2, 3, 4, 2], device=dev)[pix]
        point = pix_factor is not None                                # a retro-console clip: point-sampled blocks
        if point:
            factors = pix_factor
        out = x.clone()
        for f in (2, 3, 4):
            sel = (factors == f).nonzero().flatten()
            if sel.numel() == 0:
                continue
            flat = x[sel].reshape(-1, C, H, W)
            if point:
                small = flat[:, :, f // 2::f, f // 2::f]
                up = small.repeat_interleave(f, 2).repeat_interleave(f, 3)[:, :, :H, :W]
            else:
                small = F.adaptive_avg_pool2d(flat, (math.ceil(H / f), math.ceil(W / f)))
                up = F.interpolate(small, size=(H, W), mode='nearest')
            out[sel] = up.view(sel.numel(), T, C, H, W)
        bits = torch.where(torch.rand(b, device=dev, generator=gen) < 0.45, 5, 8)
        dith = (torch.rand(b, device=dev, generator=gen) < 0.5).float()
        levels = (2 ** bits - 1).float().view(b, 1, 1, 1, 1)
        noise = self.dither.view(1, 1, 1, H, W) * dith.view(b, 1, 1, 1, 1) / levels
        out = torch.round((out + noise).clamp(0, 1) * levels) / levels
        # a fade or flash across the clip on a few
        mode = torch.randint(0, 12, (b,), device=dev, generator=gen)
        t = torch.arange(T, device=dev).float().view(1, T, 1, 1, 1) / (T - 1)
        amt = _rand((b, 1, 1, 1, 1), gen, dev, -0.5, 0.5)
        fade = (mode == 0).float().view(b, 1, 1, 1, 1)
        out = out * (1 + fade * amt * t)
        return out.clamp(0, 1)
