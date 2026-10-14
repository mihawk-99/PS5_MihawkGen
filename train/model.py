# PS5_FrameGen - the interpolation network (docs/plan.md, phase 5).
#
# From two frames and a time t it predicts, coarse to fine, the flow from the
# frame at t to each of them and a blend mask, then warps and blends the
# frames at full resolution. Three blocks work at 1/8, 1/4 and 1/2 of the
# frames' size; each takes the frames, the flow so far and the frames warped by
# it, and adds a refinement. This is the structure of coarse-to-fine
# intermediate-flow estimation published as RIFE (Huang et al., ECCV 2022, MIT)
# and IFRNet (Kong et al., CVPR 2022, Apache-2.0): written here from the papers'
# descriptions, smaller, and with only the operations the console's compute
# shaders have (docs/design.md): 3x3 convolutions with a leaky ReLU, area
# pooling, bilinear resizing, backward warping, and a blend.
#
# Flow is in pixels of the input frames: the colour of the output at x is the
# frame's colour at x + flow(x). Blocks predict it in their own pixels, which the
# scale turns into the frames'.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# (scale, width, convolutions, matching radius): the configuration the shaders implement
DEFAULT_BLOCKS = ((16, 64, 5, 3), (8, 64, 4, 2), (4, 48, 4, 2), (2, 32, 3, 1))


def warp(img, flow):
    """Backward warp: out(x) = img(x + flow(x)), flow in pixels, border-clamped."""
    b, _, h, w = img.shape
    ys = torch.arange(h, device=img.device, dtype=img.dtype).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, w).expand(b, h, w)
    gx = (xs + flow[:, 0] + 0.5) / w * 2 - 1
    gy = (ys + flow[:, 1] + 0.5) / h * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode='bilinear', padding_mode='border', align_corners=False)


SHARP_WARP = os.environ.get('FG_SHARP_WARP', '1') == '1'


def warp_sharp(img, flow):
    """The last pass's read of a frame (nn_final.comp, sampleSharp): the same backward warp read with a Catmull-Rom kernel
    (four texels in each direction, border-clamped) and kept within the range of the four texels nearest the point, so that it
    does not ring at an edge. A bilinear read at a fraction of a pixel low-passes the frame; this keeps more of it."""
    b, c, h, w = img.shape
    ys = torch.arange(h, device=img.device, dtype=img.dtype).view(1, h, 1).expand(b, h, w)
    xs = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, w).expand(b, h, w)
    px, py = xs + flow[:, 0] + 0.5, ys + flow[:, 1] + 0.5
    ix, iy = torch.floor(px - 0.5), torch.floor(py - 0.5)
    fx, fy = px - (ix + 0.5), py - (iy + 0.5)

    def weights(f):
        return (f * (-0.5 + f * (1.0 - 0.5 * f)), 1.0 + f * f * (-2.5 + 1.5 * f), f * (0.5 + f * (2.0 - 1.5 * f)), f * f * (-0.5 + 0.5 * f))
    wx, wy = weights(fx), weights(fy)
    flat = img.flatten(2)

    def texel(dx, dy):
        x = (ix + dx).clamp(0, w - 1).long()
        y = (iy + dy).clamp(0, h - 1).long()
        return flat.gather(2, (y * w + x).view(b, 1, -1).expand(b, c, -1)).view(b, c, h, w)
    out = 0
    near = []
    for j in range(4):
        for i in range(4):
            t = texel(i - 1, j - 1)
            out = out + t * (wx[i] * wy[j]).unsqueeze(1)
            if i in (1, 2) and j in (1, 2):
                near.append(t)
    lo, hi = torch.stack(near).amin(0), torch.stack(near).amax(0)
    return torch.minimum(torch.maximum(out, lo), hi)


def area(x, scale):
    return x if scale == 1 else F.avg_pool2d(x, scale)


def resize(x, scale):
    """Bilinear upsampling by an integer scale."""
    return x if scale == 1 else F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)


def cost_volume(a0, a1, t, flow, radius):
    """Matching costs around the current flow: for each candidate displacement d (integer, in the block's pixels,
    within the radius) the mean absolute difference between frame 0 at x + flow0 - t d and frame 1 at
    x + flow1 + (1 - t) d, the positions a point moving by d between the frames has at time t. Channel
    (dy + R) (2R + 1) + (dx + R). Small where the candidate fits."""
    b = a0.shape[0]
    tt = t.view(b, 1, 1, 1).to(a0.dtype)
    costs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = torch.tensor([dx, dy], device=a0.device, dtype=a0.dtype).view(1, 2, 1, 1)
            f0 = flow[:, :2] - tt * d
            f1 = flow[:, 2:] + (1 - tt) * d
            costs.append((warp(a0, f0) - warp(a1, f1)).abs().mean(1, keepdim=True))
    return torch.cat(costs, 1)


def cost_volume_prev(ap, a0, radius):
    """Matching costs of the pair before: for each candidate displacement d within the radius, the mean absolute difference between
    frame 0 at x and the frame before it at x - d (what moved by d between them). Channel (dy + R) (2R + 1) + (dx + R)."""
    costs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d = torch.tensor([-dx, -dy], device=a0.device, dtype=a0.dtype).view(1, 2, 1, 1).expand(a0.shape[0], 2, a0.shape[2], a0.shape[3])
            costs.append((warp(ap, d) - a0).abs().mean(1, keepdim=True))
    return torch.cat(costs, 1)


class Block(nn.Module):
    def __init__(self, in_ch, width, convs, out_ch=5):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        for _ in range(convs - 1):
            layers += [nn.Conv2d(width, width, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(width, out_ch, 3, 1, 1)]   # flow to frame 0 (2), flow to frame 1 (2), mask logit (and a confidence logit)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


import os
GATE = tuple(float(x) for x in os.environ.get('FG_GATE', '2,12').split(','))   # the levels where the picture starts to leave the blend and where it is the network's


STATIC_STRUCTURE = os.environ.get('FG_STATIC_STRUCTURE', '1') == '1'


def _smoothstep(lo, hi, x):
    u = ((x - lo) / (hi - lo)).clamp(0, 1)
    return u * u * (3 - 2 * u)


def static_structure(img0, img1):
    """What the last pass (nn_final.comp) takes as a thing that fades or flashes in place rather than moves: the normalised
    correlation of the two frames' brightness over nine points two pixels apart is high (0.85 to 0.97) where both have
    strong structure (a standard deviation of 8 to 20 levels). There the blend is exact; the network's warp would double it."""
    w = img0.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    la, lc = (img0 * w).sum(1, keepdim=True), (img1 * w).sum(1, keepdim=True)
    h, wd = la.shape[-2:]
    pa, pc = F.pad(la, (2, 2, 2, 2), mode='replicate'), F.pad(lc, (2, 2, 2, 2), mode='replicate')
    sa = [pa[:, :, j:j + h, i:i + wd] for j in (0, 2, 4) for i in (0, 2, 4)]
    sc = [pc[:, :, j:j + h, i:i + wd] for j in (0, 2, 4) for i in (0, 2, 4)]
    ma, mc = sum(sa) / 9, sum(sc) / 9
    saa = sum((x - ma) ** 2 for x in sa)
    scc = sum((x - mc) ** 2 for x in sc)
    sac = sum((x - ma) * (y - mc) for x, y in zip(sa, sc))
    structure = _smoothstep(8 / 255, 20 / 255, (torch.minimum(saa, scc) / 9).sqrt())
    ncc = sac * torch.rsqrt((saa * scc).clamp_min(1e-8))
    return _smoothstep(0.85, 0.97, ncc) * structure


def gate(img0, img1, moved, t):
    """What the generator's last pass does (framegen/shaders/nn_final.comp): where the two frames agree within a few levels (the
    largest difference over a 3x3 grid of points two pixels apart) the picture is their blend, exact; the network's picture is
    taken in proportion as they differ (smoothstep from 2 to 12 levels)."""
    d = (img0 - img1).abs().amax(1, keepdim=True)
    pad = F.pad(d, (2, 2, 2, 2), mode='replicate')
    h, w = d.shape[-2:]
    diff = torch.zeros_like(d)
    for j in (0, 2, 4):
        for i in (0, 2, 4):
            diff = torch.maximum(diff, pad[:, :, j:j + h, i:i + w])
    u = ((diff - GATE[0] / 255) / ((GATE[1] - GATE[0]) / 255)).clamp(0, 1)
    motion = u * u * (3 - 2 * u)
    if STATIC_STRUCTURE:
        motion = motion * (1 - static_structure(img0, img1))
    tt = t.view(-1, 1, 1, 1).to(img0.dtype)
    still = img0 * (1 - tt) + img1 * tt
    return still + (moved - still) * motion


GATE_MODE = int(os.environ.get('FG_GATE2', '0'))      # 0: the shipped last pass (gate, structure test); 4: by agreement and motion (gate4)
GATE2 = GATE_MODE > 0     # the last pass decides by evidence (see gate2) and reads only the frame that has the picture at the border
GATE2_PARAMS = tuple(float(x) for x in os.environ.get('FG_GATE2_PARAMS', '2,12,5').split(','))     # levels where the evidence starts and where it is full; the window


def border_valid(flow, margin=0.0, ramp=1.0):
    """1 where the point a flow reads from lies inside the frame (a ramp of `ramp` pixels at its border), 0 outside: a content that enters
    at the border is in one of the two frames only."""
    h, w = flow.shape[-2:]
    ys = torch.arange(h, device=flow.device, dtype=flow.dtype).view(1, h, 1) + 0.5
    xs = torch.arange(w, device=flow.device, dtype=flow.dtype).view(1, 1, w) + 0.5
    px, py = xs + flow[:, 0], ys + flow[:, 1]
    d = torch.minimum(torch.minimum(px, w - px), torch.minimum(py, h - py))
    return ((d - margin) / ramp).clamp(0, 1)[:, None]


def gate2(img0, img1, moved, t, w0, w1):
    """The last pass by evidence. Where the frames compensated for the motion (w0, w1) agree better than the raw frames, the picture is the
    motion-compensated one; where they do not (nothing moves, the picture only fades or flickers in place, or the flow is wrong) it is the
    blend. A window mean of the absolute differences over the colour channels (levels GATE2_PARAMS[0..1], window GATE2_PARAMS[2])."""
    lo, hi, win = GATE2_PARAMS
    win = int(win)
    d_still = (img0 - img1).abs().mean(1, keepdim=True)
    d_moved = (w0 - w1).abs().mean(1, keepdim=True)
    es, em = F.avg_pool2d(d_still, win, 1, win // 2, count_include_pad=False), F.avg_pool2d(d_moved, win, 1, win // 2, count_include_pad=False)
    u = ((es - em - lo / 255) / ((hi - lo) / 255)).clamp(0, 1)
    motion = u * u * (3 - 2 * u)
    tt = t.view(-1, 1, 1, 1).to(img0.dtype)
    still = img0 * (1 - tt) + img1 * tt
    return still + (moved - still) * motion


GATE3_PARAMS = tuple(float(x) for x in os.environ.get('FG_GATE3_PARAMS', '1,3,0.15,0.6,3').split(','))     # levels of the frames' agreement (blend below the first), flow in pixels (blend below the first), window


def gate3(img0, img1, moved, t, flow):
    """The last pass by motion. A pixel is the network's picture in proportion as (1) the two frames differ there (smoothstep over a window
    mean of the absolute difference, from GATE3_PARAMS[0] to [1] levels: frames that agree to a level are the blend, exact) and (2) the
    network says it moves (flow length in pixels, from GATE3_PARAMS[2] to [3]: where nothing moves the blend is exact, and a picture that
    fades or flickers in place has no flow). Where it moves, the network's picture is kept whole, as precise as the flow."""
    lo, hi, f0, f1, win = GATE3_PARAMS
    win = int(win)
    d = (img0 - img1).abs().amax(1, keepdim=True)
    dm = F.avg_pool2d(d, win, 1, win // 2, count_include_pad=False)
    u = ((dm - lo / 255) / ((hi - lo) / 255)).clamp(0, 1)
    fl = torch.maximum(flow[:, :2].pow(2).sum(1, keepdim=True), flow[:, 2:].pow(2).sum(1, keepdim=True)).sqrt()
    v = ((fl - f0) / (f1 - f0)).clamp(0, 1)
    motion = u * u * (3 - 2 * u) * v * v * (3 - 2 * v)
    tt = t.view(-1, 1, 1, 1).to(img0.dtype)
    still = img0 * (1 - tt) + img1 * tt
    return still + (moved - still) * motion


GATE4_STATIC = os.environ.get('FG_GATE4_STATIC', '1') == '1'
BIG_NEEDS_DIFF = int(os.environ.get('FG_BIG_NEEDS_DIFF', '2'))      # 0: a flow of 1 to 2 pixels is motion everywhere; 1: where the frames differ at the taps; 2: where there is anything to move
BIG_FLAT = tuple(float(x) for x in os.environ.get('FG_BIG_FLAT', '2,4').split(','))
GATE4_PARAMS = tuple(float(x) for x in os.environ.get('FG_GATE4_PARAMS', '2,12,0.2,0.8,1.0,2.0').split(','))   # diff levels lo/hi, flow (px) lo/hi of the weak gate, flow lo/hi of the strong override


def gate4(img0, img1, moved, t, flow):
    """The last pass by agreement and motion. The old gate (frames that agree within a few levels are the blend, exact) now needs the network to
    agree that the pixel is still: it counts in proportion as the flow is short (below 0.2 to 0.8 pixels), and a flow of 1 to 2 pixels or more is
    motion whatever the difference (a repeating pattern that scrolls by its own period has the same pixels in both frames). Nothing that fades
    or flickers in place has a flow: the blend. No test of structure (a normalised correlation) is needed."""
    lo, hi, f0, f1, g0, g1 = GATE4_PARAMS
    d = (img0 - img1).abs().amax(1, keepdim=True)
    pad = F.pad(d, (2, 2, 2, 2), mode='replicate')
    h, w = d.shape[-2:]
    diff = torch.zeros_like(d)
    for j in (0, 2, 4):
        for i in (0, 2, 4):
            diff = torch.maximum(diff, pad[:, :, j:j + h, i:i + w])
    u = ((diff - lo / 255) / ((hi - lo) / 255)).clamp(0, 1)
    u = u * u * (3 - 2 * u)
    fl = torch.maximum(flow[:, :2].pow(2).sum(1, keepdim=True), flow[:, 2:].pow(2).sum(1, keepdim=True)).sqrt()
    v = ((fl - f0) / (f1 - f0)).clamp(0, 1)
    v = v * v * (3 - 2 * v)
    big = ((fl - g0) / (g1 - g0)).clamp(0, 1)
    big = big * big * (3 - 2 * big)
    if GATE4_STATIC:
        u = u * (1 - static_structure(img0, img1))      # a thing that fades in place: the blend (a flow of 1 to 2 pixels or more still counts as motion)
    if BIG_NEEDS_DIFF == 2:
        # a flat place (no frame has a difference of more than a couple of levels in the 3x3 around the pixel) has nothing to move
        rng = torch.zeros_like(u)
        for x in (img0, img1):
            rng = torch.maximum(rng, (F.max_pool2d(x, 3, 1, 1) - (-F.max_pool2d(-x, 3, 1, 1))).amax(1, keepdim=True))
        flat = ((rng - BIG_FLAT[0] / 255) / ((BIG_FLAT[1] - BIG_FLAT[0]) / 255)).clamp(0, 1)
        big = big * torch.where(u > 0, torch.ones_like(flat), flat)       # (where the frames differ at the taps there is something to move)
        motion = torch.maximum(u * v, big)
    else:
        motion = torch.maximum(u * v, big * ((u > 0).to(big.dtype) if BIG_NEEDS_DIFF == 1 else 1.0))      # (frames that agree to a couple of levels all around are the blend, exact, whatever the flow: no warp is read there)
    tt = t.view(-1, 1, 1, 1).to(img0.dtype)
    still = img0 * (1 - tt) + img1 * tt
    return still + (moved - still) * motion


class Refine(nn.Module):
    """A residual refinement on the warped frames, at 1/scale of the frames' size (research variant, not in the shaders)."""
    def __init__(self, width, convs, scale):
        super().__init__()
        self.scale = scale
        layers = [nn.Conv2d(3 * 4 + 5 + 1, width, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        for _ in range(convs - 1):
            layers += [nn.Conv2d(width, width, 3, 1, 1), nn.LeakyReLU(0.2, True)]
        last = nn.Conv2d(width, 3, 3, 1, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers += [last]
        self.net = nn.Sequential(*layers)

    def forward(self, img0, img1, w0, w1, flow, mask, t):
        s = self.scale
        x = torch.cat([area(img0, s), area(img1, s), area(w0, s), area(w1, s), area(flow, s) / s, area(mask, s),
                       t.view(-1, 1, 1, 1).expand(img0.shape[0], 1, img0.shape[2] // s, img0.shape[3] // s).to(img0.dtype)], 1)
        return resize(self.net(x), s)


class Interpolator(nn.Module):
    def __init__(self, blocks=DEFAULT_BLOCKS, conf=False, refine=None, gated=False, privileged=False, prev=False, mh=False):
        super().__init__()
        self.blocks_cfg = tuple(blocks)
        self.conf, self.gated, self.privileged, self.prev, self.mh = conf, gated, privileged, prev, mh   # privileged: also given the true middle frame (a teacher in training)
        mods = []
        for i, (scale, width, convs, radius) in enumerate(self.blocks_cfg):
            # frames (6) + time (1); later blocks also the warped frames (6), the flow (4) and the mask (1);
            # then the matching costs, one for each candidate displacement
            mods.append(Block((7 if i == 0 else 18) + (2 * radius + 1) ** 2 + (3 if privileged else 0) + ((3 + (2 * radius + 1) ** 2) if prev else 0), width, convs, (5 + (1 if conf else 0) + (6 if mh else 0)) if i == len(self.blocks_cfg) - 1 else 5))
        self.blocks = nn.ModuleList(mods)
        # refine: (width, convs, scale) of a residual refinement after the warp
        self.refine = Refine(*refine) if refine else None
        self.options = dict(conf=conf, refine=list(refine) if refine else None, gated=gated, prev=prev, mh=mh)

    def _prev_inputs(self, prev, a0, scale, radius):
        if not self.prev:
            return []
        ap = area(prev if prev is not None else a0.new_zeros(0), scale) if prev is not None else a0
        return [ap, cost_volume_prev(ap, a0, radius)]

    def flows(self, img0, img1, t, gt=None, prev=None):
        """The flow to each frame and the mask logit, at the frames' size. self.levels keeps the flow after each block, at the
        frames' size, for the teacher's distillation."""
        b, _, h, w = img0.shape
        flow_s = mask_s = None
        self.levels = []
        for i, (scale, _, _, radius) in enumerate(self.blocks_cfg):
            a0, a1 = area(img0, scale), area(img1, scale)
            tmap = t.view(-1, 1, 1, 1).expand(b, 1, a0.shape[2], a0.shape[3]).to(a0.dtype)
            if i == 0:
                flow_s = torch.zeros(b, 4, a0.shape[2], a0.shape[3], device=a0.device, dtype=a0.dtype)
                cost = cost_volume(a0, a1, t, flow_s, radius)
                extra = ([area(gt, scale)] if self.privileged else []) + self._prev_inputs(prev, a0, scale, radius)
                out = self.blocks[i](torch.cat([a0, a1, tmap, cost] + extra, 1))
                flow_s, mask_s = out[:, :4], out[:, 4:5]
                self.levels.append(resize(flow_s, scale) * scale)
            else:
                prev_scale = self.blocks_cfg[i - 1][0]
                ratio = prev_scale // scale
                flow_s = resize(flow_s, ratio) * ratio
                mask_s = resize(mask_s, ratio)
                w0, w1 = warp(a0, flow_s[:, :2]), warp(a1, flow_s[:, 2:])
                cost = cost_volume(a0, a1, t, flow_s, radius)
                extra = ([area(gt, scale)] if self.privileged else []) + self._prev_inputs(prev, a0, scale, radius)
                out = self.blocks[i](torch.cat([a0, a1, w0, w1, flow_s, mask_s, tmap, cost] + extra, 1))
                flow_s = flow_s + out[:, :4]
                mask_s = mask_s + out[:, 4:5]
                self.levels.append(resize(flow_s, scale) * scale)
                k = 5
                if self.conf:
                    self._conf = out[:, k:k + 1]
                    k += 1
                if self.mh:
                    self._mh = out[:, k:k + 6]
        scale = self.blocks_cfg[-1][0]
        return resize(flow_s, scale) * scale, resize(mask_s, scale)

    def forward(self, img0, img1, t, with_flow=False, gt=None, prev=None):
        self._conf = None
        self._mh = None
        flow, mask = self.flows(img0, img1, t, gt, prev)
        sharp = SHARP_WARP and not self.training
        w0, w1 = (warp_sharp if sharp else warp)(img0, flow[:, :2]), (warp_sharp if sharp else warp)(img1, flow[:, 2:])
        if self._mh is not None:
            # a second hypothesis of the motion (the first plus a correction from the last block) and four candidate colours (frame 0 and
            # frame 1, each at each hypothesis); each candidate's weight is read where the candidate was taken from, then a softmax
            # (the idea of the four-candidate blend of multiple-hypothesis interpolation, arXiv 2608.07120)
            sc = self.blocks_cfg[-1][0]
            mh = resize(self._mh, sc)
            flow_b = flow + mh[:, :4] * sc
            w0b, w1b = warp(img0, flow_b[:, :2]), warp(img1, flow_b[:, 2:])
            zero = torch.zeros_like(mask)
            logits = torch.cat([warp(mask, flow[:, :2]), warp(mask + mh[:, 4:5] - 6.0, flow_b[:, :2]),
                                warp(zero, flow[:, 2:]), warp(mh[:, 5:6] - 6.0, flow_b[:, 2:])], 1)
            wts = torch.softmax(logits, 1)
            out = w0 * wts[:, 0:1] + w0b * wts[:, 1:2] + w1 * wts[:, 2:3] + w1b * wts[:, 3:4]
            m = None
        else:
            m = torch.sigmoid(mask)
            if GATE2:
                v0, v1 = border_valid(flow[:, :2]), border_valid(flow[:, 2:])
                wa, wb = m * v0 + 1e-6, (1 - m) * v1 + 1e-6
                m = wa / (wa + wb)
            out = w0 * m + w1 * (1 - m)
        if self.refine is not None:
            out = out + self.refine(img0, img1, w0, w1, flow, mask, t)
        if self._conf is not None:
            # where the flow cannot be trusted the plain blend is the better guess: a learned confidence picks between them
            tt = t.view(-1, 1, 1, 1).to(img0.dtype)
            c = torch.sigmoid(resize(self._conf, self.blocks_cfg[-1][0]) + 3.0)
            out = out * c + (img0 * (1 - tt) + img1 * tt) * (1 - c)
        if self.gated:
            out = (out if GATE_MODE == 3 else gate4(img0, img1, out, t, flow) if GATE_MODE == 4 else gate3(img0, img1, out, t, flow) if GATE_MODE == 2 else gate2(img0, img1, out, t, w0, w1)) if GATE2 else gate(img0, img1, out, t)
        return (out, flow) if with_flow else out

    def count(self):
        return sum(p.numel() for p in self.parameters())


def build(state):
    """The network a checkpoint holds (its options are stored with it)."""
    return Interpolator([tuple(b) for b in state.get('blocks', DEFAULT_BLOCKS)], **state.get('options', {}))


def load_partial(model, weights, seed=0):
    """Loads what fits. A tensor of the same shape is copied. A tensor with more output rows than the checkpoint's (a confidence
    channel added to a last convolution) takes the rows there are and starts the rest at zero. A convolution that is wider
    than the checkpoint's (more channels in or out) takes the old weights in its corner and the model computes at the start what
    the checkpoint's did: new input channels of old units are zero, new units start small and random, and the weights that
    read new units start at zero. What was added entirely (a refinement) keeps its own start."""
    g = torch.Generator().manual_seed(seed)
    own = model.state_dict()
    for k, v in weights.items():
        if k not in own:
            continue
        o = own[k]
        if o.shape == v.shape:
            own[k] = v.clone()
        elif o.dim() == 1 and v.dim() == 1 and o.shape[0] > v.shape[0]:     # a bias
            z = torch.zeros_like(o)
            z[:v.shape[0]] = v
            own[k] = z
        elif o.dim() == 4 and all(a >= b for a, b in zip(o.shape, v.shape)):   # a convolution weight [out, in, 3, 3]
            z = torch.zeros_like(o)
            fan_in = o.shape[1] * 9
            new_rows = torch.randn(o.shape, generator=g) * (0.5 / fan_in ** 0.5)
            z[v.shape[0]:] = new_rows[v.shape[0]:] * 0.2      # new units: small random, over all inputs
            z[:v.shape[0], :v.shape[1]] = v                    # the old corner; new inputs of old units stay zero
            own[k] = z
    model.load_state_dict(own)
