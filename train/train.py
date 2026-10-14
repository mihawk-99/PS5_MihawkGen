# PS5_FrameGen - trains the interpolation network on the data factory's scenes
# (docs/plan.md, phases 4 and 5). Clips come from train/synth.py on the GPU, a
# fresh batch each step, so there is no dataset to store and no scene is seen
# twice; held-out clips come from a fixed seed no training batch uses.
#
#   .venv/bin/python train/train.py --name v1 --steps 20000
#
# Checkpoints and the log go to train/runs/<name>/ (ignored by git).
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import numpy as np
import copy
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import charbonnier, laplacian_loss, psnr, ssim  # noqa: E402
from model import Interpolator, DEFAULT_BLOCKS, build, load_partial  # noqa: E402
from synth import PRESETS, Synth, STEPS  # noqa: E402
from real import Clips  # noqa: E402

HELD_OUT_SEED = 987654321


def make_batch(synth, clips, n, crop, device, cut_fraction=0.0):
    """n samples from the clips: a random gap (2 or 4 steps), start, target step and crop; random flips, a swap of the
    two frames (t becomes 1 - t) and a colour permutation. Returns frame0, frame1, the target, t and the exact flows
    from the target to each frame."""
    frames = clips.frames
    b, _, _, H, W = frames.shape
    ks = torch.randint(0, b, (n,), device=device)
    gap = torch.where(torch.rand(n, device=device) < 0.5, 2, 4)
    start = (torch.rand(n, device=device) * (STEPS - gap)).long()
    mid = start + 1 + (torch.rand(n, device=device) * (gap - 1)).long().clamp(max=2)
    mid = torch.minimum(mid, start + gap - 1)
    t = (mid - start).float() / gap
    swap = torch.rand(n, device=device) < 0.5
    s0 = torch.where(swap, start + gap, start)
    s1 = torch.where(swap, start, start + gap)
    t = torch.where(swap, 1 - t, t)
    a, c, m = frames[ks, s0], frames[ks, s1], frames[ks, mid]
    sp = s0 + torch.where(swap, gap, -gap)                      # the frame before the pair, as far back as the pair is long
    have = (sp >= 0) & (sp < STEPS)
    pf = torch.where(have.view(n, 1, 1, 1), frames[ks, sp.clamp(0, STEPS - 1)], a)
    fa, fc = synth.flow_gt(clips, ks, mid, s0), synth.flow_gt(clips, ks, mid, s1)
    # Scene cuts: the second frame comes from another clip; the truth is the nearer frame (the frame
    # before the cut for t < 1/2, after it from 1/2 on) and there is no flow to learn
    cut = torch.rand(n, device=device) < cut_fraction
    if cut.any():
        other = torch.randint(0, b, (n,), device=device)
        other = torch.where(other == ks, (other + 1) % b, other)
        c2 = frames[other, s1]
        c = torch.where(cut.view(n, 1, 1, 1), c2, c)
        m = torch.where(cut.view(n, 1, 1, 1), torch.where((t < 0.5).view(n, 1, 1, 1), a, c2), m)
    ys = torch.randint(0, H - crop + 1, (n,)).tolist()
    xs = torch.randint(0, W - crop + 1, (n,)).tolist()
    hflip = (torch.rand(n) < 0.5).tolist()
    vflip = (torch.rand(n) < 0.5).tolist()
    out = [[], [], [], [], [], []]
    for i in range(n):
        sl = (slice(None), slice(ys[i], ys[i] + crop), slice(xs[i], xs[i] + crop))
        parts = [a[i][sl], c[i][sl], m[i][sl], fa[i][sl], fc[i][sl], pf[i][sl]]
        if hflip[i]:
            parts = [x.flip(-1) for x in parts]
            parts[3] = parts[3] * torch.tensor([-1.0, 1.0], device=device).view(2, 1, 1)
            parts[4] = parts[4] * torch.tensor([-1.0, 1.0], device=device).view(2, 1, 1)
        if vflip[i]:
            parts = [x.flip(-2) for x in parts]
            parts[3] = parts[3] * torch.tensor([1.0, -1.0], device=device).view(2, 1, 1)
            parts[4] = parts[4] * torch.tensor([1.0, -1.0], device=device).view(2, 1, 1)
        perm = torch.randperm(3, device=device)
        for k in (0, 1, 2, 5):
            parts[k] = parts[k][perm]
        for k in range(6):
            out[k].append(parts[k])
    res = [torch.stack(x) for x in out]
    return res[0], res[1], res[2], res[3], res[4], t, cut, res[5]


@torch.no_grad()
def evaluate(model, held, device):
    """PSNR and SSIM on the held-out clips at full size, for gaps of 2 and 4, against blend and repeat."""
    model.eval()
    sums = {}

    def add(name, value):
        sums.setdefault(name, []).append(value)

    pieces = [c[i:i + 4] for c in held for i in range(0, c.shape[0], 4)]
    for clips in pieces:
        for gap, starts in ((2, (0, 2)), (4, (0,))):
            for start in starts:
                for mid in range(start + 1, start + gap):
                    t = (mid - start) / gap
                    a, c, m = clips[:, start], clips[:, start + gap], clips[:, mid]
                    tt = torch.full((a.shape[0],), t, device=device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        out = model(a, c, tt)
                    out = out.float().clamp(0, 1)
                    blend = a * (1 - t) + c * t
                    key = f'gap{gap}'
                    add(f'{key}/model_psnr', psnr(out, m).mean().item())
                    add(f'{key}/model_ssim', ssim(out, m).mean().item())
                    add(f'{key}/blend_psnr', psnr(blend, m).mean().item())
                    add(f'{key}/repeat_psnr', psnr(a if t < 0.5 else c, m).mean().item())
    model.train()
    return {k: sum(v) / len(v) for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--clips', type=int, default=24, help='clips rendered per step')
    ap.add_argument('--batch', type=int, default=48, help='training pairs per step')
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--flow-weight', type=float, default=0.05, help='weight of the supervised flow loss')
    ap.add_argument('--presets', type=float, default=0.0, help='fraction of steps whose scenes follow one failure-mode preset (train/synth.py)')
    ap.add_argument('--pixel', type=float, default=0.0, help='fraction of steps that render retro-console pixel clips (the \'pixel\' preset)')
    ap.add_argument('--real', type=float, default=0.0, help='fraction of steps whose batch is real clips (train/real.py, ~/retro-tests/train-clips)')
    ap.add_argument('--cuts', type=float, default=0.0, help='fraction of samples that are scene cuts')
    ap.add_argument('--init', default=None, help='start from this checkpoint\'s weights, with a fresh optimiser and schedule')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--blocks', default=None, help='JSON list of [scale, width, convs, radius]')
    ap.add_argument('--conf', action='store_true', help='a learned confidence between the warped picture and the plain blend')
    ap.add_argument('--refine', default=None, help='width,convs,scale of a residual refinement after the warp')
    ap.add_argument('--gated', action='store_true', help='train the picture the generator shows (the last pass\'s gate on the frames\' agreement)')
    ap.add_argument('--distill', type=float, default=0.0, help='weight of the flow distillation from a teacher that also sees the true middle frame (trained alongside)')
    ap.add_argument('--scales', default='1', help='comma list of scale factors: a batch takes a smaller crop and enlarges it by one of them (motion in pixels grows with it), so the network sees the sizes it will work at')
    ap.add_argument('--mh', action='store_true', help='a second motion hypothesis and a four-candidate blend')
    ap.add_argument('--prev', action='store_true', help='the network is also given the frame before the pair (a motion prior)')
    ap.add_argument('--teacher-dir', default=None, help='a folder of public models\' pictures for the pairs of real clips (train/teachers.py): an extra target')
    ap.add_argument('--teacher-weight', type=float, default=1.0, help='weight of the loss against the teachers\' pictures (behind the same last pass)')
    ap.add_argument('--sit', type=float, default=0.0, help='fraction of steps whose batch is random situations (train/situations.py: scrolls, pans, zooms, fades, sprites ...) made from the training games\' pictures')
    ap.add_argument('--deep-flow', type=float, default=0.0, help='weight of the flow loss at the blocks before the last, relative to the flow weight')
    ap.add_argument('--sit-batch', type=int, default=24)
    ap.add_argument('--mse', type=float, default=0.0, help='weight of a squared error term (PSNR is a squared error measure)')
    args = ap.parse_args()

    device = 'cuda'
    if float(os.environ.get('FG_GPU_FRACTION', '0')) > 0:
        torch.cuda.set_per_process_memory_fraction(float(os.environ['FG_GPU_FRACTION']))   # leaves room for other programs on the GPU
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', args.name)
    os.makedirs(out, exist_ok=True)
    blocks = json.loads(args.blocks) if args.blocks else DEFAULT_BLOCKS
    opts = dict(prev=args.prev, mh=args.mh, conf=args.conf, refine=[int(x) for x in args.refine.split(',')] if args.refine else None, gated=args.gated)
    if args.init:
        base = torch.load(args.init, map_location=device)
        if not args.blocks:
            blocks = base.get('blocks', blocks)
        for k, v in base.get('options', {}).items():
            opts.setdefault(k, v)
            if not opts[k] and v:
                opts[k] = v
    model = Interpolator([tuple(b) for b in blocks], **opts).to(device)
    teacher = None
    if args.distill > 0:
        teacher = Interpolator([tuple(b) for b in blocks], privileged=True).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f'parameters: {model.count():,}', flush=True)
    opt = torch.optim.AdamW(list(model.parameters()) + (list(teacher.parameters()) if teacher is not None else []), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    step0 = 0
    if args.init:
        state = torch.load(args.init, map_location=device)
        load_partial(model, state['model']); load_partial(ema, state['ema'])
        if teacher is not None:
            load_partial(teacher, {k: v for k, v in state['model'].items() if k.startswith('blocks.')})   # the teacher starts as the student, its new inputs at zero
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); opt.load_state_dict(state['opt'])
        step0 = state['step']

    synth = Synth(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(time.time()) % 2 ** 31)
    gen_py = random.Random(int(time.time()))
    held_gen = torch.Generator(device=device)
    held_gen.manual_seed(HELD_OUT_SEED)
    held = [synth.render(8, held_gen).frames for _ in range(2)]
    real_train = real_held = None
    if args.real > 0:
        real_train, real_held = Clips(), Clips(held=True)
        real_train.teacher_dir = os.path.expanduser(args.teacher_dir) if args.teacher_dir else None
        print(f'real clips: {len(real_train)} to train on, {len(real_held)} held out', flush=True)
    real_rng = random.Random(7)
    sit_pool, sit_rng = None, np.random.default_rng(11)
    if args.sit > 0:
        import situations
        sit_pool = situations.Pool(real_train)
    held_batches = None
    if real_held is not None and len(real_held):
        hr = random.Random(99)
        held_batches = [real_held.sample(16, 256, device, hr, keep_still=0.0) for _ in range(6)]
    log = open(os.path.join(out, 'log.jsonl'), 'a')
    t_start = time.time()
    warmup = 300
    for step in range(step0, args.steps):
        lr = args.lr * min(1.0, (step + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * step / args.steps)) * 0.97 + 0.03)
        for g in opt.param_groups:
            g['lr'] = lr
        preset = None
        if args.pixel > 0 and gen_py.random() < args.pixel:
            preset = 'pixel'
        elif args.presets > 0 and gen_py.random() < args.presets:
            preset = gen_py.choice(sorted(PRESETS))
        scale = gen_py.choice([float(x) for x in args.scales.split(',')])
        crop_src = max(32, int(round(args.crop / scale / 2)) * 2) if scale != 1.0 else args.crop
        if sit_pool is not None and gen_py.random() < args.sit:
            a, c, tgt, t, fa, fc, has_flow = sit_pool.batch(args.sit_batch, crop_src, sit_rng, device)
            pf, tpic, tmask = a, tgt, torch.zeros(args.sit_batch, device=device)
            cut = has_flow < 0.5      # (the flow is exact for the situations that move rigidly)
        elif real_train is not None and len(real_train) and gen_py.random() < args.real:
            a, c, tgt, t, pf, tpic, tmask = real_train.sample(args.batch, crop_src, device, real_rng, with_teacher=True)
            fa = fc = torch.zeros(args.batch, 2, crop_src, crop_src, device=device)
            cut = torch.ones(args.batch, dtype=torch.bool, device=device)    # (no flow truth: the flow loss is off)
        else:
            clips = synth.render(args.clips, gen, preset=preset)
            a, c, tgt, fa, fc, t, cut, pf = make_batch(synth, clips, args.batch, crop_src, device, args.cuts)
            tpic, tmask = tgt, torch.zeros(args.batch, device=device)
        if crop_src != args.crop:
            up = lambda x: torch.nn.functional.interpolate(x, size=(args.crop, args.crop), mode='bicubic', align_corners=False).clamp(0, 1)
            a, c, tgt, pf, tpic = up(a), up(c), up(tgt), up(pf), up(tpic)
            fa = torch.nn.functional.interpolate(fa, size=(args.crop, args.crop), mode='bilinear', align_corners=False) * (args.crop / crop_src)
            fc = torch.nn.functional.interpolate(fc, size=(args.crop, args.crop), mode='bilinear', align_corners=False) * (args.crop / crop_src)
        with torch.autocast('cuda', dtype=torch.float16):
            pred, flow = model(a, c, t, with_flow=True, prev=pf if args.prev else None)
        pred, flow = pred.float(), flow.float()
        rec = charbonnier(pred - tgt).mean() + 0.5 * laplacian_loss(pred, tgt)
        if args.teacher_dir and bool((tmask > 0).any()):
            sel = (tmask > 0).nonzero().flatten()
            from model import gate as final_gate
            with torch.no_grad():
                tg = final_gate(a[sel], c[sel], tpic[sel], t[sel])
            rec = rec + args.teacher_weight * (charbonnier(pred[sel] - tg).mean() + 0.5 * laplacian_loss(pred[sel], tg))
        distill_loss = torch.zeros((), device=device)
        if teacher is not None:
            levels_s = model.levels
            with torch.autocast('cuda', dtype=torch.float16):
                pred_t = teacher(a, c, t, gt=tgt)
            pred_t = pred_t.float()
            rec = rec + charbonnier(pred_t - tgt).mean() + 0.5 * laplacian_loss(pred_t, tgt)
            # where the teacher (who has seen the answer) is the better of the two, the student's flow at every scale is pulled to its
            with torch.no_grad():
                better = ((pred_t - tgt).abs().mean(1, keepdim=True) < (pred - tgt).abs().mean(1, keepdim=True)).float()
            for fs, ft in zip(levels_s, teacher.levels):
                distill_loss = distill_loss + ((fs.float() - ft.float().detach()).abs().mean(1, keepdim=True) * better).mean() / len(levels_s)
            rec = rec + args.distill * distill_loss
        if args.mse > 0:
            rec = rec + args.mse * ((pred - tgt) ** 2).mean()
        keep = (~cut).float().view(-1, 1, 1, 1)
        flow_loss = (torch.nn.functional.smooth_l1_loss(flow, torch.cat([fa, fc], 1), beta=1.0, reduction='none') * keep).mean()
        if args.deep_flow > 0:
            gt_flow = torch.cat([fa, fc], 1)
            for lv in model.levels[:-1]:
                flow_loss = flow_loss + args.deep_flow * (torch.nn.functional.smooth_l1_loss(lv.float(), gt_flow, beta=1.0, reduction='none') * keep).mean()
        loss = rec + args.flow_weight * flow_loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(teacher.parameters()) if teacher is not None else []), 1.0)
        scaler.step(opt)
        scaler.update()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        if step % 50 == 0:
            rec = {'step': step, 'loss': loss.item(), 'rec': rec.item(), 'flow': flow_loss.item(), 'lr': lr, 'elapsed': time.time() - t_start}
            log.write(json.dumps(rec) + '\n'); log.flush()
            if step % 200 == 0:
                print(f'step {step} loss {loss.item():.4f} rec {rec["rec"]:.4f} flow {rec["flow"]:.3f} lr {lr:.2e} {time.time() - t_start:.0f}s', flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            res = evaluate(ema, held, device)
            if held_batches:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    res['real_held'] = float(sum(psnr(ema(a_, c_, t_, prev=p_ if args.prev else None).float(), g_).mean() for a_, c_, g_, t_, p_ in held_batches) / len(held_batches))
            rec = {'step': step + 1, 'eval': res}
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(f'eval @ {step + 1}: ' + ' '.join(f'{k}={v:.2f}' for k, v in sorted(res.items())), flush=True)
            torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'step': step + 1,
                        'blocks': [list(b) for b in blocks], 'options': model.options}, os.path.join(out, 'ckpt.pt'))


if __name__ == '__main__':
    main()
