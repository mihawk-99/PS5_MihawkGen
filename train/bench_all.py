# PS5_FrameGen - the honest benchmark: every method makes the middle picture of the same real triplets (frames the networks never
# saw), the pictures are stored, and one scorer measures all of them the same way (PSNR, SSIM, LPIPS), raw and behind the gate.
#
#   bench_all.py run    SET METHOD [METHOD ...]   make and store the pictures of a method (blend, repeat, ours:CKPT, ours-raw:CKPT,
#                                                 shaders:WEIGHTS, fsr, public:NAME = a model of bench_public.py)
#   bench_all.py score  SET [METHOD ...]          PSNR, SSIM and LPIPS over games (macro), over triplets (micro), and per game
#   bench_all.py table  SET METHOD METHOD ...     one method against another: the games it wins, the games it loses
#
# SET a: the frozen benchmark (160 clips of 45 games never trained on), SET b: the 3D set (3D clips
# set aside whole). The triplets are ~/retro-tests/bench-triplets (a) and bench-triplets-b (b): a, m, c, p (uint8 (T, H, W, 3)): the
# frame, the truth between, the next frame, the frame before. Pictures: ~/retro-tests/bench-out/SET/METHOD/CLIP.npy.
# FSR 3.1 runs as a black box on the same triplets: AMD's FidelityFX SDK 1.1.4 (optical flow and frame interpolation) with no engine data
# (zero motion vectors, flat depth). It is not trained on, copied or shipped.
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
RT = os.path.expanduser('~/retro-tests')
SETS = {'a': RT + '/bench-triplets', 'b': RT + '/bench-triplets-b', 's': RT + '/situations'}
TIMES = {'a': 0.5, 'b': 0.5, 's': 0.5}
for _set in ('a', 'b'):
    for _t in (25, 50, 75):
        SETS['q%s%d' % (_set, _t)] = RT + '/bench-quads-%s-t%d' % (_set, _t)
        TIMES['q%s%d' % (_set, _t)] = _t / 100.0
OUT = RT + '/bench-out'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEV == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.35)      # the desktop shares this GPU: never starve it (a diffusion model once did, and the display went black)


def clips(setname):
    for f in sorted(glob.glob(SETS[setname] + '/*.npz')):
        d = np.load(f)
        yield os.path.basename(f)[:-4], d['a'], d['m'], d['c'], (d['p'] if 'p' in d.files else d['a'])


def game(clip):
    return clip.rsplit('_', 1)[0]


def save(setname, method, clip, y):
    d = os.path.join(OUT, setname, method)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, clip + '.npy'), y)


def load(setname, method, clip):
    p = os.path.join(OUT, setname, method, clip + '.npy')
    return np.load(p) if os.path.exists(p) else None


def to_t(x):
    return torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255


def to_u8(t):
    return t[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


# ---- the methods

def run_simple(setname, method):
    for clip, a, m, c, p in clips(setname):
        if method == 'blend':
            t = TIMES[setname]
            y = (a.astype(np.float32) * (1 - t) + c.astype(np.float32) * t).round().astype(np.uint8)
        else:
            y = a.copy()
        save(setname, method, clip, y)


def run_ours(setname, method, ckpt):
    t = TIMES[setname]
    import bench
    mdl = bench.load_model(ckpt)
    raw = method.startswith('ours-raw')
    if raw:
        mdl.gated = False
    for clip, a, m, c, p in clips(setname):
        ys = [bench.run_model(mdl, a[i], c[i], p[i], gated=not raw, t=t) for i in range(len(a))]
        save(setname, method, clip, np.stack(ys))
    print(method, 'done', flush=True)


def run_shaders(setname, method, weights):
    """The whole generator, as the console runs it: build/neural_run (the compute shaders on this PC's Vulkan device)."""
    binary = os.path.join(ROOT, 'build', 'neural_run')
    for clip, a, m, c, p in clips(setname):
        ys = []
        with tempfile.TemporaryDirectory() as td:
            for i in range(len(a)):
                h, w = a[i].shape[:2]
                for n, x in (('a', a[i]), ('c', c[i])):
                    np.concatenate([x, np.full((h, w, 1), 255, np.uint8)], 2).tofile(f'{td}/{n}.rgba')
                subprocess.run([binary, weights, f'{td}/a.rgba', f'{td}/c.rgba', str(w), str(h), str(TIMES[setname]), f'{td}/o.rgba'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ys.append(np.fromfile(f'{td}/o.rgba', np.uint8).reshape(h, w, 4)[:, :, :3])
        save(setname, method, clip, np.stack(ys))
    print(method, 'done', flush=True)


def run_public(setname, name):
    """A public interpolation model (bench_public.py's registry, weights under ~/retro-tests/bench-models) on every triplet, at the midpoint only.
    Measurement references: nothing of them is trained on, copied or shipped."""
    if TIMES[setname] != 0.5:
        print(name, 'makes the midpoint only: nothing to run at', TIMES[setname])
        return
    import bench_public as BP
    net = BP.REGISTRY[name][0]()
    net.to('cuda')
    for clip, a, m, c, p in clips(setname):
        ys = []
        for i in range(len(a)):
            h, w = a[i].shape[:2]
            ph, pw = (-h) % net.pad, (-w) % net.pad

            def prep(img):
                x = torch.from_numpy(img).cuda().permute(2, 0, 1).float()[None] / 255.0
                return F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=net.fp16_ok):
                y = net.fn(prep(a[i]), prep(c[i]))
            ys.append(to_u8(y.float()[:, :, :h, :w]))
        save(setname, 'pub-' + name, clip, np.stack(ys))
    print('pub-' + name, 'done', flush=True)


def by_size(setname):
    groups = {}
    for clip, a, m, c, p in clips(setname):
        for i in range(len(a)):
            groups.setdefault(a[i].shape[:2], []).append((clip, i, a[i], c[i]))
    return groups


def run_black_box(setname, method):
    """FSR 3.1 on every triplet, in groups of one frame size (the tool takes a size for a job list)."""
    if method == 'fsr' and TIMES[setname] != 0.5:
        print('FSR 3 interpolates the midpoint only: nothing to run at', TIMES[setname])
        return
    tmp = tempfile.mkdtemp(prefix='bench_' + method + '_')
    results = {}
    for (H, W), items in sorted(by_size(setname).items()):
        jobs = []
        np.zeros((H, W, 2), np.float32).tofile(f'{tmp}/zmv.f32')
        np.full((H, W), 0.5, np.float32).tofile(f'{tmp}/dep.f32')
        for k, (clip, i, a, c) in enumerate(items):
            for n, x in (('a', a), ('c', c)):
                np.concatenate([x, np.full((H, W, 1), 255, np.uint8)], 2).tofile(f'{tmp}/{n}{k}.rgba')
            jobs.append(f'{tmp}/a{k}.rgba {tmp}/c{k}.rgba {tmp}/zmv.f32 {tmp}/zmv.f32 {tmp}/dep.f32 {tmp}/dep.f32 {tmp}/o{k}.rgba')
        open(f'{tmp}/jobs.txt', 'w').write('\n'.join(jobs) + '\n')
        cmd = [RT + '/bench-models/fsr/fsr_bench', str(W), str(H), f'{tmp}/jobs.txt']
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(method, 'failed at', (H, W), r.stderr[-300:], flush=True)
            continue
        for k, (clip, i, a, c) in enumerate(items):
            o = np.fromfile(f'{tmp}/o{k}.rgba', np.uint8).reshape(H, W, 4)[:, :, :3]
            results.setdefault(clip, {})[i] = o
        print(method, 'size', (H, W), len(items), 'done', flush=True)
    for clip, d in results.items():
        save(setname, method, clip, np.stack([d[i] for i in sorted(d)]))


# ---- the measures

def gaussian_window(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


_WIN = None


def ssim(x, y):
    """SSIM of two (1, 3, H, W) tensors in [0, 1] (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5, mean over the channels)."""
    global _WIN
    if _WIN is None:
        _WIN = gaussian_window().to(x.device)
    c = x.shape[1]
    w = _WIN.expand(c, 1, -1, -1)
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean().item()


def psnr(x, y):
    mse = ((x - y) ** 2).mean().item()
    return 99.0 if mse < 1e-12 else -10 * np.log10(mse)


_LPIPS = None


def lpips_of(x, y):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net='alex', verbose=False).to(DEV).eval()
    with torch.no_grad():
        return _LPIPS(x * 2 - 1, y * 2 - 1).item()


def scores_of(setname, method, gated):
    """{clip: (psnr[], ssim[], lpips[])} of a method; gated: the picture behind the last pass's gate (model.gate)."""
    from model import gate
    cache = os.path.join(OUT, setname, method + ('+gate' if gated else '') + '.scores.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    res = {}
    for clip, a, m, c, p in clips(setname):
        y = load(setname, method, clip)
        if y is None:
            continue
        ps, ss, ls = [], [], []
        for i in range(len(a)):
            ty = to_t(y[i]).to(DEV)
            if gated:
                x0, x1 = to_t(a[i]).to(DEV), to_t(c[i]).to(DEV)
                ty = to_t(to_u8(gate(x0, x1, ty, torch.tensor([TIMES[setname]], device=DEV)))).to(DEV)
            tm = to_t(m[i]).to(DEV)
            ps.append(psnr(ty, tm))
            ss.append(ssim(ty, tm))
            ls.append(lpips_of(ty, tm))
        res[clip] = (ps, ss, ls)
    json.dump(res, open(cache, 'w'))
    return res


CLEAN = False


def excluded_games():
    p = os.path.join(RT, 'bench-excluded.txt')
    return set(l.strip() for l in open(p)) if os.path.exists(p) else set()


def summarize(res):
    games = {}
    drop = excluded_games() if CLEAN else set()
    for clip, (ps, ss, ls) in res.items():
        if game(clip) in drop:
            continue
        g = games.setdefault(game(clip), ([], [], []))
        g[0].extend(ps)
        g[1].extend(ss)
        g[2].extend(ls)
    per = {g: tuple(float(np.mean(v)) for v in vals) for g, vals in games.items()}
    macro = tuple(float(np.mean([per[g][k] for g in per])) for k in range(3))
    micro = tuple(float(np.mean([x for vals in games.values() for x in vals[k]])) for k in range(3))
    return per, macro, micro


def export_quads(which, per_clip=6):
    """Frames four new frames apart and the three between them (truth at t = 0.25, 0.5, 0.75): the x4 test (a source of a quarter of the
    rate, harder than 30 to 120). Written as three triplet folders, one for each instant."""
    from real import Clips
    held = Clips(held=True, bench3d=False) if which == 'a' else Clips(root=os.path.expanduser('~/retro-tests/bench3d-clips'), held=True, bench3d=False)
    out = {t: {} for t in (25, 50, 75)}
    for name, clip, new in zip(held.names, held.clips, held.new):
        n = len(new) - 4
        if n < 1:
            continue
        picks = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(per_clip, n))))
        a = np.stack([clip[new[j]].numpy() for j in picks])
        c = np.stack([clip[new[j + 4]].numpy() for j in picks])
        for k, t in enumerate((25, 50, 75), 1):
            m = np.stack([clip[new[j + k]].numpy() for j in picks])
            out[t][name] = dict(a=a, m=m, c=c, p=a)
    for t, clips_ in out.items():
        d = SETS['q%s%d' % (which, t)]
        os.makedirs(d, exist_ok=True)
        for name, arrs in clips_.items():
            np.savez(os.path.join(d, name + '.npz'), **arrs)
    print(which, len(out[50]), 'clips,', sum(len(v['a']) for v in out[50].values()), 'quads')


def main():
    if sys.argv[1] == 'export-quads':
        export_quads(sys.argv[2])
        return
    global CLEAN
    cmd, setname = sys.argv[1], sys.argv[2]
    rest = [x for x in sys.argv[3:] if x != '--clean']
    CLEAN = '--clean' in sys.argv[3:]
    if cmd == 'run':
        for m in rest:
            kind, _, arg = m.partition(':')
            if kind in ('blend', 'repeat'):
                run_simple(setname, kind)
            elif kind in ('ours', 'ours-raw'):
                run_ours(setname, kind + '-' + os.path.basename(os.path.dirname(os.path.abspath(arg))), arg)
            elif kind == 'shaders':
                run_shaders(setname, 'shaders-' + os.path.basename(arg).replace('.fgw', ''), os.path.abspath(arg))
            elif kind == 'public':
                run_public(setname, arg)
            elif kind == 'fsr':
                run_black_box(setname, kind)
            else:
                print('unknown method', m)
    elif cmd == 'grid':
        # one row per situation-level (the file's name before the last underscore), one column per method, raw pictures
        k = 1 if '--ssim' in sys.argv else 2 if '--lpips' in sys.argv else 0
        cols = {m: summarize(scores_of(setname, m, False))[0] for m in rest if scores_of(setname, m, False)}
        names = list(cols)
        print(f'{"situation":24s}' + ''.join(f'{n[:11]:>12s}' for n in names))
        fam = {}
        for g in sorted(cols[names[0]]):
            vals = [cols[n].get(g, (float('nan'),) * 3)[k] for n in names]
            print(f'{g:24s}' + ''.join(f'{v:12.3f}' for v in vals))
            fam.setdefault(g.split('-')[0], []).append(vals)
        print(f'{"-- families (mean)":24s}')
        for f_, rows in fam.items():
            print(f'{f_:24s}' + ''.join(f'{v:12.3f}' for v in np.mean(rows, 0)))
    elif cmd in ('score', 'table'):
        methods = rest or sorted(d for d in os.listdir(os.path.join(OUT, setname)) if os.path.isdir(os.path.join(OUT, setname, d)))
        allres = {}
        for m in methods:
            for gated in (False, True):
                r = scores_of(setname, m, gated)
                if r:
                    allres[m + ('+gate' if gated else '')] = summarize(r)
        print(f'{"method":28s} {"PSNR":>7s} {"SSIM":>7s} {"LPIPS":>7s}   (games average; triplets average PSNR)')
        for k, (per, macro, micro) in sorted(allres.items(), key=lambda kv: -kv[1][1][0]):
            print(f'{k:28s} {macro[0]:7.3f} {macro[1]:7.4f} {macro[2]:7.4f}   {micro[0]:7.3f}   ({len(per)} games)')
        if cmd == 'table' and len(rest) >= 2:
            a, b = rest[0], rest[1]
            for suffix in ('', '+gate'):
                if a + suffix in allres and b + suffix in allres:
                    pa, pb = allres[a + suffix][0], allres[b + suffix][0]
                    wins = [g for g in pa if g in pb and pa[g][0] > pb[g][0]]
                    print(f'{a}{suffix} beats {b}{suffix} in PSNR on {len(wins)} of {len(pa)} games')
                    for g in sorted(pa, key=lambda g: pa[g][0] - pb[g][0]):
                        if pa[g][0] < pb[g][0]:
                            print(f'   loses {g:24s} {pa[g][0]:7.2f} against {pb[g][0]:7.2f}   ({pa[g][0] - pb[g][0]:+.2f})')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
