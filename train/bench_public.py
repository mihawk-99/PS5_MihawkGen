# PS5_FrameGen - the benchmark of train/bench.py, run with public interpolation models (measurement references only):
# the middle frame (t = 1/2) of the exported triplets of the held-out game clips, full frames, PSNR on 8-bit rounded
# output (psnr8 and the aggregation are those of bench.py: per clip, per game, over games, over triplets, median clip).
# Nothing here trains anything, and nothing of these models is copied into the repository: the code and the weights live
# under ~/retro-tests/bench-models/<name>/ (SOURCES.md there: URLs, versions, licences) and are imported from there. The
# script imports nothing else from outside that directory (it does not import train/model.py).
#
#   ~/retro-tests/bench-models/venv/bin/python train/bench_public.py [--models rife-4.26,ifrnet-l,...] [--per-game]
#                                                                    [--json OUT.json] [--limit CLIPS] [--fp16] [--no-cache]
#   ... --list                                                       the models and whether their files are there
#
# Frames are padded (replicate, right and bottom) to the multiple each model needs and cropped back. A frame that runs
# out of GPU memory (the GPU is shared with a training run) is retried once, then the model moves to the CPU (fp32) for
# the rest; every retry and move is counted and printed, none is silent. A model that
# fails to load, or fails on a frame in every way, is reported as such and left out of the table (never skipped quietly).
# Each model's per-clip scores are cached in ~/retro-tests/bench-public-cache/ (fp32 and fp16 apart; --no-cache redoes).
#
# Copyright (C) 2026 Mihawk
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import contextlib
import gc
import glob
import importlib
import json
import os
import sys
import time
import typing

# the script's own directory holds train/model.py, which would shadow the models' own "model" packages
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HOME = os.path.expanduser('~')
MODELS_DIR = os.path.join(HOME, 'retro-tests', 'bench-models')
VENV_DIR = os.path.join(MODELS_DIR, 'venv')
TRIPLETS_DIR = os.path.join(HOME, 'retro-tests', 'bench-triplets')
CACHE_DIR = os.path.join(HOME, 'retro-tests', 'bench-public-cache')
OUT_JSON = os.path.join(HOME, 'retro-tests', 'bench-public.json')


def psnr8(pred, truth):
    """PSNR in dB between two uint8 images (as a display shows them)."""
    mse = np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-12 else float(10 * np.log10(255.0 ** 2 / mse))


def gate(x0, x1, moved, t=0.5):
    """The last pass of the generator (framegen/shaders/nn_final.comp), as train/bench.py applies it to our own network: where
    the two frames agree within a few levels (the largest difference over a 3x3 grid of points two pixels apart) the picture is
    their blend, exact; the model's picture is taken in proportion as they differ (smoothstep from 2 to 12 levels). Reported
    as "<model>+gate" next to the model's own output, so that the comparison with our gated numbers is like for like."""
    d = (x0 - x1).abs().amax(1, keepdim=True)
    pad = F.pad(d, (2, 2, 2, 2), mode='replicate')
    h, w = d.shape[-2:]
    diff = torch.zeros_like(d)
    for j in (0, 2, 4):
        for i in (0, 2, 4):
            diff = torch.maximum(diff, pad[:, :, j:j + h, i:i + w])
    u = ((diff - 2 / 255) / (10 / 255)).clamp(0, 1)
    motion = u * u * (3 - 2 * u)
    still = x0 * (1 - t) + x1 * t
    return still + (moved - still) * motion


@contextlib.contextmanager
def sandbox(*dirs):
    """Import code from the model's directory; afterwards forget the modules that came from under MODELS_DIR, so that the
    next model's own "model", "utils", "config" or "train_log" are not shadowed by this one's."""
    before = set(sys.modules)
    for d in reversed(dirs):
        sys.path.insert(0, d)
    try:
        yield
    finally:
        for d in dirs:
            if d in sys.path:
                sys.path.remove(d)
        for k in set(sys.modules) - before:
            f = getattr(sys.modules.get(k), '__file__', None) or ''
            if (f.startswith(MODELS_DIR) and not f.startswith(VENV_DIR)) or k in ('model', 'models', 'utils', 'loss', 'config', 'train_log', 'networks'):
                sys.modules.pop(k, None)


class Net:
    """A loaded model: fn(x0, x1) -> prediction, (1,3,H,W) float in 0..1 at t = 1/2, on whatever device the inputs are on;
    pad = the multiple the model needs; module = the torch module(s) to move between devices."""

    def __init__(self, fn, module, pad, fp16_ok=True):
        self.fn, self.module, self.pad, self.fp16_ok = fn, module, pad, fp16_ok
        self.dev = 'cpu'
        self.gpu_only = False

    def to(self, dev):
        self.module.to(dev)
        self.dev = dev


def _need(*paths):
    for p in paths:
        if not os.path.exists(os.path.join(MODELS_DIR, p)):
            raise FileNotFoundError(os.path.join(MODELS_DIR, p))


# --- the adapters: one loader per model, each as its authors run it for one middle frame --------------------------------

def load_rife(version):
    def load():
        root = os.path.join(MODELS_DIR, 'rife', 'v' + version)
        _need(os.path.join('rife', 'v' + version, 'train_log', 'flownet.pkl'))
        with sandbox(root):
            from train_log.RIFE_HDv3 import Model
            m = Model()
            m.load_model(os.path.join(root, 'train_log'), -1)
            m.eval()
        m.flownet.cpu()
        return Net(lambda x0, x1: m.inference(x0, x1, 0.5, 1.0), m.flownet, 128)
    return load


def load_ifrnet(variant):
    def load():
        root = os.path.join(MODELS_DIR, 'IFRNet')
        ck = {'l': ('IFRNet_large', 'IFRNet_L_Vimeo90K.pth', 'IFRNet_L'), 'b': ('IFRNet', 'IFRNet_Vimeo90K.pth', 'IFRNet')}[variant]
        _need(os.path.join('IFRNet', 'checkpoints', ck[0], ck[1]))
        with sandbox(root):
            Model = importlib.import_module('models.' + ck[2]).Model
            m = Model().eval()
            m.load_state_dict(torch.load(os.path.join(root, 'checkpoints', ck[0], ck[1]), map_location='cpu'))
        return Net(lambda x0, x1: m.inference(x0, x1, x0.new_full((1, 1, 1, 1), 0.5)), m, 32)
    return load


def load_emavfi(name, F_, depth):
    def load():
        root = os.path.join(MODELS_DIR, 'EMA-VFI')
        _need(os.path.join('EMA-VFI', 'ckpt', name + '.pkl'))
        with sandbox(root):
            import config as cfg
            backbone, multiscale = cfg.MODEL_CONFIG['MODEL_TYPE']
            bcfg, mcfg = cfg.init_model_config(F=F_, depth=depth)
            net = multiscale(backbone(**bcfg), **mcfg)       # what Trainer.Model builds, minus its move to the GPU
            sd = torch.load(os.path.join(root, 'ckpt', name + '.pkl'), map_location='cpu')
            sd = {k.replace('module.', ''): v for k, v in sd.items()
                  if 'module.' in k and 'attn_mask' not in k and 'HW' not in k}
            net.load_state_dict(sd)
            net.eval()
        return Net(lambda x0, x1: net(torch.cat((x0, x1), 1), timestep=0.5)[3], net, 32)    # Trainer.inference, TTA off
    return load


def load_amt(name, cfgname, ckpt):
    def load():
        import yaml
        root = os.path.join(MODELS_DIR, 'AMT')
        _need(os.path.join('AMT', 'pretrained', ckpt))
        cfg = yaml.safe_load(open(os.path.join(root, 'cfgs', cfgname)))['network']
        with sandbox(root):
            mod, cls = cfg['name'].rsplit('.', 1)
            m = getattr(importlib.import_module(mod), cls)(**cfg.get('params', {}))
            with torch.serialization.safe_globals([typing.OrderedDict]):     # the one non-tensor global in the file
                sd = torch.load(os.path.join(root, 'pretrained', ckpt), map_location='cpu')['state_dict']
            m.load_state_dict(sd)
            m.eval()
        return Net(lambda x0, x1: m(x0, x1, x0.new_full((1, 1, 1, 1), 0.5), scale_factor=1.0, eval=True)['imgt_pred'], m, 32)
    return load


def load_film():
    root = os.path.join(MODELS_DIR, 'FILM')
    path = os.path.join(root, 'weights', 'film_net_fp32.pt')

    def load():
        _need(os.path.join('FILM', 'weights', 'film_net_fp32.pt'))
        m = torch.jit.load(path, map_location='cpu').float().eval()
        return Net(lambda x0, x1: m(x0, x1, x0.new_full((1, 1), 0.5)), m, 64, fp16_ok=False)
    return load


def load_vfimamba(name, F_, depth):
    def load():
        root = os.path.join(MODELS_DIR, 'VFIMamba')
        _need(os.path.join('VFIMamba', 'ckpt', name + '.pkl'), os.path.join('shims', 'mamba_ssm'))
        # mamba_ssm's CUDA kernel is not built here (no nvcc): shims/ holds a pure-PyTorch selective scan, same recurrence
        with sandbox(root, os.path.join(MODELS_DIR, 'shims')):
            import config as cfg
            backbone, multiscale = cfg.MODEL_CONFIG['MODEL_TYPE']
            bcfg, mcfg = cfg.init_model_config(F=F_, depth=depth)
            net = multiscale(backbone(**bcfg), **mcfg)       # what Trainer_finetune.Model builds, minus its move to the GPU
            sd = torch.load(os.path.join(root, 'ckpt', name + '.pkl'), map_location='cpu')
            sd = {k.replace('module.', ''): v for k, v in sd.items()
                  if 'module.' in k and 'attn_mask' not in k and 'HW' not in k}
            net.load_state_dict(sd, strict=True)
            net.eval()
        # Trainer_finetune.Model.inference with TTA off, local on, scale 0
        return Net(lambda x0, x1: net(torch.cat((x0, x1), 1), timestep=0.5, scale=0, local=True)[3], net, 32, fp16_ok=False)
    return load


def load_gimm(variant):
    """GIMM-VFI (Guo et al., NeurIPS 2024), -R (RAFT flow) or -F (FlowFormer flow), the non-perceptual weights, as its video_Nx.py runs
    them for one middle frame (t = 1/2, ds_factor 1). Two local patches for Python 3.12 and cupy 14, in ~/retro-tests/bench-models/GIMM-VFI
    (the .orig files beside them): dataclass defaults as default_factory, softsplat's kernel compiled as a cupy.RawKernel."""
    def load():
        root = os.path.join(MODELS_DIR, 'GIMM-VFI')
        ck = {'r': 'gimmvfi_r_arb.pt', 'f': 'gimmvfi_f_arb.pt'}[variant]
        _need(os.path.join('GIMM-VFI', 'pretrained_ckpt', ck))
        cwd = os.getcwd()
        os.chdir(root)          # the flow estimators load their weights from ./pretrained_ckpt
        try:
            with sandbox(os.path.join(root, 'src')):
                from models import create_model
                from utils.config import load_config, augment_arch_defaults
                cfg = load_config(os.path.join(root, 'configs', 'gimmvfi', 'gimmvfi_%s_arb.yaml' % variant))
                m, _ = create_model(augment_arch_defaults(cfg.arch))
                m.load_state_dict(torch.load(os.path.join(root, 'pretrained_ckpt', ck), map_location='cpu', weights_only=False)['state_dict'], strict=True)
                m.eval()
        finally:
            os.chdir(cwd)

        def fn(x0, x1):
            xs = torch.stack((x0, x1), 2)
            coord = [(m.sample_coord_input(1, xs.shape[-2:], [0.5], device=xs.device, upsample_ratio=1.0), None)]
            return m(xs, coord, t=[0.5 * x0.new_ones(1)], ds_factor=1.0)['imgt_pred'][0]
        return Net(fn, m, 32, fp16_ok=False)
    return load


def load_sgm():
    """SGM-VFI (Liu et al., CVPR 2024), the small model with sparse global matching (ours-1-2-points, 0.5 key points), as its demo_2x.py
    runs it for one middle frame (no test-time augmentation). Weights are loaded strictly (the authors' loader uses strict=False)."""
    def load():
        root = os.path.join(MODELS_DIR, 'SGM-VFI')
        _need(os.path.join('SGM-VFI', 'log', 'ours-1-2-points', 'ckpt', 'ours-1-2-points.pkl'),
              os.path.join('SGM-VFI', 'pretrained', 'gmflow_sintel-0c07dcb3.pth'))
        cwd = os.getcwd()
        os.chdir(root)          # the model loads its local branch and GMFlow from relative paths
        try:
            with sandbox(root):
                import config as cfg
                from Trainer_x4k import Model
                cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
                cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=16, depth=[2, 2, 2, 4], num_key_points=0.5)
                m = Model(-1)
                ck = torch.load(os.path.join(root, 'log', 'ours-1-2-points', 'ckpt', 'ours-1-2-points.pkl'), map_location='cpu', weights_only=True)
                sd = {k.replace('module.', ''): v for k, v in ck['model'].items()
                      if 'module.' in k and 'attn_mask' not in k and 'HW' not in k}
                m.net.load_state_dict(sd, strict=True)
                m.net.eval()
        finally:
            os.chdir(cwd)
        return Net(lambda x0, x1: m.inference(x0, x1, TTA=False), m.net, 32, fp16_ok=False)
    return load


def load_bimvfi():
    """BiM-VFI (Seo et al., CVPR 2025; code and weights for research and education only), as its demo runs it for one middle frame
    (pyr_level 3, feat_channels 32). Its package __init__ files pull in the training stack, so the two package names are stubbed and the
    model's own module is imported."""
    def load():
        import types
        root = os.path.join(MODELS_DIR, 'BiM-VFI')
        _need(os.path.join('BiM-VFI', 'pretrained', 'bim_vfi.pth'))
        with sandbox(root):
            for n, pth in (('modules', 'modules'), ('modules.components', os.path.join('modules', 'components'))):
                mod = types.ModuleType(n)
                mod.__path__ = [os.path.join(root, pth)]
                sys.modules[n] = mod
            from modules.components.bim_vfi import BiMVFI
            m = BiMVFI(pyr_level=3, feat_channels=32)
            ck = torch.load(os.path.join(root, 'pretrained', 'bim_vfi.pth'), map_location='cpu', weights_only=False)
            m.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()}, strict=True)
            m.eval()
        return Net(lambda x0, x1: m(x0, x1, 0.5)['imgt_pred'], m, 16, fp16_ok=False)
    return load


def load_ldmvfi():
    """LDMVFI (Danier et al., AAAI 2024; MIT), latent diffusion, DDIM 200 steps, eta 1 (the authors' evaluation setting), seeded. The training
    stack's pytorch_lightning.utilities.distributed (removed from current Lightning) is pointed at its rank_zero module; taming-transformers
    (MIT) is cloned beside the models for its VQ quantizer."""
    def load():
        from functools import partial
        root = os.path.join(MODELS_DIR, 'LDMVFI')
        ckpt = os.path.join(root, 'ckpt', 'ldmvfi-vqflow-f32-c256-concat_max.ckpt')
        _need(os.path.join('LDMVFI', 'ckpt', 'ldmvfi-vqflow-f32-c256-concat_max.ckpt'), 'taming-transformers')
        import pytorch_lightning.utilities.rank_zero as rz
        sys.modules['pytorch_lightning.utilities.distributed'] = rz
        with sandbox(root, os.path.join(MODELS_DIR, 'taming-transformers')):
            from omegaconf import OmegaConf
            from ldm.util import instantiate_from_config
            from ldm.models.diffusion.ddim import DDIMSampler
            cfg = OmegaConf.load(os.path.join(root, 'configs', 'ldm', 'ldmvfi-vqflow-f32-c256-concat_max.yaml'))
            m = instantiate_from_config(cfg.model)
            m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'])    # the authors' checkpoint (a Lightning pickle)
            m.eval()
            m.model_ema.copy_to(m.model)      # the EMA weights, once (ema_scope would keep a second copy on the 8 GB GPU)
            m.model_ema = None
            m.use_ema = False
        sampler = DDIMSampler(m)
        sample = partial(sampler.sample, S=200, eta=1.0, verbose=False)

        def fn(x0, x1):
            torch.manual_seed(0)
            xc = {'prev_frame': x0 * 2 - 1, 'next_frame': x1 * 2 - 1}
            c, pp, pn = m.get_learned_conditioning(xc)
            out = sample(conditioning=c, batch_size=1, shape=(m.channels, c.shape[2], c.shape[3]), x_T=None)[0]
            out = m.decode_first_stage(out, xc, pp, pn)
            return (out.clamp(-1, 1) + 1) / 2
        return Net(fn, m, 32, fp16_ok=False)
    return load


# name -> (loader, what it is: the weights' training set and the licence of code and weights)
REGISTRY = {
    'rife-4.26': (load_rife('4.26'), 'Practical-RIFE v4.26 (MIT)'),
    'rife-4.25': (load_rife('4.25'), 'Practical-RIFE v4.25 (MIT)'),
    'rife-4.22': (load_rife('4.22'), 'Practical-RIFE v4.22 (MIT)'),
    'ifrnet-l': (load_ifrnet('l'), 'IFRNet large, Vimeo90K (MIT)'),
    'ifrnet': (load_ifrnet('b'), 'IFRNet base, Vimeo90K (MIT)'),
    'emavfi': (load_emavfi('ours', 32, [2, 2, 2, 4, 4]), 'EMA-VFI, Vimeo90K (Apache-2.0)'),
    'emavfi-s': (load_emavfi('ours_small', 16, [2, 2, 2, 2, 2]), 'EMA-VFI small, Vimeo90K (Apache-2.0)'),
    'amt-g': (load_amt('amt-g', 'AMT-G.yaml', 'amt-g.pth'), 'AMT-G, Vimeo90K (CC BY-NC 4.0)'),
    'amt-l': (load_amt('amt-l', 'AMT-L.yaml', 'amt-l.pth'), 'AMT-L, Vimeo90K (CC BY-NC 4.0)'),
    'amt-s': (load_amt('amt-s', 'AMT-S.yaml', 'amt-s.pth'), 'AMT-S, Vimeo90K (CC BY-NC 4.0)'),
    'film': (load_film(), 'FILM (PyTorch port of Google FILM, Vgg-style, Vimeo90K; Apache-2.0)'),
    'vfimamba': (load_vfimamba('VFIMamba', 32, [2, 2, 2, 3, 3]), 'VFIMamba, Vimeo90K+X-Train (Apache-2.0)'),
    'gimm-r': (load_gimm('r'), 'GIMM-VFI-R, Vimeo90K (S-Lab License 1.0, non-commercial: measurement only)'),
    'gimm-f': (load_gimm('f'), 'GIMM-VFI-F, Vimeo90K (S-Lab License 1.0, non-commercial: measurement only)'),
    'sgm-vfi': (load_sgm(), 'SGM-VFI small, 1/2 points, X4K1000FPS (Apache-2.0)'),
    'bim-vfi': (load_bimvfi(), 'BiM-VFI (research and education only: measurement only)'),
    'ldmvfi': (load_ldmvfi(), 'LDMVFI, Vimeo90K (MIT code; the paper is CC BY-NC-SA)'),
    'vfimamba-s': (load_vfimamba('VFIMamba_S', 16, [2, 2, 2, 3, 3]), 'VFIMamba small (Apache-2.0)'),
}
DEFAULT_MODELS = ['rife-4.26', 'rife-4.25', 'rife-4.22', 'ifrnet-l', 'emavfi', 'amt-g', 'amt-l', 'film', 'vfimamba']


# --- running -------------------------------------------------------------------------------------------------------

def is_oom(e):
    """Out of memory, whichever way CUDA reports it (torch.OutOfMemoryError, or a RuntimeError/AcceleratorError, or a
    device that cannot take a new context for want of memory)."""
    m = str(e).lower()
    return isinstance(e, torch.OutOfMemoryError) or 'out of memory' in m or 'busy or unavailable' in m


def free_gpu():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def predict(net, a, c, fp16, counters):
    """The middle frame of a and c (uint8 (H,W,3)) as uint8, as the model gives it and gated: (y, y_gated, ms), ms the
    model time. On a GPU out of memory
    (it is shared with a training run) the frame is retried once, then the model moves to the CPU for good (fp32)."""
    h, w = a.shape[:2]
    ph, pw = (-h) % net.pad, (-w) % net.pad

    def prep(img):
        x = torch.from_numpy(img).to(net.dev).permute(2, 0, 1).float()[None] / 255.0
        return F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x

    def go():
        x0, x1 = prep(a), prep(c)
        cuda = net.dev == 'cuda'
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=cuda and fp16 and net.fp16_ok):
            y = net.fn(x0, x1)
        if cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        y = y.float()
        yg = gate(x0, x1, y)

        def out(v):
            return v[0, :, :h, :w].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        return out(y), out(yg), ms

    for attempt in range(2):
        try:
            return go()
        except Exception as e:
            if not is_oom(e) or net.dev == 'cpu':
                raise
            free_gpu()
            counters['oom_retry'] += 1
    if net.gpu_only:
        raise torch.OutOfMemoryError('GPU out of memory (--device cuda: no fallback to the CPU)')
    counters['to_cpu'] += 1
    net.to('cpu')
    free_gpu()
    return go()


def run_model(name, files, fp16, device, counters_all):
    """Per clip: [psnr per triplet]; the mean model time per frame (ms), the mean pixels per frame, the device it ran on."""
    loader, _ = REGISTRY[name]
    net = loader()
    counters = {'oom_retry': 0, 'to_cpu': 0}
    net.gpu_only = device == 'cuda'
    if device != 'cpu':
        try:
            net.to('cuda')
        except Exception as e:
            if not is_oom(e) or net.gpu_only:
                raise
            net.to('cpu')
            free_gpu()
            counters['to_cpu'] += 1
    scores, scores_g, times, pix = {}, {}, [], []
    t_start = time.time()
    try:
        for i, f in enumerate(files):
            clip = os.path.basename(f)[:-4]
            d = np.load(f)
            out, outg = [], []
            for a, m, c in zip(d['a'], d['m'], d['c']):
                y, yg, ms = predict(net, a, c, fp16, counters)
                out.append(psnr8(y, m))
                outg.append(psnr8(yg, m))
                times.append((ms, net.dev))
                pix.append(a.shape[0] * a.shape[1])
            scores[clip], scores_g[clip] = out, outg
            if (i + 1) % 10 == 0:
                print(f'  {name}: {i + 1}/{len(files)} clips, {time.time() - t_start:.0f} s, on {net.dev}', flush=True)
    finally:
        del net
        free_gpu()
    counters['frames_on_cpu'] = sum(1 for _, dv in times if dv == 'cpu')
    counters_all[name] = counters
    warm = min(3, len(times) - 1)
    # the time of the frames that ran on the GPU (if any), else of those on the CPU
    on = 'cuda' if any(dv == 'cuda' for _, dv in times[warm:]) else 'cpu'
    ms = float(np.mean([t for t, dv in times[warm:] if dv == on]))
    return scores, scores_g, ms, float(np.mean(pix)), on


def time_models(names, files, iters=30):
    """Median ms per frame on the GPU for one 288x512 pair (a real frame, padded or cropped to that), fp32 and fp16; -1 when the
    model does not fit in the GPU memory that is free (the card is shared with a training run)."""
    d = np.load(files[0])
    a, c = d['a'][0], d['c'][0]
    reps = (-(-288 // a.shape[0]), -(-512 // a.shape[1]), 1)
    a, c = np.tile(a, reps)[:288, :512].copy(), np.tile(c, reps)[:288, :512].copy()
    out = {}
    for name in names:
        out[name] = {}
        try:
            net = REGISTRY[name][0]()
            net.to('cuda')
        except Exception as e:
            if not is_oom(e):
                raise
            out[name] = {'fp32': -1.0, 'fp16': -1.0}
            free_gpu()
            continue
        for tag, half in (('fp32', False), ('fp16', True)):
            if half and not net.fp16_ok:
                out[name][tag] = None
                continue
            try:
                ms = []
                for i in range(iters + 3):
                    ms.append(predict(net, a, c, half, {'oom_retry': 0, 'to_cpu': 0})[2])
                    if net.dev != 'cuda':
                        raise torch.OutOfMemoryError('moved to the CPU')
                out[name][tag] = float(np.median(ms[3:]))
            except Exception as e:
                if not is_oom(e):
                    raise
                out[name][tag] = -1.0
                break
        del net
        free_gpu()
        print(f'{name.ljust(12)} fp32 {out[name].get("fp32")} ms   fp16 {out[name].get("fp16")} ms (288x512, median)', flush=True)
    return out


def baseline(files, kind):
    scores = {}
    for f in files:
        d = np.load(f)
        out = []
        for a, m, c in zip(d['a'], d['m'], d['c']):
            y = ((a.astype(np.float32) + c.astype(np.float32)) / 2).round().astype(np.uint8) if kind == 'blend' else a
            out.append(psnr8(y, m))
        scores[os.path.basename(f)[:-4]] = out
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(DEFAULT_MODELS), help='comma list, or "all"; --list shows the names')
    ap.add_argument('--json', default=OUT_JSON)
    ap.add_argument('--per-game', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='only the first N clips (a smoke test; nothing is cached)')
    ap.add_argument('--fp16', action='store_true', help='autocast to fp16 where the model allows (default fp32)')
    ap.add_argument('--stride', type=int, default=1, help='every Nth clip only (a subset for slow models; nothing is cached)')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                    help='auto: the GPU, falling back to the CPU on OOM; cuda: the GPU only (an error if it does not fit); cpu')
    ap.add_argument('--threads', type=int, default=0, help='CPU threads (0: torch default)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--time', action='store_true', help='only time the models on the GPU (288x512, fp32 and fp16), no scoring')
    ap.add_argument('--triplets', default=TRIPLETS_DIR)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.list:
        for k, (_, what) in REGISTRY.items():
            print(k.ljust(12), what)
        return 0
    files = sorted(glob.glob(os.path.join(args.triplets, '*.npz')))
    if args.limit:
        files = files[:args.limit]
    if args.stride > 1:
        files = files[::args.stride]
    partial = bool(args.limit or args.stride > 1)
    names = list(REGISTRY) if args.models == 'all' else [s for s in args.models.split(',') if s]
    os.makedirs(CACHE_DIR, exist_ok=True)
    if args.time:
        json.dump(time_models(names, files), open(os.path.join(os.path.dirname(args.json), 'bench-public-times.json'), 'w'), indent=1)
        return 0
    tag = 'fp16' if args.fp16 else 'fp32'
    scores = {'blend': baseline(files, 'blend'), 'repeat': baseline(files, 'repeat')}
    info, failed, counters = {}, {}, {}
    for name in names:
        if name not in REGISTRY:
            failed[name] = 'unknown model name'
            continue
        cache = os.path.join(CACHE_DIR, f'{name}-{tag}.json')
        got = None
        if not args.no_cache and os.path.exists(cache):
            c = json.load(open(cache))
            want = [os.path.basename(f)[:-4] for f in files]
            if all(k in c['scores'] for k in want):       # a full run also serves any subset of its clips
                got = ({k: c['scores'][k] for k in want}, {k: c['scores_gate'][k] for k in want}, c['ms'], c['pixels'],
                       c['device'])
                counters[name] = c.get('fallbacks', {})
                print(f'{name}: from cache')
        if got is None:
            print(f'{name}: running', flush=True)
            try:
                got = run_model(name, files, args.fp16, args.device, counters)
            except Exception as e:     # a model that cannot be loaded or run is reported, never quietly dropped
                failed[name] = f'{type(e).__name__}: {str(e)[:200]}'
                print(f'{name}: FAILED, {failed[name]}', flush=True)
                free_gpu()
                continue
            if not partial:
                json.dump(dict(scores=got[0], scores_gate=got[1], ms=got[2], pixels=got[3], device=got[4],
                               fallbacks=counters[name]), open(cache, 'w'))
        scores[name], scores[name + '+gate'], ms, px, dev = got
        info[name] = dict(ms_per_frame=ms, mean_pixels=px, precision=tag if dev == 'cuda' else 'fp32', device=dev)
    games = {}
    for k in scores:
        for name, v in scores[k].items():
            games.setdefault(name.rsplit('_', 1)[0], {}).setdefault(k, []).extend(v)
    ks = list(scores)
    if args.per_game:
        print('game'.ljust(22), ' '.join(k[:14].rjust(14) for k in ks))
        for g in sorted(games):
            print(g.ljust(22), ' '.join(f'{np.mean(games[g][k]):14.2f}' for k in ks))
    n_trip = sum(len(v) for v in scores['blend'].values())
    print(f'{len(files)} unseen clips of {len(games)} games, {n_trip} triplets (middle frame, full frames, {tag})')
    result = {}
    for k in ks:
        macro = float(np.mean([np.mean(games[g][k]) for g in games]))
        micro = float(np.mean([x for v in scores[k].values() for x in v]))
        med = float(np.median([np.mean(v) for v in scores[k].values()]))
        result[k] = dict(macro=macro, micro=micro, median_clip=med, **info.get(k, {}))
        ms = f'   {info[k]["ms_per_frame"]:7.1f} ms/frame on {info[k]["device"]}' if k in info else ''
        print(f'{k.ljust(16)} games-average {macro:6.2f} dB   triplets-average {micro:6.2f}   median clip {med:6.2f}{ms}')
    for k, why in failed.items():
        print(f'{k.ljust(16)} NOT RUN: {why}')
    for k, c in counters.items():
        if c and (c.get('oom_retry') or c.get('to_cpu')):
            print(f'{k}: GPU out of memory: {c.get("oom_retry", 0)} retries, moved to the CPU {c.get("to_cpu", 0)} time(s), '
                  f'{c.get("frames_on_cpu", 0)} frames run there (fp32)')
    if not args.limit:
        json.dump(dict(subset=f'every {args.stride}th clip' if args.stride > 1 else 'all', result=result, failed=failed, fallbacks=counters,
                       games={g: {k: float(np.mean(v)) for k, v in games[g].items()} for g in games}),
                  open(args.json, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
