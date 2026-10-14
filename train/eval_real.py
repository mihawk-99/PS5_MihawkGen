# PS5_FrameGen - PSNR of checkpoints on the held-out real clips (games and scenes no training step saw), per game, against blend.
import random, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import psnr
from model import Interpolator, DEFAULT_BLOCKS
from real import Clips
held = Clips(held=True)
rows = {}
models = {}
for path in sys.argv[1:]:
    st = torch.load(path, map_location='cuda'); m = Interpolator([tuple(b) for b in st.get('blocks', DEFAULT_BLOCKS)]).cuda().eval(); m.load_state_dict(st['ema']); models[path] = m
print(len(held), 'held clips', flush=True)
for gi, (name, clip) in enumerate(zip(held.names, held.clips)):
    if gi % 2: continue
    one = Clips.__new__(Clips); one.clips = [clip]; one.names = [name]; one.new = [held.new[gi]]; one.weights = [1.0]
    got = one.sample(8, 256, 'cuda', random.Random(5), keep_still=0.0)
    if got is None or len(got[0]) < 4: continue
    a, c, g, t, _ = got
    res = {'blend': float(psnr(a * (1 - t.view(-1, 1, 1, 1)) + c * t.view(-1, 1, 1, 1), g).mean())}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for p, m in models.items():
            res[os.path.basename(os.path.dirname(p))] = float(psnr(m(a, c, t).float(), g).mean())
    game = name.rsplit('_', 1)[0]
    rows.setdefault(game, []).append(res)
for game, rs in rows.items():
    print(game.ljust(10), ' '.join(f'{k}={sum(r[k] for r in rs) / len(rs):.2f}' for k in rs[0]))
