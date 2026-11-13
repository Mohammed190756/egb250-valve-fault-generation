"""Shared pieces for all generative models: data, conditioning, circular 1-D conv backbone, logging."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, time, os
torch.set_num_threads(2)
D = np.load('../ds180.npz'); X = D['X']; P = D['P']; R = D['R']; MU = D['mu'].astype(np.float32); SD = D['sd'].astype(np.float32)
VAL = {**{p: (3, p - 2) for p in range(2, 6)}, **{p: (2, p - 6) for p in range(6, 10)}, **{p: (1, p - 10) for p in range(10, 14)}, **{p: (0, p - 14) for p in range(14, 18)}}
# valve index: 0 1S_IVb, 1 1S_DV, 2 2S_IV, 3 2S_DV ; mode index 0..3 = modes 1..4
def norm(a): return ((a.astype(np.float32) - MU) / SD).reshape(len(a), 32, 180)
def denorm(z): return (z.reshape(len(z), 2, 16, 180) * SD + MU)
SPLIT = {'train': R <= 8, 'val': (R >= 9) & (R <= 10), 'test': R >= 11}
def arrays(split, exclude=()):
    m = SPLIT[split]; H = norm(X[m & (P == 1)]); fi = np.flatnonzero(m & (P >= 2) & ~np.isin(P, list(exclude)))
    Fx = norm(X[fi]); v = np.array([VAL[p][0] for p in P[fi]]); md = np.array([VAL[p][1] for p in P[fi]]); return H, Fx, v, md, P[fi]
class CircConv(nn.Module):
    def __init__(s, ci, co, k=7, d=1):
        super().__init__(); s.p = (k - 1) // 2 * d; s.c = nn.Conv1d(ci, co, k, dilation=d)
    def forward(s, x): return s.c(F.pad(x, (s.p, s.p), mode='circular'))
class Block(nn.Module):
    def __init__(s, c, d=1, cond=0):
        super().__init__(); s.n1 = nn.GroupNorm(8, c); s.c1 = CircConv(c, c, 7, d); s.n2 = nn.GroupNorm(8, c); s.c2 = CircConv(c, c, 7, 1)
        s.film = nn.Linear(cond, 2 * c) if cond else None
    def forward(s, x, e=None):
        h = s.c1(F.silu(s.n1(x)))
        if s.film is not None: g, b = s.film(e).chunk(2, -1); h = h * (1 + g[..., None]) + b[..., None]
        return x + s.c2(F.silu(s.n2(h)))
class CondEmb(nn.Module):
    """Valve and mode embedded separately and summed (allows unseen valve-mode combinations)."""
    def __init__(s, d=64):
        super().__init__(); s.v = nn.Embedding(4, d); s.m = nn.Embedding(4, d)
    def forward(s, v, m): return s.v(v) + s.m(m)
class Logger:
    def __init__(s, name): s.name = name; s.rows = []; os.makedirs('../runs', exist_ok=True); s.t0 = time.time()
    def log(s, **k): k['t'] = time.time() - s.t0; s.rows.append(k); json.dump(s.rows, open(f'../runs/{s.name}.json', 'w')); print(k, flush=True)
def batches(n, bs, rng):
    i = rng.permutation(n); return [i[j:j + bs] for j in range(0, n, bs)]
def ck_load(name, objs, L, rng):
    """Resume training after a container restart: restores model, optimiser and scheduler states, log rows, RNG states and early-stopping counters."""
    f = f'../runs/{name}.ckpt'
    if not os.path.exists(f): return 0, 1e18, 0
    st = torch.load(f, weights_only=False)
    for k, o in objs.items(): o.load_state_dict(st[k])
    L.rows = st['rows']; rng.bit_generator.state = st['rng']; torch.set_rng_state(st['trng']); print('resumed', name, 'at epoch', st['ep'] + 1, flush=True)
    return st['ep'] + 1, st['best'], st['bad']
def ck_save(name, objs, L, rng, ep, best, bad):
    f = f'../runs/{name}.ckpt'
    torch.save({**{k: o.state_dict() for k, o in objs.items()}, 'rows': L.rows, 'rng': rng.bit_generator.state, 'trng': torch.get_rng_state(), 'ep': ep, 'best': best, 'bad': bad}, f + '.tmp'); os.replace(f + '.tmp', f)
