"""Conditional normalizing flow p(x_f | x_h, c) (Dinh et al., 2017; Kingma and Dhariwal, 2018):
K steps of [actnorm (per channel), invertible 1x1 convolution over the 32 channel-band rows, affine coupling on half
of the channels], coupling networks = circular convolutions conditioned on x_h and the condition embedding.
log p(x) = log N(z; 0, I) + sum of log-determinants; trained by maximum likelihood."""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
from gcommon import *
K = 8; CH = 64
class ActNorm(nn.Module):
    def __init__(s, c): super().__init__(); s.b = nn.Parameter(torch.zeros(1, c, 1)); s.logs = nn.Parameter(torch.zeros(1, c, 1)); s.register_buffer('inited', torch.tensor(0))
    def forward(s, x):
        if not bool(s.inited):
            with torch.no_grad(): s.b.data = -x.mean((0, 2), keepdim=True); s.logs.data = -torch.log(x.std((0, 2), keepdim=True) + 1e-6); s.inited.fill_(1)
        return (x + s.b) * s.logs.exp(), s.logs.sum() * x.shape[2]
    def inverse(s, y): return y * (-s.logs).exp() - s.b
class Inv1x1(nn.Module):
    def __init__(s, c): super().__init__(); w = torch.linalg.qr(torch.randn(c, c))[0]; s.W = nn.Parameter(w)
    def forward(s, x): return torch.einsum('ij,bjl->bil', s.W, x), torch.slogdet(s.W)[1] * x.shape[2]
    def inverse(s, y): return torch.einsum('ij,bjl->bil', torch.inverse(s.W), y)
class Coupling(nn.Module):
    def __init__(s):
        super().__init__(); s.stem = CircConv(16 + 32, CH, 7); s.b = nn.ModuleList([Block(CH, d, 64) for d in (1, 4)]); s.out = CircConv(CH, 32, 7)
        nn.init.zeros_(s.out.c.weight); nn.init.zeros_(s.out.c.bias)
    def st(s, xa, xh, e):
        h = s.stem(torch.cat([xa, xh], 1))
        for b in s.b: h = b(h, e)
        sc, t = s.out(F.silu(h)).chunk(2, 1); return torch.tanh(sc), t           # bounded log-scale for stability
    def forward(s, x, xh, e):
        xa, xb = x.chunk(2, 1); sc, t = s.st(xa, xh, e); yb = xb * sc.exp() + t; return torch.cat([xa, yb], 1), sc.sum((1, 2))
    def inverse(s, y, xh, e):
        ya, yb = y.chunk(2, 1); sc, t = s.st(ya, xh, e); return torch.cat([ya, (yb - t) * (-sc).exp()], 1)
class Flow(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = CondEmb(64)
        s.steps = nn.ModuleList([nn.ModuleList([ActNorm(32), Inv1x1(32), Coupling()]) for _ in range(K)])
    def nll(s, x, xh, v, m):
        e = s.emb(v, m); ld = 0; z = x - xh                                      # model the change from the paired healthy revolution
        for an, iv, cp in s.steps:
            z, a = an(z); z, b = iv(z); z, c = cp(z, xh, e); ld = ld + a + b + c
        logp = -0.5 * (z ** 2).sum((1, 2)) - 0.5 * math.log(2 * math.pi) * z[0].numel()
        return -(logp + ld).mean() / z[0].numel()                               # nats per dimension
    def loss(s, xf, xh, v, m): return s.nll(xf, xh, v, m)
    @torch.no_grad()
    def generate(s, xh, v, m, temp=1.0, seed=0):
        g = torch.Generator().manual_seed(seed); e = s.emb(v, m); z = torch.randn(xh.shape, generator=g) * temp
        for an, iv, cp in reversed(s.steps): z = an.inverse(iv.inverse(cp.inverse(z, xh, e)))
        return xh + z
def train(name='flow', exclude=(), epochs=40, patience=6, bs=128, lr=5e-4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H, Fx, v, m, _ = arrays('train', exclude); Hv, Fv, vv, mv, _ = arrays('val', exclude)
    H, Fx, Hv, Fv = map(torch.tensor, (H, Fx, Hv, Fv)); v, m, vv, mv = map(lambda a: torch.tensor(a, dtype=torch.long), (v, m, vv, mv))
    net = Flow()
    with torch.no_grad(): b0 = rng.choice(len(Fx), 512, replace=False); net.loss(Fx[b0], H[rng.integers(0, len(H), 512)], v[b0], m[b0])   # data-dependent actnorm init
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    L = Logger(name); best = 1e18; bad = 0
    OBJ = dict(net=net, opt=opt, sched=sched); ep0, best, bad = ck_load(name, OBJ, L, rng)
    for ep in range(ep0, epochs):
        net.train(); tl = nb = 0
        for b in batches(len(Fx), bs, rng):
            xh = H[rng.integers(0, len(H), len(b))]; loss = net.loss(Fx[b], xh, v[b], m[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); nb += 1
        sched.step(); net.eval()
        with torch.no_grad():
            g = torch.Generator().manual_seed(123); vl = nb2 = 0
            for j in range(0, len(Fv), 512):
                xh = Hv[torch.randint(0, len(Hv), (len(Fv[j:j + 512]),), generator=g)]; vl += net.loss(Fv[j:j + 512], xh, vv[j:j + 512], mv[j:j + 512]).item(); nb2 += 1
        L.log(epoch=ep, train_nll_nats_per_dim=tl / nb, val_nll_nats_per_dim=vl / nb2)
        if vl / nb2 < best - 1e-4: best = vl / nb2; bad = 0; torch.save(net.state_dict(), f'../runs/{name}.pt')
        else:
            bad += 1
            if bad >= patience and ep >= 15: break
        ck_save(name, OBJ, L, rng, ep, best, bad)
if __name__ == '__main__':
    train()
