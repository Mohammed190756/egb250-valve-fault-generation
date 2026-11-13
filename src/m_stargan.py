"""StarGAN v2-style translator (Choi et al., 2020) adapted to crank-angle maps:
mapping network F(z, d) -> style s, style encoder E(x, d) -> s, generator G(x, s), discriminator D(x, d);
domains d = healthy or (valve, mode) with valve and mode embedded separately (projection-type domain conditioning in D).
Objective (Choi et al. Eq. 5): min_{G,F,E} max_D L_adv + l_sty L_sty - l_ds L_ds + l_cyc L_cyc, with L1 style, diversity and cycle terms."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
from gcommon import *
C = 64; SD_ = 64; ZD = 16; TAU = 5.0
class DomEmb(nn.Module):
    def __init__(s): super().__init__(); s.c = CondEmb(64); s.h = nn.Parameter(torch.randn(64) * 0.1)
    def forward(s, v, m, healthy=False): return s.h.expand(len(v), -1) if healthy else s.c(v, m)
class Gen(nn.Module):
    def __init__(s):
        super().__init__(); s.stem = CircConv(32, C, 7); s.b = nn.ModuleList([Block(C, d, SD_) for d in (1, 2, 4, 8, 1, 2)]); s.out = CircConv(C, 32, 7)
    def forward(s, x, st):
        h = s.stem(x)
        for b in s.b: h = b(h, st)
        return x + TAU * torch.tanh(s.out(F.silu(h)) / TAU)         # our choice: residual bounded to +/- TAU standardised units, so the diversity term cannot grow without bound
class Mapping(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(ZD + 64, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, SD_))
    def forward(s, z, d): return s.f(torch.cat([z, d], -1))
class Feat(nn.Module):
    def __init__(s, out):
        super().__init__(); s.stem = CircConv(32, C, 7); s.b = nn.ModuleList([Block(C, d, 0) for d in (1, 2, 4, 8)]); s.o = nn.Linear(C + 64, out)
    def forward(s, x, d):
        h = s.stem(x)
        for b in s.b: h = b(h)
        return s.o(torch.cat([F.silu(h).mean(-1), d], -1))
class StarGAN(nn.Module):
    def __init__(s): super().__init__(); s.dom = DomEmb(); s.G = Gen(); s.F = Mapping(); s.E = Feat(SD_); s.D = Feat(1)
    @torch.no_grad()
    def generate(s, xh, v, m, seed=0):
        g = torch.Generator().manual_seed(seed); z = torch.randn(len(xh), ZD, generator=g); return s.G(xh, s.F(z, s.dom(v, m)))
def train(name='stargan', exclude=(), epochs=40, patience=8, bs=128, lr=1e-4, seed=0, l_sty=1.0, l_ds=1.0, l_cyc=1.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H, Fx, v, m, _ = arrays('train', exclude); Hv, Fv, vv, mv, _ = arrays('val', exclude)
    H, Fx, Hv, Fv = map(torch.tensor, (H, Fx, Hv, Fv)); v, m, vv, mv = map(lambda a: torch.tensor(a, dtype=torch.long), (v, m, vv, mv))
    net = StarGAN(); og = torch.optim.Adam(list(net.G.parameters()) + list(net.F.parameters()) + list(net.E.parameters()) + list(net.dom.parameters()), lr=lr, betas=(0.0, 0.99))
    od = torch.optim.Adam(net.D.parameters(), lr=lr, betas=(0.0, 0.99)); L = Logger(name); best = 1e18; bad = 0
    sp = F.softplus                                                            # log(1 + e^x): -log(sigmoid(a)) = softplus(-a)
    OBJ = dict(net=net, og=og, od=od); ep0, best, bad = ck_load(name, OBJ, L, rng)
    for ep in range(ep0, epochs):
        net.train(); acc = dict(d=0, adv=0, sty=0, ds=0, cyc=0); nb = 0; lds = l_ds * max(0.0, 1 - ep / epochs)   # diversity weight decayed linearly
        for b in batches(len(Fx), bs, rng):
            xh = H[rng.integers(0, len(H), len(b))]; xf = Fx[b]; dv = net.dom(v[b], m[b]); dh = net.dom(v[b], m[b], healthy=True)
            z1, z2 = torch.randn(len(b), ZD), torch.randn(len(b), ZD)
            with torch.no_grad(): fake = net.G(xh, net.F(z1, dv))
            ld = sp(-net.D(xf, dv.detach())).mean() + sp(net.D(fake, dv.detach())).mean() + sp(-net.D(xh, dh.detach())).mean()
            od.zero_grad(); ld.backward(); od.step()
            s1, s2 = net.F(z1, dv), net.F(z2, dv); f1, f2 = net.G(xh, s1), net.G(xh, s2)
            adv = sp(-net.D(f1, dv)).mean(); sty = (s1 - net.E(f1, dv)).abs().mean(); ds = (f1 - f2).abs().mean()
            cyc = (xh - net.G(f1, net.E(xh, dh))).abs().mean()
            lg = adv + l_sty * sty - lds * ds + l_cyc * cyc; og.zero_grad(); lg.backward(); og.step()
            for k, val in zip(acc, (ld, adv, sty, ds, cyc)): acc[k] += val.item()
            nb += 1
        net.eval()
        with torch.no_grad():                                                   # validation: distance of generated to real validation maps (condition means)
            g = torch.Generator().manual_seed(123); err = 0; n = 0
            for p in np.unique(v.numpy() * 4 + m.numpy()):
                sel = (vv * 4 + mv) == int(p)
                if sel.sum() == 0: continue
                xh = Hv[torch.randint(0, len(Hv), (int(sel.sum()),), generator=g)]; gen = net.generate(xh, vv[sel], mv[sel])
                err += ((gen.mean(0) - Fv[sel].mean(0)) ** 2).mean().item(); n += 1
            val = err / n
        L.log(epoch=ep, **{k: acc[k] / nb for k in acc}, val_mean_map_mse=val)
        if val < best - 1e-4: best = val; bad = 0; torch.save(net.state_dict(), f'../runs/{name}.pt')
        else:
            bad += 1
            if bad >= patience and ep >= 15: break
        ck_save(name, OBJ, L, rng, ep, best, bad)
if __name__ == '__main__':
    train()
