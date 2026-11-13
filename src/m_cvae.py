"""Conditional VAE translator: q(z | x_f, x_h, c), p(x_f | z, x_h, c) with x_f = x_h + decoder residual.
ELBO = E_q[log p(x_f|z,x_h,c)] - KL(q(z|.) || N(0,I)) (Kingma and Welling, 2014; Sohn et al., 2015)."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
from gcommon import *
C = 64; ZD = 32
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.stem = CircConv(64, C, 7); s.b = nn.ModuleList([Block(C, d, 64) for d in (1, 2, 4, 8)]); s.out = nn.Linear(C, 2 * ZD)
    def forward(s, xf, xh, e):
        h = s.stem(torch.cat([xf, xh], 1))
        for b in s.b: h = b(h, e)
        return s.out(h.mean(-1)).chunk(2, -1)
class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.stem = CircConv(32, C, 7); s.zf = nn.Linear(ZD + 64, 2 * C); s.b = nn.ModuleList([Block(C, d, ZD + 64) for d in (1, 2, 4, 8, 1, 2)]); s.out = CircConv(C, 32, 7)
        s.logs = nn.Parameter(torch.zeros(32, 1))
    def forward(s, z, xh, e):
        c = torch.cat([z, e], -1); h = s.stem(xh); g, b = s.zf(c).chunk(2, -1); h = h * (1 + g[..., None]) + b[..., None]
        for bl in s.b: h = bl(h, c)
        return xh + s.out(F.silu(h))
class CVAE(nn.Module):
    def __init__(s): super().__init__(); s.emb = CondEmb(64); s.enc = Enc(); s.dec = Dec()
    def loss(s, xf, xh, v, m):
        e = s.emb(v, m); mu, lv = s.enc(xf, xh, e); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        xr = s.dec(z, xh, e); ls = s.dec.logs
        nll = (0.5 * ((xf - xr) / ls.exp()) ** 2 + ls + 0.9189385).sum((1, 2)).mean()
        kl = (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).sum(1).mean(); return nll, kl
    @torch.no_grad()
    def generate(s, xh, v, m, sample=False, seed=0):
        g = torch.Generator().manual_seed(seed); e = s.emb(v, m); z = torch.randn(len(xh), ZD, generator=g); mu = s.dec(z, xh, e)
        return mu + s.dec.logs.exp() * torch.randn(mu.shape, generator=g) if sample else mu   # sample=True: draw from the Gaussian decoder, Eq. (5.6)
def train(name='cvae', exclude=(), epochs=40, patience=6, bs=128, lr=2e-3, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H, Fx, v, m, _ = arrays('train', exclude); Hv, Fv, vv, mv, _ = arrays('val', exclude)
    H, Fx, Hv, Fv = map(torch.tensor, (H, Fx, Hv, Fv)); v, m, vv, mv = map(lambda a: torch.tensor(a, dtype=torch.long), (v, m, vv, mv))
    net = CVAE(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); L = Logger(name); best = 1e18; bad = 0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    OBJ = dict(net=net, opt=opt, sched=sched); ep0, best, bad = ck_load(name, OBJ, L, rng)
    for ep in range(ep0, epochs):
        net.train(); tn = tk = nb = 0; beta = min(1.0, (ep + 1) / 10)          # KL warm-up over 10 epochs
        for b in batches(len(Fx), bs, rng):
            xh = H[rng.integers(0, len(H), len(b))]; nll, kl = net.loss(Fx[b], xh, v[b], m[b]); loss = nll + beta * kl
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tn += nll.item(); tk += kl.item(); nb += 1
        sched.step(); net.eval()
        with torch.no_grad():
            g = torch.Generator().manual_seed(1); vn = vk = 0; nb2 = 0
            for j in range(0, len(Fv), 512):
                xh = Hv[torch.randint(0, len(Hv), (len(Fv[j:j + 512]),), generator=g)]; a, k_ = net.loss(Fv[j:j + 512], xh, vv[j:j + 512], mv[j:j + 512]); vn += a.item(); vk += k_.item(); nb2 += 1
        val = (vn + vk) / nb2
        L.log(epoch=ep, train_nll=tn / nb, train_kl=tk / nb, beta=beta, val_nll=vn / nb2, val_kl=vk / nb2, val_negelbo=val)
        if val < best - 1e-3: best = val; bad = 0; torch.save(net.state_dict(), f'../runs/{name}.pt')
        else:
            bad += 1
            if bad >= patience and ep >= 12: break
        ck_save(name, OBJ, L, rng, ep, best, bad)
    return net
if __name__ == '__main__':
    train()
