"""Diffusion (Schroedinger) bridge translator after I2SB (Liu et al., 2023) with f = 0:
X_0 = faulty revolution (target), X_1 = healthy revolution (source); q(X_t | X_0, X_1) = N(mu_t, Sigma_t) with
mu_t = sb2/(sb2 + s2) X_0 + s2/(sb2 + s2) X_1, Sigma_t = s2 sb2/(sb2 + s2) I (Eq. 11), s2 = int_0^t beta, sb2 = int_t^1 beta.
Network eps(X_t, t, c) trained on ||eps - (X_t - X_0)/sigma_t||^2 (Eq. 12); sampling by the Gaussian bridge posterior
q(X_s | X_0^hat, X_t), s < t, from X_1 = x_h (Algorithm 2)."""
import sys, numpy as np, torch, torch.nn as nn; sys.path.insert(0, '..')
from gcommon import *
from m_fm import VNet
NT = 1000; tt = torch.linspace(0, 1, NT + 1)
BETA = torch.cat([torch.linspace(1e-4, 0.3, NT // 2), torch.linspace(0.3, 1e-4, NT // 2)]) / NT * 20   # symmetric schedule (I2SB uses a symmetric beta)
S2 = torch.cat([torch.zeros(1), torch.cumsum(BETA, 0)]); SB2 = S2[-1] - S2
class Bridge(nn.Module):
    def __init__(s): super().__init__(); s.net = VNet()
    def loss(s, xf, xh, v, m):
        i = torch.randint(1, NT, (len(xf),)); s2, sb2 = S2[i][:, None, None], SB2[i][:, None, None]
        mu = sb2 / (sb2 + s2) * xf + s2 / (sb2 + s2) * xh; var = s2 * sb2 / (sb2 + s2)
        xt = mu + var.sqrt() * torch.randn_like(xf); target = (xt - xf) / s2.sqrt()
        return ((s.net(xt, i.float() / NT, v, m) - target) ** 2).mean()
    @torch.no_grad()
    def generate(s, xh, v, m, steps=50, seed=0):
        g = torch.Generator().manual_seed(seed); idx = torch.linspace(NT, 0, steps + 1).long(); x = xh.clone()
        for a, b in zip(idx[:-1], idx[1:]):
            ta = torch.full((len(x),), float(a) / NT); x0 = x - S2[a].sqrt() * s.net(x, ta, v, m)
            if b == 0: x = x0; break
            s2a, s2b = S2[a], S2[b]                                   # Brownian-bridge posterior between X_0^hat (time 0) and X_a
            mean = x0 + (s2b / s2a) * (x - x0); var = s2b * (s2a - s2b) / s2a
            x = mean + var.sqrt() * torch.randn(x.shape, generator=g)
        return x
def train(name='bridge', exclude=(), epochs=60, patience=8, bs=128, lr=1e-3, seed=0):
    from m_fm import train as _t
    import m_fm; m_fm.FM_ = None
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H, Fx, v, m, _ = arrays('train', exclude); Hv, Fv, vv, mv, _ = arrays('val', exclude)
    H, Fx, Hv, Fv = map(torch.tensor, (H, Fx, Hv, Fv)); v, m, vv, mv = map(lambda a: torch.tensor(a, dtype=torch.long), (v, m, vv, mv))
    net = Bridge(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    L = Logger(name); best = 1e18; bad = 0
    OBJ = dict(net=net, opt=opt, sched=sched); ep0, best, bad = ck_load(name, OBJ, L, rng)
    for ep in range(ep0, epochs):
        net.train(); tl = nb = 0
        for b in batches(len(Fx), bs, rng):
            xh = H[rng.integers(0, len(H), len(b))]; loss = net.loss(Fx[b], xh, v[b], m[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); nb += 1
        sched.step(); net.eval()
        with torch.no_grad():
            _st = torch.get_rng_state(); torch.manual_seed(123); vl = nb2 = 0
            for j in range(0, len(Fv), 512):
                xh = Hv[torch.randint(0, len(Hv), (len(Fv[j:j + 512]),))]; vl += net.loss(Fv[j:j + 512], xh, vv[j:j + 512], mv[j:j + 512]).item(); nb2 += 1
        torch.set_rng_state(_st)                                                  # validation noise must not reset the training noise stream
        L.log(epoch=ep, train_bridge_mse=tl / nb, val_bridge_mse=vl / nb2)
        if vl / nb2 < best - 1e-4: best = vl / nb2; bad = 0; torch.save(net.state_dict(), f'../runs/{name}.pt')
        else:
            bad += 1
            if bad >= patience and ep >= 15: break
        ck_save(name, OBJ, L, rng, ep, best, bad)
if __name__ == '__main__':
    train()
