"""Conditional flow matching / rectified flow translator (Lipman et al., 2023; Liu et al., 2023):
source X_0 = healthy revolution, target X_1 = faulty revolution of condition c, independent coupling;
X_t = t X_1 + (1 - t) X_0, loss ||(X_1 - X_0) - v_theta(X_t, t, c)||^2; generation by Euler integration of dZ = v dt from Z_0 = x_h."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
from gcommon import *
from m_diff import temb, C
class VNet(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = CondEmb(64); s.tm = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        s.stem = CircConv(32, C, 7); s.b = nn.ModuleList([Block(C, d, 64) for d in (1, 2, 4, 8, 1, 2)]); s.out = CircConv(C, 32, 7)
    def forward(s, x, t, v, m):
        e = s.emb(v, m) + s.tm(temb(t * 999)); h = s.stem(x)
        for b in s.b: h = b(h, e)
        return s.out(F.silu(h))
class FM(nn.Module):
    def __init__(s): super().__init__(); s.net = VNet()
    def loss(s, xf, xh, v, m):
        t = torch.rand(len(xf)); xt = t[:, None, None] * xf + (1 - t[:, None, None]) * xh
        return ((xf - xh - s.net(xt, t, v, m)) ** 2).mean()
    @torch.no_grad()
    def generate(s, xh, v, m, steps=20):
        z = xh.clone(); dt = 1.0 / steps
        for i in range(steps): t = torch.full((len(z),), i * dt); z = z + dt * s.net(z, t, v, m)
        return z
def train(name='fm', exclude=(), epochs=60, patience=8, bs=128, lr=1e-3, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    H, Fx, v, m, _ = arrays('train', exclude); Hv, Fv, vv, mv, _ = arrays('val', exclude)
    H, Fx, Hv, Fv = map(torch.tensor, (H, Fx, Hv, Fv)); v, m, vv, mv = map(lambda a: torch.tensor(a, dtype=torch.long), (v, m, vv, mv))
    net = FM(); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
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
        L.log(epoch=ep, train_fm_mse=tl / nb, val_fm_mse=vl / nb2)
        if vl / nb2 < best - 1e-4: best = vl / nb2; bad = 0; torch.save(net.state_dict(), f'../runs/{name}.pt')
        else:
            bad += 1
            if bad >= patience and ep >= 15: break
        ck_save(name, OBJ, L, rng, ep, best, bad)
if __name__ == '__main__':
    train()
