"""Independent 'judge' classifier: a circular CNN trained ONLY on real revolutions (R1-R8; early stopping on R9-R10), 17 classes
(healthy + 16 single faults). It labels generated revolutions, so that the requested fault (true label) can be compared with the
recognised fault (predicted label) using standard supervised-learning metrics."""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
import gcommon as gc
from gcommon import CircConv, Block
D = np.load('../ds180.npz'); X, P, R = D['X'], D['P'], D['R']
class Judge(nn.Module):
    def __init__(s, C=48):
        super().__init__(); s.stem = CircConv(32, C, 7); s.b = nn.ModuleList([Block(C, d, 0) for d in (1, 2, 4, 8)]); s.drop = nn.Dropout(0.2); s.out = nn.Linear(2 * C, 17)
    def forward(s, x):
        h = s.stem(x)
        for b in s.b: h = b(h)
        h = F.silu(h); return s.out(s.drop(torch.cat([h.mean(-1), h.amax(-1)], 1)))
def data(mask):
    return torch.tensor(gc.norm(X[mask])), torch.tensor(P[mask] - 1, dtype=torch.long)
def train(epochs=12, patience=3, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xt, yt = data(R <= 8); Xv, yv = data((R >= 9) & (R <= 10)); net = Judge(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    best, bad, log = 0, 0, []
    for ep in range(epochs):
        net.train(); idx = rng.permutation(len(Xt)); tl = 0
        for j in range(0, len(idx), 128):
            b = idx[j:j + 128]; loss = F.cross_entropy(net(Xt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
        net.eval()
        with torch.no_grad(): va = float((torch.cat([net(Xv[j:j + 512]).argmax(1) for j in range(0, len(Xv), 512)]) == yv).float().mean())
        log.append(dict(epoch=ep, train_ce=tl / len(Xt), val_acc=va)); print(log[-1], flush=True)
        if va > best + 1e-4: best, bad = va, 0; torch.save(net.state_dict(), 'runs/judge.pt')
        else:
            bad += 1
            if bad >= patience: break
    json.dump(log, open('runs/judge.json', 'w'))
@torch.no_grad()
def predict(maps_log10):   # maps in log10 energy units (n, 2, 16, 180)
    net = Judge(); net.load_state_dict(torch.load('runs/judge.pt')); net.eval()
    x = torch.tensor(gc.norm(maps_log10)); return np.concatenate([net(x[j:j + 512]).argmax(1).numpy() for j in range(0, len(x), 512)]) + 1
if __name__ == '__main__':
    if not os.path.exists('runs/judge.pt'): train()
