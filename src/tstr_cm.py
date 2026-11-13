"""TSTR confusion matrices: logistic regression (as in evalsuite) trained on real healthy training revolutions + generated faults,
tested on real test revolutions of all 17 conditions. Also TRTR reference (trained on real training revolutions)."""
import sys, numpy as np, torch, importlib; sys.path.insert(0, '..')
import gcommon as gc
from evalsuite import X, P, R, TR, TE, CONDS, pz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
def generated(modname, cls, ckpt):
    M = importlib.import_module(modname); net = getattr(M, cls)(); net.load_state_dict(torch.load(ckpt)); net.eval()
    Ht = torch.tensor(gc.norm(X[TE & (P == 1)])); G = {}
    for p in CONDS:
        v, m = gc.VAL[p]; out = []
        for j in range(0, len(Ht), 256):
            xb = Ht[j:j + 256]; out.append(net.generate(xb, torch.full((len(xb),), v), torch.full((len(xb),), m)).numpy())
        G[p] = gc.denorm(np.concatenate(out))
    return G
def cm(G):
    Xs = np.concatenate([pz(X[TR & (P == 1)])] + [pz(G[p]) for p in CONDS]); ys = np.concatenate([np.ones((TR & (P == 1)).sum())] + [np.full(len(G[p]), p) for p in CONDS])
    sc = StandardScaler().fit(Xs); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xs), ys)
    yp = clf.predict(sc.transform(pz(X[TE]))); yt = P[TE]; return confusion_matrix(yt, yp, labels=list(range(1, 18))), float((yp == yt).mean())
out = {}
H = X[TE & (P == 1)].astype(np.float32); muH = X[TR & (P == 1)].astype(np.float32).mean(0)
out['B0'] = cm({p: H + (X[TR & (P == p)].astype(np.float32).mean(0) - muH) for p in CONDS})
out['REAL'] = cm({p: X[TR & (P == p)].astype(np.float32) for p in CONDS})
print('B0', out['B0'][1], 'REAL', out['REAL'][1], flush=True)
for mod, cls, ck, name in [('m_fm', 'FM', 'runs/fm.pt', 'FlowMatching'), ('m_bridge', 'Bridge', 'runs/bridge.pt', 'Bridge')]:
    out[name] = cm(generated(mod, cls, ck)); print(name, out[name][1], flush=True)
np.savez('res/tstr_confusion.npz', **{k: v[0] for k, v in out.items()}, acc=np.array([out[k][1] for k in out]), names=np.array(list(out)))
