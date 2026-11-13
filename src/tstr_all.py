"""TSTR confusion matrices for all generators, using the saved generated faults (res/gen/*.npz) and the classifier of evalsuite (Section 6.6)."""
import sys, numpy as np; sys.path.insert(0, '..')
from evalsuite import X, P, R, TR, TE, CONDS, pz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
def cm(G):
    Xs = np.concatenate([pz(X[TR & (P == 1)])] + [pz(G[p]) for p in CONDS]); ys = np.concatenate([np.ones((TR & (P == 1)).sum())] + [np.full(len(G[p]), p) for p in CONDS])
    sc = StandardScaler().fit(Xs); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xs), ys)
    yp = clf.predict(sc.transform(pz(X[TE]))); yt = P[TE]; return confusion_matrix(yt, yp, labels=list(range(1, 18))), float((yp == yt).mean())
out = {'REAL': cm({p: X[TR & (P == p)].astype(np.float32) for p in CONDS})}
for k in ['B0', 'Bridge', 'FlowMatching', 'CVAE', 'DiT', 'NormFlow', 'Diffusion', 'StarGANv2']:
    Z = np.load(f'res/gen/{k}.npz'); out[k] = cm({p: Z[str(p)].astype(np.float32) for p in CONDS}); print(k, out[k][1], flush=True)
np.savez('res/tstr_confusion_all.npz', **{k: v[0] for k, v in out.items()}, acc=np.array([out[k][1] for k in out]), names=np.array(list(out)))
