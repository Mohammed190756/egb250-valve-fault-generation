"""Evaluation suite for generated faulty revolutions (log10 band-energy maps, 2 x 16 x 360)."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
D = np.load('../ds180.npz'); X = D['X']; P = D['P']; R = D['R']
TR = R <= 8; TE = R >= 11; CONDS = list(range(2, 18))   # train R1-R8 (R9-R10 validation), test R11-R15
OWN_OPEN = {**{p: (1, 178) for p in range(2, 6)}, **{p: (1, 312) for p in range(6, 10)}, **{p: (0, 33) for p in range(10, 14)}, **{p: (0, 141) for p in range(14, 18)}}
rng = np.random.default_rng(0)
def flat(a): return a.reshape(len(a), -1).astype(np.float32)
# PCA on a balanced subsample of real training revolutions (all 17 conditions)
sub = np.concatenate([rng.choice(np.flatnonzero(TR & (P == p)), 400, replace=False) for p in range(1, 18)])
PCA_ = PCA(64, random_state=0).fit(flat(X[sub]))
def pz(a): return PCA_.transform(flat(a))
def fd(a, b):
    m1, m2 = a.mean(0), b.mean(0); c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = sqrtm(c1 @ c2).real; return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cs))
def mmd2(a, b, n=500):
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]; b = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    Z = np.vstack([a, b]); d2 = ((Z[:, None] - Z[None]) ** 2).sum(-1); s = np.median(d2[d2 > 0]); K = np.exp(-d2 / s)
    na = len(a); Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
    return float((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1)) + (Kbb.sum() - np.trace(Kbb)) / (len(b) * (len(b) - 1)) - 2 * Kab.mean())
def event(a, p):
    c, th = OWN_OPEN[p]; E = (10 ** a[:, c].astype(np.float64)).sum(1)          # broadband energy per 2-degree bin
    w = np.arange(th // 2 - 5, th // 2 + 6) % 180; seg = E[:, w]; i = seg.argmax(1)
    return 10 * np.log10(seg.max(1)), 2.0 * w[i] + 1.0
def nn_dist(a, ref):
    a, ref = pz(a), pz(ref); d = ((a[:, None] - ref[None]) ** 2).sum(-1); return np.sqrt(d.min(1))
def evaluate(G, name, tstr=True):
    """G: dict cond -> generated maps (n,2,16,360). Returns per-condition metrics and summary."""
    out = {}
    for p in CONDS:
        g = G[p].astype(np.float32); real = X[TE & (P == p)].astype(np.float32); trn = X[TR & (P == p)]
        gz, rz = pz(g), pz(real)
        Lg, Tg = event(g, p); Lr, Tr_ = event(real, p)
        sel = rng.choice(len(trn), min(800, len(trn)), replace=False)
        dg = nn_dist(g[:300], trn[sel]); dr = nn_dist(real[:300], trn[sel])
        out[p] = dict(rmse_dB=float(10 * np.sqrt(((g.mean(0) - real.mean(0)) ** 2).mean())), FD=fd(gz, rz), MMD2=mmd2(gz, rz),
                      dL_dB=float(Lg.mean() - Lr.mean()), dTheta=float(Tg.mean() - Tr_.mean()), DCR_ratio=float(np.median(dg) / np.median(dr)))
    if tstr:
        Xs = np.concatenate([pz(X[TR & (P == 1)])] + [pz(G[p]) for p in CONDS]); ys = np.concatenate([np.ones((TR & (P == 1)).sum())] + [np.full(len(G[p]), p) for p in CONDS])
        Xt = pz(X[TE]); yt = P[TE]
        sc = StandardScaler().fit(Xs); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xs), ys)
        acc = float((clf.predict(sc.transform(Xt)) == yt).mean())
    else: acc = np.nan
    import pandas as pd
    df = pd.DataFrame(out).T; df.index.name = 'P'
    summ = dict(model=name, rmse_dB=df.rmse_dB.median(), FD=df.FD.median(), MMD2=df.MMD2.median(), abs_dL=df.dL_dB.abs().median(), abs_dTheta=df.dTheta.abs().median(),
                DCR=df.DCR_ratio.median(), TSTR=acc)
    return df, summ
