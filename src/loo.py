"""Compositional hold-out: four (valve, mode) cells never seen in training, one per valve and one per mode.
Each held-out cell's valve and mode are both seen in other combinations, so generation tests composition, not recall.
Baseline: additive two-way model of the mean difference maps, D(v,m) = a(v) + b(m) + c, fitted on the 12 seen cells."""
import sys, numpy as np, pandas as pd, importlib; sys.path.insert(0, '..')
import gcommon as gc
from evalsuite import evaluate, X, P, R, TR, TE, CONDS
HOLD = (2, 7, 12, 17)        # 2S_DV m1, 2S_IV m2, 1S_DV m3, 1S_IVb m4
def additive():
    H = X[TE & (P == 1)].astype(np.float32); muH = X[TR & (P == 1)].astype(np.float32).mean(0)
    D = {p: X[TR & (P == p)].astype(np.float32).mean(0) - muH for p in CONDS}
    seen = [p for p in CONDS if p not in HOLD]; V = np.array([gc.VAL[p][0] for p in seen]); M = np.array([gc.VAL[p][1] for p in seen])
    A = np.zeros((len(seen), 1 + 3 + 3)); A[:, 0] = 1                       # intercept + valve (ref 0) + mode (ref 0) dummies
    for i, (v, m) in enumerate(zip(V, M)):
        if v: A[i, v] = 1
        if m: A[i, 3 + m] = 1
    Y = np.stack([D[p].ravel() for p in seen]); B = np.linalg.lstsq(A, Y, rcond=None)[0]
    G = {}
    for p in CONDS:
        v, m = gc.VAL[p]; a = np.zeros(7); a[0] = 1
        if v: a[v] = 1
        if m: a[3 + m] = 1
        G[p] = H + (a @ B).reshape(D[p].shape) if p in HOLD else H + D[p]
    return G
def tag(df, name):
    df = df.copy(); df['held_out'] = [int(p) in HOLD for p in df.index if True] if 'P' not in df else df['P'].isin(HOLD); df['model'] = name; return df
if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'additive':
        df, s = evaluate(additive(), 'Additive (LOO)'); df.to_csv('res/loo_eval_additive.csv')
    else:                                                    # e.g. python3 loo.py m_fm FM 40 6
        mod, cls, ep, pat = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); name = f'loo_{mod[2:]}'
        M = importlib.import_module(mod)
        import os
        if not os.path.exists(f'runs/{name}.done'):
            M.train(name=name, exclude=HOLD, epochs=ep, patience=pat); open(f'runs/{name}.done', 'w').close()
        import gen_eval; df, s = gen_eval.run(mod, cls, f'runs/{name}.pt', f'LOO_{cls}'); df.to_csv(f'res/loo_eval_{cls}.csv')
