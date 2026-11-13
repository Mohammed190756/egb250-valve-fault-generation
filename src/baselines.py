import numpy as np, pandas as pd, sys; sys.path.insert(0, '..')
from evalsuite import *
H_te = X[TE & (P == 1)].astype(np.float32); muH = X[TR & (P == 1)].astype(np.float32).mean(0)
n = len(H_te); res = []; rows = []
# B0 mean-difference translation of fresh healthy revolutions
G = {p: H_te + (X[TR & (P == p)].astype(np.float32).mean(0) - muH) for p in CONDS}
df, s = evaluate(G, 'B0 mean difference'); res.append(s); df.to_csv('res/eval_B0.csv')
# Reference: real training revolutions of the condition (upper bound for realism, memorisation by construction)
G = {p: X[TR & (P == p)][:n].astype(np.float32) for p in CONDS}
df, s = evaluate(G, 'REF real train (copy)'); res.append(s); df.to_csv('res/eval_REFtrain.csv')
# Reference: untouched fresh healthy (no fault generated)
G = {p: H_te for p in CONDS}
df, s = evaluate(G, 'REF healthy (no change)'); res.append(s)
S = pd.DataFrame(res); S.to_csv('res/eval_summary_baselines.csv', index=False); print(S.round(3).to_string())
