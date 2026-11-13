"""Crank-angle origin of every recording from the stage pressures (Pr1-Pr3), relative to the healthy template.
Continuous estimate: Fourier shift of the 72-bin crank-angle profiles, harmonics 1-6, grid 0.1 deg."""
import numpy as np, pandas as pd
import os; F = os.environ.get('EGB_FEAT', '../features/')
r = pd.read_csv(F + 'recordings.csv'); v = r[(r.campaign == 'Valve') & (r.P <= 17)].copy()
z = np.load(F + 'angle_profiles.npz', allow_pickle=True); P = z['prof']; idx = {x: i for i, x in enumerate(z['records'])}; ch = list(z['channels'])
K = np.arange(1, 7); GRID = np.arange(-180, 180, 0.1)
def spec(x): X = np.fft.rfft((x - x.mean()) / x.std()); return X[K]
def shift(Ft, Ff):   # delta (deg) such that f(theta + delta) best matches template
    ph = np.exp(1j * np.outer(GRID * np.pi / 180, K))
    cost = np.abs(Ft[None] - Ff[None] * ph) ** 2
    return GRID[np.argmin(cost.sum(1))]
prs = ['Pr1', 'Pr2', 'Pr3']; acc = ['A1', 'A2']
tmpl = {c: spec(np.stack([P[idx[x], :, ch.index(c)].mean(0) for x in v[v.P == 1].record]).mean(0)) for c in prs + acc}
rows = []
for _, x in v.iterrows():
    i = idx[x.record]; d = {'record': x.record, 'P': x.P, 'R': x.R}
    for c in prs + acc: d['d_' + c] = shift(tmpl[c], spec(P[i, :, ch.index(c)].mean(0)))
    Ft = np.concatenate([tmpl[c] for c in prs]); Ff = np.concatenate([spec(P[i, :, ch.index(c)].mean(0)) for c in prs])
    ph = np.exp(1j * np.outer(GRID * np.pi / 180, np.tile(K, 3))); d['offset'] = GRID[np.argmin((np.abs(Ft[None] - Ff[None] * ph) ** 2).sum(1))]
    w = [shift(np.concatenate([tmpl[c] for c in prs]) if False else tmpl['Pr2'], spec(P[i, k, ch.index('Pr2')])) for k in range(9)]
    d['win_range_Pr2'] = np.ptp(w); rows.append(d)
o = pd.DataFrame(rows).sort_values(['P', 'R']); o.to_csv('../res/offsets.csv', index=False)
print(o.groupby('P')[['offset', 'd_Pr1', 'd_Pr2', 'd_Pr3', 'd_A1', 'd_A2', 'win_range_Pr2']].agg(['mean', 'min', 'max']).round(1).to_string())
h = o[o.P == 1]; print('healthy: corr(offset, d_A2) =', np.corrcoef(h.offset, h.d_A2)[0, 1].round(2), ' corr(offset,d_A1)=', np.corrcoef(h.offset, h.d_A1)[0, 1].round(2))
print(h[['R', 'offset', 'd_A1', 'd_A2']].to_string())
