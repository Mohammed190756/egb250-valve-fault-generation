"""Per-window (1 s) crank-angle origin from the stage pressure of the stage WITHOUT the faulty valve:
2S faults (P2-P9) -> Pr1; 1S faults (P10-P17) -> Pr3; healthy -> Pr1 (healthy template = mean healthy profile).
Quality q = normalised residual after the shift (0 = perfect)."""
import numpy as np, pandas as pd
import os; F = os.environ.get('EGB_FEAT', '../features/')
r = pd.read_csv(F + 'recordings.csv'); v = r[(r.campaign == 'Valve') & (r.P <= 17)]
z = np.load(F + 'angle_profiles.npz', allow_pickle=True); P = z['prof']; idx = {x: i for i, x in enumerate(z['records'])}; ch = list(z['channels'])
K = np.arange(1, 7); GRID = np.arange(-180, 180, 0.1); PH = np.exp(1j * np.outer(GRID * np.pi / 180, K))
def spec(x): x = (x - x.mean()) / x.std(); return np.fft.rfft(x)[K]
T = {c: spec(np.stack([P[idx[x], :, ch.index(c)].mean(0) for x in v[v.P == 1].record]).mean(0)) for c in ('Pr1', 'Pr3')}
rows = []
for _, x in v.iterrows():
    c = 'Pr3' if x.P >= 10 else 'Pr1'
    for w in range(9):
        Ff = spec(P[idx[x.record], w, ch.index(c)]); cost = (np.abs(T[c][None] - Ff[None] * PH) ** 2).sum(1); k = np.argmin(cost)
        rows.append(dict(P=x.P, R=x.R, w=w, ref=c, off=GRID[k], q=cost[k] / (np.abs(T[c]) ** 2).sum()))
o = pd.DataFrame(rows); o.to_csv('res/offsets_win.csv', index=False)
g = o.groupby(['P', 'R']).off.agg(lambda s: np.ptp(s))
print('recordings with within-recording offset range > 5 deg:'); print(g[g > 5].to_string())
print(o.groupby('P').q.describe()[['mean', '50%', 'max']].round(3).to_string())
