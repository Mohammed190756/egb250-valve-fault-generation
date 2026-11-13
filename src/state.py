"""Vibration-only thermodynamic state index s: mean of (1S_DV opening delay) and (2S_DV opening advance), from A1/A2 event timings.
Validated against the interstage pressure (pressure used only for validation)."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr
e = pd.read_csv('res/events_rec.csv')
piv = e.pivot_table(index=['P', 'R'], columns=['ch', 'ev'], values='thp')
h = piv.loc[1].mean()
d = piv - h
s = (d[('A1', 33)] + d[('A2', 34)] - d[('A1', 179)] - d[('A2', 178)]) / 4
S = s.rename('s').reset_index()
import os; F = os.environ.get('EGB_FEAT', '../features/')
r = pd.read_csv(F + 'recordings.csv'); w = pd.read_parquet(F + 'window_features.parquet')
pr2 = w.groupby('record').Pr2_mean.mean().reset_index(); r = r[r.campaign == 'Valve'].merge(pr2, on='record')[['P', 'R', 'Pr2_mean']]
S = S.merge(r, on=['P', 'R'], how='left')
m = S.Pr2_mean.notna(); rr, p = pearsonr(S.s[m], S.Pr2_mean[m]); b = np.polyfit(S.Pr2_mean[m] - S.Pr2_mean[S.P == 1].mean(), S.s[m], 1)
print(f'n={m.sum()} r={rr:.3f} p={p:.1e} slope={b[0]:.2f} deg/bar')
print(S.groupby('P')[['s', 'Pr2_mean']].mean().round(2).T.to_string())
S.to_csv('res/state_s.csv', index=False)
