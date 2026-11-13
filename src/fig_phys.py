"""Fig. 11: own-valve opening change (level, angle) of generated vs real faults, per fault, computed with the event definition of Eq. (6.4)."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evalsuite import X, P, R, TE, event, CONDS
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.linewidth': 0.6, 'legend.fontsize': 7.5})
VAL = {**{p: ('2S_DV', p - 1) for p in range(2, 6)}, **{p: ('2S_IV', p - 5) for p in range(6, 10)}, **{p: ('1S_DV', p - 9) for p in range(10, 14)}, **{p: ('1S_IVb', p - 13) for p in range(14, 18)}}
H = X[TE & (P == 1)].astype(np.float32); rows = []
for p in CONDS:
    Lr, Tr = event(X[TE & (P == p)].astype(np.float32), p); Lh, Th = event(H, p)
    rows.append(dict(P=p, dL=Lr.mean() - Lh.mean(), dT=Tr.mean() - Th.mean()))
real = pd.DataFrame(rows).set_index('P')
MODELS = [('res/eval_B0.csv', 'Mean-difference baseline', '#52514e', 's'), ('res/eval_FlowMatching.csv', 'Flow matching', '#eb6834', 'o'), ('res/eval_Bridge.csv', 'Diffusion bridge', '#2a78d6', 'D')]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, col, gcol, lab in ((axs[0], 'dL', 'dL_dB', 'opening level change $\\Delta L$ (dB)'), (axs[1], 'dT', 'dTheta', 'opening angle change $\\Delta\\theta$ (°)')):
    lim = np.abs(real[col]).max() * 1.25
    ax.plot([-lim, lim], [-lim, lim], color='#0b0b0b', lw=0.7, zorder=1); ax.axhline(0, color='#c3c2b7', lw=0.5); ax.axvline(0, color='#c3c2b7', lw=0.5)
    for f, name, c, mk in MODELS:
        try: e = pd.read_csv(f).set_index('P')
        except FileNotFoundError: continue
        ax.scatter(real[col], real[col] + e.loc[real.index, gcol], s=18, c=c, marker=mk, label=name, edgecolors='white', linewidths=0.4, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.grid(color='#e6e5e1', lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel('Real ' + lab); ax.set_ylabel('Generated ' + lab)
axs[0].text(-0.2, 1.03, '(a)', transform=axs[0].transAxes, fontweight='bold'); axs[1].text(-0.2, 1.03, '(b)', transform=axs[1].transAxes, fontweight='bold')
axs[1].legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='none')
fig.tight_layout(); fig.savefig('fig/Fig_physical_fidelity.png', dpi=600); fig.savefig('fig/Fig_physical_fidelity.pdf')
real.to_csv('res/real_own_event_change.csv')
for f, name, c, mk in MODELS:
    e = pd.read_csv(f).set_index('P'); g = real.dL + e.loc[real.index, 'dL_dB']; gt = real.dT + e.loc[real.index, 'dTheta']
    print(name, 'corr L', np.corrcoef(real.dL, g)[0, 1].round(3), 'corr theta', np.corrcoef(real.dT, gt)[0, 1].round(3), 'sign agree theta', int((np.sign(gt) == np.sign(real.dT)).sum()))
