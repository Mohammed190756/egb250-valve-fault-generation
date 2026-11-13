"""Performance matrices: (1) generator x metric summary matrix, (2) generator x fault matrices of the per-fault errors."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
M = [('B0', 'Mean-difference baseline'), ('Bridge', 'Diffusion bridge'), ('FlowMatching', 'Flow matching'), ('CVAE', 'CVAE'), ('DiT', 'DiT'),
     ('NormFlow', 'Normalising flow'), ('Diffusion', 'DDPM (CNN)'), ('StarGANv2', 'StarGAN v2')]
S = pd.concat([pd.read_csv('res/eval_summary_baselines.csv')] + [pd.read_csv(f'res/evalsum_{k}.csv') for k, _ in M if k != 'B0']).set_index('model')
S = S.rename(index={'B0 mean difference': 'B0'})
cols = [('rmse_dB', 'Mean error\n(dB)', 1), ('FD', 'Fréchet\ndistance', 1), ('MMD2', 'MMD²', 1), ('abs_dL', '|ΔL|\n(dB)', 1), ('abs_dTheta', '|Δθ|\n(°)', 1), ('TSTR', 'TSTR\naccuracy', -1), ('DCR', 'DCR\nratio', 0)]
V = np.array([[S.loc[k, c] for c, _, _ in cols] for k, _ in M])
# colour = rank among generators per column (dark = best); DCR scored by |log ratio|
score = np.zeros_like(V)
for j, (c, _, d) in enumerate(cols):
    x = V[:, j] * d if d else np.abs(np.log(V[:, j]))
    score[:, j] = (x.argsort().argsort()) / (len(x) - 1)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.imshow(1 - score, cmap=SEQ, aspect='auto', vmin=0, vmax=1)
for i in range(len(M)):
    for j, (c, _, _) in enumerate(cols):
        v = V[i, j]; t = f'{v:.0f}' if c == 'FD' else (f'{v:.3f}' if c in ('MMD2', 'TSTR') else f'{v:.2f}')
        ax.text(j, i, t, ha='center', va='center', fontsize=8, color='white' if score[i, j] < 0.35 else '#0b0b0b', fontweight='bold' if score[i, j] == 0 else 'normal')
ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l, _ in cols]); ax.xaxis.tick_top()
ax.set_yticks(range(len(M))); ax.set_yticklabels([l for _, l in M])
ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(M)), minor=True); ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='both', length=0)
fig.text(0.01, 0.01, 'Median over 16 faults. Shading = rank per column (darker = better); DCR ranked by closeness to 1.', fontsize=7.5, color='#52514e')
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('fig/Fig_performance_matrix.png', dpi=600); fig.savefig('fig/Fig_performance_matrix.pdf')
# per-fault matrices
LAB = {**{p: f'2S_DV-m{p-1}' for p in range(2, 6)}, **{p: f'2S_IV-m{p-5}' for p in range(6, 10)}, **{p: f'1S_DV-m{p-9}' for p in range(10, 14)}, **{p: f'1S_IVb-m{p-13}' for p in range(14, 18)}}
fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.4))
for ax, (c, lab, vmax) in zip(axs, [('rmse_dB', '(a) Mean-signature error (dB)', 4.5), ('dL_dB', '(b) Opening-level error |ΔL| (dB)', 10), ('dTheta', '(c) Opening-angle error |Δθ| (°)', 8)]):
    A = np.array([np.abs(pd.read_csv(f'res/eval_{k}.csv').set_index('P').loc[range(2, 18), c].values) for k, _ in M])
    im = ax.imshow(A, cmap=SEQ, aspect='auto', vmin=0, vmax=vmax); cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.ax.tick_params(labelsize=7)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]): ax.text(j, i, f'{A[i, j]:.1f}', ha='center', va='center', fontsize=6.5, color='white' if A[i, j] > 0.6 * vmax else '#0b0b0b')
    ax.set_yticks(range(len(M))); ax.set_yticklabels([l for _, l in M], fontsize=7.5)
    ax.set_xticks(range(16)); ax.set_xticklabels([f'P{p}\n' + LAB[p].replace('-', '\n') for p in range(2, 18)], fontsize=6.5)
    ax.set_title(lab, loc='left', fontsize=9, fontweight='bold'); ax.tick_params(length=0)
fig.text(0.01, 0.005, 'Absolute error per fault on the held-out test recordings; darker = larger error. m1 seat wear, m2 plate corrosion, m3 plate fracture, m4 spring breakage.', fontsize=7.5, color='#52514e')
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig('fig/Fig_perfault_matrices.png', dpi=600); fig.savefig('fig/Fig_perfault_matrices.pdf')
print('ok')
