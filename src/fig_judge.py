import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
Z = np.load('res/judge_cm.npz'); S = pd.read_csv('res/judge_summary.csv').set_index('key')
PAN = [('REAL', 'Real test faults (reference)'), ('B0', 'Mean-difference baseline'), ('Bridge', 'Diffusion bridge'), ('FlowMatching', 'Flow matching'), ('CVAE', 'CVAE'),
       ('DiT', 'DiT'), ('NormFlow', 'Normalising flow'), ('Diffusion', 'DDPM (CNN)'), ('StarGANv2', 'StarGAN v2')]
fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.9))
for ax, (k, t) in zip(axs.flat, PAN):
    C = Z[k].astype(float); C = 100 * C / C.sum(1, keepdims=True)
    im = ax.imshow(C, cmap=SEQ, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(17)); ax.set_xticklabels(['H'] + [str(p) for p in range(2, 18)], fontsize=5.5); ax.set_yticks(range(16)); ax.set_yticklabels([str(p) for p in range(2, 18)], fontsize=5.5)
    for b in (0.5, 4.5, 8.5, 12.5): ax.axvline(b, color='#52514e', lw=0.4)
    for b in (3.5, 7.5, 11.5): ax.axhline(b, color='#52514e', lw=0.4)
    ax.set_title(f'{t}\naccuracy {100 * S.loc[k, "Accuracy"]:.1f}%, F1 {S.loc[k, "F1"]:.3f}', loc='left', fontsize=7.5, fontweight='bold')
    ax.tick_params(length=0)
for ax in axs[-1]: ax.set_xlabel('Recognised by the judge (H, P2–P17)', fontsize=7)
for ax in axs[:, 0]: ax.set_ylabel('Requested fault (P)', fontsize=7)
fig.subplots_adjust(left=0.07, right=0.9, top=0.95, bottom=0.06, hspace=0.45, wspace=0.25)
cax = fig.add_axes([0.92, 0.3, 0.015, 0.4]); cb = fig.colorbar(im, cax=cax); cb.set_label('Share of requested revolutions (%)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.savefig('fig/Fig_judge_confusion.png', dpi=600); fig.savefig('fig/Fig_judge_confusion.pdf'); print('ok')
