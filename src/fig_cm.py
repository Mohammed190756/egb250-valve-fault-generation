import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
Z = np.load('res/tstr_confusion.npz'); names = list(Z['names']); acc = dict(zip(names, Z['acc']))
PAN = [('REAL', '(a) Trained on real faults (reference)'), ('B0', '(b) Mean-difference baseline'), ('FlowMatching', '(c) Flow matching'), ('Bridge', '(d) Diffusion bridge')]
LAB = ['H'] + [f'{p}' for p in range(2, 18)]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0))
for ax, (k, t) in zip(axs.flat, PAN):
    C = Z[k].astype(float); C = 100 * C / C.sum(1, keepdims=True)
    im = ax.imshow(C, cmap=SEQ, vmin=0, vmax=100)
    for i in range(17):
        for j in range(17):
            if C[i, j] >= 5: ax.text(j, i, f'{C[i, j]:.0f}', ha='center', va='center', fontsize=5.2, color='white' if C[i, j] > 60 else '#0b0b0b')
    ax.set_xticks(range(17)); ax.set_xticklabels(LAB, fontsize=6.5); ax.set_yticks(range(17)); ax.set_yticklabels(LAB, fontsize=6.5)
    for b in (0.5, 4.5, 8.5, 12.5): ax.axhline(b, color='#52514e', lw=0.5); ax.axvline(b, color='#52514e', lw=0.5)
    ax.set_xlabel('Predicted condition (P)'); ax.set_ylabel('True condition (P)')
    ax.set_title(f'{t}\naccuracy {100 * acc[k]:.1f}%', loc='left', fontsize=8.5, fontweight='bold')
fig.subplots_adjust(left=0.08, right=0.9, top=0.93, bottom=0.07, hspace=0.38, wspace=0.3)
cax = fig.add_axes([0.92, 0.3, 0.015, 0.4]); cb = fig.colorbar(im, cax=cax); cb.set_label('Share of true-class test revolutions (%)', fontsize=8)
fig.savefig('fig/Fig_tstr_confusion.png', dpi=600); fig.savefig('fig/Fig_tstr_confusion.pdf'); print('ok', acc)
