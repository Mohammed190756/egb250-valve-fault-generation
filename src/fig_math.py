"""Figures for Section 4 (mathematical model). Journal style: Times-like serif, 8-9 pt labels, one-hue sequential maps,
blue-grey-red diverging maps (neutral midpoint), validated categorical palette for valves."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.fontsize': 7.5})
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec', '#f3a9a3', '#e34948', '#8f1f1f'])
VCOL = {'1S_IVb': '#2a78d6', '1S_DV': '#eb6834', '2S_IV': '#1baf7a', '2S_DV': '#eda100'}
EDGES = np.geomspace(250, 24000, 17); FC = np.sqrt(EDGES[:-1] * EDGES[1:]) / 1000
D = np.load('ds180.npz'); X = D['X']; P = D['P']; R = D['R']
VAL = {**{p: ('2S_DV', p - 1) for p in range(2, 6)}, **{p: ('2S_IV', p - 5) for p in range(6, 10)}, **{p: ('1S_DV', p - 9) for p in range(10, 14)}, **{p: ('1S_IVb', p - 13) for p in range(14, 18)}}
MODE = {1: 'seat wear', 2: 'plate corrosion', 3: 'plate fracture', 4: 'spring breakage'}
EVL = [(33, '1S_DV\nopens'), (107, '1S_DV, 2S_IV\nclose'), (141, '1S_IV\nopens'), (178, '2S_DV\nopens'), (285, '1S_IV,\n2S_DV close'), (312, '2S_IV\nopens')]
def mean_map(p): return X[P == p].astype(np.float64).mean(0)
def save(fig, n): fig.savefig(f'fig/{n}.png', dpi=600); fig.savefig(f'fig/{n}.pdf'); plt.close(fig); print(n)
ang = np.arange(180) * 2 + 1
# ---------- Fig. A: healthy mean maps with event annotations + broadband profile
H = mean_map(1)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12))
for j, ax in enumerate(axs):
    im = ax.pcolormesh(ang, np.arange(16), 10 * H[j], cmap=SEQ, shading='nearest', rasterized=True)
    ax.set_yticks(np.arange(0, 16, 3)); ax.set_yticklabels([f'{FC[k]:.1f}' for k in range(0, 16, 3)]); ax.set_ylabel('Band centre (kHz)')
    ax.set_ylim(-0.5, 15.5); ax.text(0.005, 0.96, f'({"ab"[j]}) A{j + 1}', transform=ax.transAxes, va='top', fontweight='bold', fontsize=9)
    for th, lab in EVL:
        ax.axvline(th, color='#0b0b0b', lw=0.6, ls=(0, (3, 2)))
        if j == 0: ax.text(th + {285: -6, 312: 7}.get(th, 0), 16.0, lab, ha={285: 'right', 312: 'left'}.get(th, 'center'), va='bottom', fontsize=7, linespacing=0.95)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03); cb.set_label('Band energy (dB re 1 (m/s²)²)', fontsize=7.5); cb.ax.tick_params(labelsize=7)
axs[-1].set_xlabel('Crank angle (°)'); axs[-1].set_xticks(range(0, 361, 45)); axs[-1].set_xlim(0, 360)
fig.subplots_adjust(top=0.88, left=0.08, right=0.93, bottom=0.09); save(fig, "Fig_healthy_map")
# ---------- Fig. B: fault signatures Delta_c for the four modes of each valve (A2 for 2S valves, A1 for 1S valves)
order = [('1S_IVb', 0), ('1S_DV', 0), ('2S_IV', 1), ('2S_DV', 1)]
fig, axs = plt.subplots(4, 4, figsize=(7.2, 6.4), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.18, wspace=0.06))
norm = TwoSlopeNorm(0, -15, 15)
for i, (v, ch) in enumerate(order):
    for m in range(1, 5):
        p = [q for q in VAL if VAL[q] == (v, m)][0]; d = 10 * (mean_map(p)[ch] - H[ch]); ax = axs[i, m - 1]
        im = ax.pcolormesh(ang, np.arange(16), np.clip(d, -15, 15), cmap=DIV, norm=norm, shading='nearest', rasterized=True)
        th = {'1S_IVb': 141, '1S_DV': 33, '2S_IV': 312, '2S_DV': 178}[v]; ax.axvline(th, color='#0b0b0b', lw=0.7, ls=(0, (3, 2)))
        if i == 0: ax.set_title(f'M{m}: {MODE[m]}', fontsize=8.5)
        if m == 1: ax.set_ylabel(f'{v}\n(A{ch + 1})  kHz', fontsize=8); ax.set_yticks([0, 5, 10, 15]); ax.set_yticklabels([f'{FC[k]:.1f}' for k in (0, 5, 10, 15)])
        if i == 3: ax.set_xlabel('Crank angle (°)'); ax.set_xticks([0, 90, 180, 270])
        ax.text(0.02, 0.95, f'P{p}', transform=ax.transAxes, va='top', fontsize=7, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.8))
fig.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.07)
cax = fig.add_axes([0.9, 0.25, 0.015, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_label('Change from healthy, $10\\,\\Delta_c$ (dB)', fontsize=8); cb.ax.tick_params(labelsize=7)
save(fig, 'Fig_fault_signatures')
# ---------- Fig. C: own-valve opening changes (Table 4.1) as annotated heat maps
t = pd.read_csv('res/own_valve_opening.csv')
def num(s): return float(s.split()[0].rstrip('*'))
vorder = ['1S_IVb', '1S_DV', '2S_IV', '2S_DV']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.45))
for ax, col, lab, lim in ((axs[0], 'dL', '$\\Delta L$ (dB)', 12.5), (axs[1], 'dth', '$\\Delta\\theta_p$ (°)', 9.5)):
    M = np.array([[num(t[(t.valve == v) & (t['mode'] == m)][col].iloc[0]) for m in range(1, 5)] for v in vorder])
    S = np.array([['*' in t[(t.valve == v) & (t['mode'] == m)][col].iloc[0] for m in range(1, 5)] for v in vorder])
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f'{M[a, b]:+.1f}' + ('*' if S[a, b] else ''), ha='center', va='center', fontsize=8, color='white' if abs(M[a, b]) > 0.62 * lim else '#0b0b0b')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Seat\nwear', 'Plate\ncorrosion', 'Plate\nfracture', 'Spring\nbreakage'], fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(vorder); ax.set_title(lab, fontsize=9)
    ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, 4), minor=True); ax.grid(which='minor', color='#fcfcfb', lw=2); ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=7)
axs[0].text(-0.32, 1.06, '(a)', transform=axs[0].transAxes, fontweight='bold'); axs[1].text(-0.32, 1.06, '(b)', transform=axs[1].transAxes, fontweight='bold')
fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.18); save(fig, 'Fig_own_valve')
# ---------- Fig. D: state index vs interstage pressure
S = pd.read_csv('res/state_s.csv').dropna(); h = S.P == 1
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.scatter(S.Pr2_mean[h], S.s[h], s=14, c='#52514e', marker='s', label='Healthy', zorder=3, edgecolors='white', linewidths=0.4)
for v in vorder:
    k = S.P.map(lambda p: VAL.get(p, ('',))[0]) == v
    ax.scatter(S.Pr2_mean[k], S.s[k], s=14, c=VCOL[v], label=f'Fault in {v}', zorder=3, edgecolors='white', linewidths=0.4)
b = np.polyfit(S.Pr2_mean, S.s, 1); xx = np.linspace(S.Pr2_mean.min(), S.Pr2_mean.max(), 10); ax.plot(xx, np.polyval(b, xx), color='#0b0b0b', lw=1)
from scipy.stats import pearsonr; r = pearsonr(S.Pr2_mean, S.s)[0]
ax.text(0.03, 0.96, f'$r_{{\mathrm{{P}}}}$ = {r:.3f}\nslope = {b[0]:.1f}°/bar\n$n$ = {len(S)} recordings', transform=ax.transAxes, va='top', fontsize=7.5)
ax.set_xlabel('Measured interstage pressure (bar)'); ax.set_ylabel('State index $s_i$ (°), Eq. (4.11)'); ax.grid(color='#e6e5e1', lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False, fontsize=7, handletextpad=0.2); fig.tight_layout(); save(fig, 'Fig_state_index')
