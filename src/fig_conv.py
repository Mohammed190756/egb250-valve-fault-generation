"""Convergence plots: training and validation criteria per epoch for each generative model, with the selected (early-stopping) epoch marked."""
import json, glob, os, numpy as np, matplotlib, matplotlib.ticker; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9})
SPEC = {'cvae': ('CVAE', [('train_negelbo', 'train $-$ELBO'), ('val_negelbo', 'val. $-$ELBO')], 'val_negelbo'),
        'diff': ('Conditional DDPM', [('train_eps_mse', 'train $\\epsilon$-MSE'), ('val_eps_mse', 'val. $\\epsilon$-MSE')], 'val_eps_mse'),
        'fm': ('Flow matching', [('train_fm_mse', 'train FM loss'), ('val_fm_mse', 'val. FM loss')], 'val_fm_mse'),
        'bridge': ('Diffusion bridge (I2SB)', [('train_bridge_mse', 'train bridge loss'), ('val_bridge_mse', 'val. bridge loss')], 'val_bridge_mse'),
        'dit': ('DiT', [('train_eps_mse', 'train $\\epsilon$-MSE'), ('val_eps_mse', 'val. $\\epsilon$-MSE')], 'val_eps_mse'),
        'flow': ('Normalising flow', [('train_nll_nats_per_dim', 'train NLL'), ('val_nll_nats_per_dim', 'val. NLL')], 'val_nll_nats_per_dim'),
        'stargan': ('StarGAN v2: adversarial game', [('adv', 'G adversarial'), ('d', 'D loss')], 'val_mean_map_mse'),
        'stargan_val': ('StarGAN v2: validation', [('val_mean_map_mse', 'val. mean-map MSE')], 'val_mean_map_mse')}
avail = [k for k in SPEC if os.path.exists(f'runs/{k.replace("_val", "")}.json')]
fig, axs = plt.subplots(int(np.ceil(len(avail) / 3)), 3, figsize=(7.2, 2.2 * int(np.ceil(len(avail) / 3))), squeeze=False)
for ax, k in zip(axs.flat, avail):
    rows = json.load(open(f'runs/{k.replace("_val", "")}.json')); t, curves, sel = SPEC[k]; ep = [r['epoch'] + 1 for r in rows]
    for r in rows:
        if 'train_nll' in r and 'train_kl' in r: r['train_negelbo'] = r['train_nll'] + r['train_kl']
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for (key, lab), c in zip(curves, ('#0B5394', '#C55A11')):
        if key in rows[0]: ax.plot(ep, [r[key] for r in rows], '-o', ms=2.5, lw=1.2, color=c, label=lab)
    if sel in rows[0]:
        b = int(np.argmin([r[sel] for r in rows])); ax.axvline(b + 1, color='0.4', ls='--', lw=0.8); right = b + 1 > 0.6 * len(rows); ax.text(b + 1 + (-0.3 if right else 0.3), ax.get_ylim()[1], f'selected: {b + 1}', va='top', ha='right' if right else 'left', fontsize=7.5, color='0.3')
    ax.set_title(t, loc='left', fontweight='bold'); ax.set_xlabel('Epoch'); ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7.5, frameon=False)
for ax in list(axs.flat)[len(avail):]: ax.axis('off')
fig.tight_layout(); os.makedirs('fig', exist_ok=True); fig.savefig('fig/Fig_convergence.png', dpi=600); fig.savefig('fig/Fig_convergence.pdf'); print(avail)
