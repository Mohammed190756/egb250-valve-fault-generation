"""Collect evaluation summaries of all generators into Table 7.x and per-fault figures."""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ORDER = [('REF healthy (no change)', 'Healthy unchanged'), ('B0 mean difference', 'Mean-difference baseline'), ('CVAE', 'CVAE'), ('CVAE_sampled', 'CVAE (sampled)'),
         ('Diffusion', 'DDPM'), ('DiT', 'DiT'), ('FlowMatching', 'Flow matching'), ('Bridge', 'Diffusion bridge'), ('NormFlow', 'Normalising flow'), ('StarGANv2', 'StarGAN v2'),
         ('REF real train (copy)', 'Copy of training data')]
S = pd.concat([pd.read_csv('res/eval_summary_baselines.csv')] + [pd.read_csv(f) for f in glob.glob('res/evalsum_*.csv')])
S = S.drop_duplicates('model', keep='last').set_index('model')
rows = [(lab, *S.loc[k, ['rmse_dB', 'FD', 'MMD2', 'abs_dL', 'abs_dTheta', 'DCR', 'TSTR']].values) for k, lab in ORDER if k in S.index]
T = pd.DataFrame(rows, columns=['Generator', 'RMSE (dB)', 'FD', 'MMD²', '|ΔL| (dB)', '|Δθ| (°)', 'DCR ratio', 'TSTR acc.'])
T.to_csv('res/table_main.csv', index=False); print(T.round(3).to_string(index=False))
# unseen single faults
HOLD = [2, 7, 12, 17]; out = []
for f, lab in [('res/loo_eval_additive.csv', 'Additive baseline'), ('res/loo_eval_FM.csv', 'Flow matching'), ('res/loo_eval_Bridge.csv', 'Diffusion bridge'), ('res/loo_eval_CVAE.csv', 'CVAE'), ('res/eval_B0.csv', 'True mean change (oracle)')]:
    try: d = pd.read_csv(f).set_index('P')
    except FileNotFoundError: continue
    for p in HOLD: out.append(dict(model=lab, P=p, rmse=d.loc[p, 'rmse_dB'], MMD2=d.loc[p, 'MMD2'], dL=d.loc[p, 'dL_dB'], dth=d.loc[p, 'dTheta']))
if out: L = pd.DataFrame(out); L.to_csv('res/table_unseen.csv', index=False); print(L.pivot_table(index='model', columns='P', values='rmse').round(2))
