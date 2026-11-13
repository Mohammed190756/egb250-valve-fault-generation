"""Two-way decomposition of fault effects: delta_{v,m} = alpha_v + beta_m + gamma_{v,m} (recording level, deltas vs healthy mean).
Reports the share of between-condition variance explained by valve, mode and interaction for each event parameter."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
VAL = {**{p: ('2S_DV', p - 1) for p in range(2, 6)}, **{p: ('2S_IV', p - 5) for p in range(6, 10)},
       **{p: ('1S_DV', p - 9) for p in range(10, 14)}, **{p: ('1S_IVb', p - 13) for p in range(14, 18)}}
df = pd.read_csv('res/events_rec.csv'); rows = []
for (ch, ev), g in df.groupby(['ch', 'ev']):
    h = g[g.P == 1]; f = g[g.P >= 2].copy(); f['valve'] = f.P.map(lambda p: VAL[p][0]); f['mode'] = f.P.map(lambda p: str(VAL[p][1]))
    for q in ['L', 'thp']:
        f['y'] = f[q] - h[q].mean()
        m = smf.ols('y ~ C(valve) * C(mode)', data=f).fit(); a = sm.stats.anova_lm(m, typ=2)
        ss = a['sum_sq']; tot_between = ss[['C(valve)', 'C(mode)', 'C(valve):C(mode)']].sum()
        rows.append(dict(ch=ch, ev=ev, par=q, valve=ss['C(valve)'] / tot_between, mode=ss['C(mode)'] / tot_between, inter=ss['C(valve):C(mode)'] / tot_between,
                         R2=m.rsquared, p_inter=a.loc['C(valve):C(mode)', 'PR(>F)']))
R = pd.DataFrame(rows); R.to_csv('res/anova_vm.csv', index=False)
print(R.round(3).to_string())
print(R.groupby('par')[['valve', 'mode', 'inter', 'R2']].mean().round(3))
