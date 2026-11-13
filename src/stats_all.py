"""Event-parameter changes of every fault condition against healthy (recording level), bootstrap 95% CI, Holm."""
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
rng = np.random.default_rng(0)
VAL = {**{p: ('2S_DV', p - 1) for p in range(2, 6)}, **{p: ('2S_IV', p - 5) for p in range(6, 10)},
       **{p: ('1S_DV', p - 9) for p in range(10, 14)}, **{p: ('1S_IVb', p - 13) for p in range(14, 18)}}
DAY = {p: ('14 Sep' if p <= 10 else '18 Sep') for p in range(2, 18)}
df = pd.read_csv('res/events_rec.csv'); PAR = ['L', 'thp', 'D', 'fc']
def boot(a, b, B=10000):
    ia = rng.integers(0, len(a), (B, len(a))); ib = rng.integers(0, len(b), (B, len(b)))
    return np.percentile(a[ia].mean(1) - b[ib].mean(1), [2.5, 97.5])
rows = []
for (ch, ev), g in df.groupby(['ch', 'ev']):
    h = g[g.P == 1]
    for p in range(2, 18):
        f = g[g.P == p]
        for q in PAR:
            a, b = f[q].values, h[q].values; lo, hi = boot(a, b)
            rows.append(dict(ch=ch, ev=ev, P=p, valve=VAL[p][0], mode=VAL[p][1], day=DAY[p], par=q, delta=a.mean() - b.mean(), lo=lo, hi=hi,
                             d=(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2), p=mannwhitneyu(a, b).pvalue))
S = pd.DataFrame(rows); ps = S.p.values; o = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0
for i, j in enumerate(o): run = max(run, (k - i) * ps[j]); adj[j] = min(1, run)
S['p_holm'] = adj; S.to_csv('res/events_stats_all.csv', index=False)
for q in ['L', 'thp']:
    t = S[S.par == q].copy(); t['cell'] = t.apply(lambda r: f"{r.delta:+.1f}" + ('*' if r.p_holm < 0.05 else ''), axis=1)
    t['cond'] = t.valve + '_' + t['mode'].astype(str)
    print(f'\n== {q}'); print(t.pivot_table(index=['ch', 'ev'], columns='P', values='cell', aggfunc='first').to_string())
