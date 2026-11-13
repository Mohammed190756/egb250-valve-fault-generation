import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
rng = np.random.default_rng(0)
df = pd.read_csv('res/events_rec.csv'); PAR = ['L', 'thp', 'tho', 'D', 'fc']
def boot(a, b, B=10000):
    ia = rng.integers(0, len(a), (B, len(a))); ib = rng.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1); return np.percentile(d, [2.5, 97.5])
rows = []
for (ch, ev), g in df.groupby(['ch', 'ev']):
    h = g[g.P == 1]
    for p in [6, 7, 8, 9]:
        f = g[g.P == p]
        for q in PAR:
            a, b = f[q].values, h[q].values; lo, hi = boot(a, b)
            dd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            rows.append(dict(ch=ch, ev=ev, P=p, par=q, delta=a.mean() - b.mean(), lo=lo, hi=hi, d=dd, p=mannwhitneyu(a, b).pvalue))
    # temporal control within healthy: R9-R15 minus R1-R8
    for q in PAR:
        a, b = h[h.R >= 9][q].values, h[h.R <= 8][q].values
        rows.append(dict(ch=ch, ev=ev, P=0, par=q, delta=a.mean() - b.mean(), lo=np.nan, hi=np.nan,
                         d=(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2), p=mannwhitneyu(a, b).pvalue))
S = pd.DataFrame(rows)
m = S.P > 0; ps = S.loc[m, 'p'].values; o = np.argsort(ps); k = len(ps); adj = np.empty(k); run = 0
for i, j in enumerate(o): run = max(run, (k - i) * ps[j]); adj[j] = min(1, run)
S.loc[m, 'p_holm'] = adj; S.to_csv('res/events_stats.csv', index=False)
pd.set_option('display.width', 250)
for q, unit in [('L', 'dB'), ('thp', 'deg'), ('D', 'dB/deg'), ('fc', 'kHz')]:
    t = S[S.par == q].copy(); t['cell'] = t.apply(lambda r: f"{r.delta:+.1f}" + ('*' if r.get('p_holm', 1) < 0.05 else ''), axis=1)
    print(f'\n== {q} ({unit}); * = Holm p<0.05; P0 = healthy late-minus-early block')
    print(t.pivot_table(index=['ch', 'ev'], columns='P', values='cell', aggfunc='first').to_string())
