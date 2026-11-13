"""Fault-operator decomposition at recording level:
  dy[k,r] = b_local[k,cond(r)]*own(k,valve(r)) + c_k * s_r + delta[k,day(r)] + e
s_r: vibration-only thermodynamic index (state.py); for the four events that define s, a leave-one-out s is used.
Reports, per event and parameter, the share of variance of dy (fault recordings) explained by day, +s, +local,
and the leave-one-condition-out R2 of the non-local part (day + s) on events that are NOT own events of the held-out valve."""
import numpy as np, pandas as pd
VAL = {**{p: '2S_DV' for p in range(2, 6)}, **{p: '2S_IV' for p in range(6, 10)}, **{p: '1S_DV' for p in range(10, 14)}, **{p: '1S_IVb' for p in range(14, 18)}}
DAY = {p: (14 if p <= 10 else 18) for p in range(2, 18)}
OWN = {('A1', 33): ['1S_DV'], ('A2', 34): ['1S_DV'], ('A1', 107): ['1S_DV', '2S_IV'], ('A2', 107): ['1S_DV', '2S_IV'],
       ('A1', 141): ['1S_IVb'], ('A2', 144): ['1S_IVb'], ('A1', 179): ['2S_DV'], ('A2', 178): ['2S_DV'],
       ('A1', 285): ['1S_IVb', '2S_DV'], ('A2', 286): ['1S_IVb', '2S_DV'], ('A1', 312): ['2S_IV'], ('A2', 312): ['2S_IV'],
       ('A2', 335): ['2S_IV'], ('A2', 130): []}
SDEF = [('A1', 33), ('A2', 34), ('A1', 179), ('A2', 178)]; SIGN = {('A1', 33): 1, ('A2', 34): 1, ('A1', 179): -1, ('A2', 178): -1}
e = pd.read_csv('res/events_rec.csv')
L = e.pivot_table(index=['P', 'R'], columns=['ch', 'ev'], values='L'); T = e.pivot_table(index=['P', 'R'], columns=['ch', 'ev'], values='thp')
dL = L - L.loc[1].mean(); dT = T - T.loc[1].mean()
def s_excl(excl=None):
    ks = [k for k in SDEF if k != excl]; return sum(SIGN[k] * dT[k] for k in ks) / len(ks)
rows = []; idx = dL.index; f = idx.get_level_values(0) >= 2
P = idx.get_level_values(0)[f]
for k in OWN:
    for par, D in (('L', dL), ('thp', dT)):
        y = D[k].values[f]; s = s_excl(k if (par == 'thp' and k in SDEF) else None).values[f]
        day = np.array([DAY[p] for p in P]) == 18; own = np.array([VAL[p] in OWN[k] for p in P])
        cond = pd.get_dummies(pd.Series(P)).values * own[:, None]; cond = cond[:, cond.any(0)]
        X0 = np.column_stack([np.ones_like(s), day]); X1 = np.column_stack([X0, s]); X2 = np.column_stack([X1, cond]) if cond.size else X1
        def r2(X): b = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b
        r0, _ = r2(X0); r1, b1 = r2(X1); r2_, b2 = r2(X2)
        # leave-one-condition-out on non-own recordings, model day + s
        no = ~own; yp = np.full(no.sum(), np.nan); Pn = P[no]; yn = y[no]; Xn = X1[no]
        for p in np.unique(Pn):
            tr = Pn != p; b = np.linalg.lstsq(Xn[tr], yn[tr], rcond=None)[0]; yp[~tr] = Xn[~tr] @ b
        loco = 1 - ((yn - yp) ** 2).sum() / ((yn - yn.mean()) ** 2).sum()
        rows.append(dict(ch=k[0], ev=k[1], par=par, R2_day=r0, R2_day_s=r1, R2_full=r2_, c_s=b1[2], LOCO_R2_nonown=loco, n_nonown=int(no.sum())))
R = pd.DataFrame(rows); R.to_csv('res/opmodel.csv', index=False)
pd.set_option('display.width', 200); print(R.round(3).to_string())
print(R.groupby('par')[['R2_day', 'R2_day_s', 'R2_full', 'LOCO_R2_nonown']].median().round(3))
