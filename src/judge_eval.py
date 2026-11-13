"""Score every generator with the real-data judge: requested fault = true label, judge's decision = predicted label."""
import sys, os, json, numpy as np, pandas as pd; sys.path.insert(0, '..')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from judge import predict, X, P, R
from evalsuite import event
CONDS = list(range(2, 18)); TE = R >= 11
NAMES = [('REAL', 'Real faults (reference)'), ('B0', 'Mean-difference baseline'), ('Bridge', 'Diffusion bridge'), ('FlowMatching', 'Flow matching'), ('CVAE', 'CVAE'),
         ('DiT', 'DiT'), ('NormFlow', 'Normalising flow'), ('Diffusion', 'DDPM (CNN)'), ('StarGANv2', 'StarGAN v2'), ('Healthy', 'Healthy unchanged')]
H = X[TE & (P == 1)].astype(np.float32)
Lh = {p: event(H, p) for p in CONDS}
rows, per, phys, cms = [], [], [], {}
for k, lab in NAMES:
    if k == 'REAL': G = {p: X[TE & (P == p)].astype(np.float32) for p in CONDS}
    else:
        f = f'res/gen/{k}.npz'
        if not os.path.exists(f): continue
        Z = np.load(f); G = {p: Z[str(p)].astype(np.float32) for p in CONDS}
    yt = np.concatenate([np.full(len(G[p]), p) for p in CONDS]); yp = np.concatenate([predict(G[p]) for p in CONDS])
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=CONDS, average='macro', zero_division=0)
    rows.append(dict(key=k, Generator=lab, Accuracy=accuracy_score(yt, yp), Precision=pr, Recall=rc, F1=f1, Kappa=cohen_kappa_score(yt, yp), MCC=matthews_corrcoef(yt, yp),
                     Healthy_share=float((yp == 1).mean())))
    pp, rr, ff, _ = precision_recall_fscore_support(yt, yp, labels=CONDS, average=None, zero_division=0)
    for i, p in enumerate(CONDS): per.append(dict(key=k, P=p, precision=pp[i], recall=rr[i], F1=ff[i]))
    cms[k] = confusion_matrix(yt, yp, labels=list(range(1, 18)))[1:]          # rows = requested P2..P17, cols = recognised H,P2..P17
    for p in CONDS:
        L, T = event(G[p], p); phys.append(dict(key=k, P=p, dL=L.mean() - Lh[p][0].mean(), dT=T.mean() - Lh[p][1].mean()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('res/judge_summary.csv', index=False); pd.DataFrame(per).to_csv('res/judge_perfault.csv', index=False)
pd.DataFrame(phys).to_csv('res/judge_phys.csv', index=False); np.savez('res/judge_cm.npz', **cms)
yv = predict(X[TE & (P == 1)]); print('judge on real healthy test:', float((yv == 1).mean()))
