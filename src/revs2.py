"""Angular resampling: cut A1, A2 into crank-angle-aligned revolutions (NA points per revolution).
Crank-angle origin per recording = encoder pulse phase corrected by the pressure-based offset (res/offsets.csv, column REF)."""
import numpy as np, pandas as pd, glob, os, sys
from scipy.interpolate import CubicSpline
FS = 51200.0; NA = 4096
RAW = '../raw/'; OUT = '../revs/'; os.makedirs(OUT, exist_ok=True)
OW = pd.read_csv('../res/offsets_win.csv')
REF = 'window'
def one(fn):
    z = np.load(fn); p, r = int(z['P']), int(z['R']); ed = z['edges']
    g = OW[(OW.P == p) & (OW.R == r)].sort_values('w'); wo = g.off.values
    jump = np.abs(np.diff(wo)); bad = np.zeros(9, bool); bad[:-1] |= jump > 8; bad[1:] |= jump > 8
    d = float(np.median(wo[~bad])) if (~bad).any() else float(np.median(wo))
    t = np.arange(len(z['A1'])) / FS
    n_ed = np.arange(len(ed))
    # revolution coordinate of target samples: rev = n + k/NA + d/360 (aligned origin)
    n0 = int(np.ceil(-wo.min() / 360)) + 1 ; n1 = len(ed) - 2
    revs = []
    for n in range(max(n0, 0), n1):
        tn = ed[n] if n < len(ed) else ed[-1]; w = min(int(tn), 8)
        if bad[w]: continue
        dn = float(np.interp(tn, np.arange(9) + 0.5, wo))
        q = n + np.arange(NA) / NA + dn / 360.0
        if q[0] < 0 or q[-1] > len(ed) - 1: continue
        tq = np.interp(q, n_ed, ed)
        if tq[-1] >= t[-1] - 1e-3: continue
        a = int(tq[0] * FS) - 4; b = int(tq[-1] * FS) + 5
        seg = []
        for c in ('A1', 'A2'):
            cs = CubicSpline(t[a:b], z[c][a:b]); seg.append(cs(tq))
        revs.append(np.stack(seg))
    X = np.asarray(revs, np.float32)
    sat = (np.abs(X) >= 495).any(axis=(1, 2))
    clip = np.array([[np.abs(z[c]).max() for c in ('A1', 'A2')]])
    np.savez_compressed(OUT + os.path.basename(fn), X=X, P=p, R=r, offset=d, ref=REF, sat=sat, fshaft=1 / np.median(np.diff(ed)))
    return p, r, X.shape
PS = [int(x) for x in os.environ.get('PS', ','.join(map(str, range(1, 18)))).split(',')]
for fn in sorted(glob.glob(RAW + 'P*.npz')):
    if int(os.path.basename(fn)[1:3]) not in PS: continue
    if os.path.exists(OUT + os.path.basename(fn)): continue
    print(one(fn), flush=True)
