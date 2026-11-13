import numpy as np, pandas as pd, scipy.io as sio
import os; D = os.path.join(os.environ.get('EGB_RAW', '../RawData/Raw_Data_Valve_Faults_DB_011V0'), 'P3/')
F = os.environ.get('EGB_FEAT', '../features/')
r = pd.read_csv(F + 'recordings.csv'); v = r[(r.campaign == 'Valve') & (r.P == 1)]
z = np.load(F + 'angle_profiles.npz', allow_pickle=True); P = z['prof']; idx = {x: i for i, x in enumerate(z['records'])}; ch = list(z['channels'])
K = np.arange(1, 7); GRID = np.arange(-180, 180, 0.1); PH = np.exp(1j * np.outer(GRID * np.pi / 180, K))
def spec(x): x = (x - x.mean()) / x.std(); return np.fft.rfft(x)[K]
T = spec(np.stack([P[idx[x], :, ch.index('Pr1')].mean(0) for x in v.record]).mean(0))
def prof(fn):
    M = sio.loadmat(D + fn)['data'][0, 0]; pr = M['Analog25k'][0, 0]['TDMSAnalog25kPressure0'].ravel()
    enc = M['Analog500k'][0, 0]['TDMSAnalog500kEncoder0'].ravel(); ed = np.flatnonzero(np.diff((enc > 0).astype(np.int8)) == 1) / 500000.0
    t = np.arange(len(pr)) / 25000.0; rev = np.interp(t, ed, np.arange(len(ed)), left=np.nan, right=np.nan); out = []
    for w in range(9):
        m = (t >= w) & (t < w + 1) & np.isfinite(rev); rr = rev[m]; r0, r1 = np.ceil(rr.min()), np.floor(rr.max()); k = (rr >= r0) & (rr < r1)
        b = ((rr[k] % 1) * 72).astype(int).clip(0, 71); out.append(np.bincount(b, pr[m][k], 72) / np.bincount(b, None, 72))
    return np.array(out)
rows = []
for fn, R in (('R7P3.mat', 7), ('R8P3.mat', 8)):
    pf = prof(fn)
    for w in range(9):
        cost = (np.abs(T[None] - spec(pf[w])[None] * PH) ** 2).sum(1); k = np.argmin(cost)
        rows.append(dict(P=3, R=R, w=w, ref='Pr1', off=GRID[k], q=cost[k] / (np.abs(T) ** 2).sum()))
o = pd.DataFrame(rows); print(o.groupby('R').off.agg(['mean', 'min', 'max']))
ow = pd.read_csv('res/offsets_win.csv'); print('existing R7 window offsets:', ow[(ow.P == 3) & (ow.R == 7)].off.round(1).tolist())
ow = pd.concat([ow, o[o.R == 8]]).sort_values(['P', 'R', 'w']); ow.to_csv('res/offsets_win.csv', index=False)
