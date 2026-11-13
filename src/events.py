"""Event-level parameters of each revolution (A1, A2) around the healthy impact events.
For each event window: peak level L (dB re 1 m/s^2, 1-deg RMS), peak angle theta_p (deg), onset angle theta_o
(first crossing of the midpoint between pre-event floor and peak, in log energy), decay slope D (dB/deg, linear fit
of log energy over the 6 deg after the peak) and centroid frequency fc (kHz) of the event window."""
import numpy as np, glob, pandas as pd
NA = 4096; DEG = NA / 360.0; FSH = 12.975; FSE = NA * FSH
EV = {0: [33, 107, 141, 179, 285, 312], 1: [34, 107, 130, 144, 178, 286, 312, 335]}
HW = 9.0
def smooth_energy(x, w=12):
    k = np.ones(w) / w; e = np.convolve(np.concatenate([x[-w:], x, x[:w]]) ** 2, k, 'same')[w:-w]; return e
def params(x, th):
    e = smooth_energy(x); n = len(x); out = {}
    i0 = int((th - HW) * DEG); i1 = int((th + HW) * DEG); idx = np.arange(i0, i1) % n; seg = e[idx]
    ip = int(np.argmax(seg)); L = 10 * np.log10(seg[ip] + 1e-12)
    pre = 10 * np.log10(np.median(e[np.arange(i0 - int(6 * DEG), i0) % n]) + 1e-12)
    half = (L + pre) / 2; lseg = 10 * np.log10(seg + 1e-12)
    on = np.flatnonzero(lseg[:ip + 1] < half); io = (on[-1] + 1) if len(on) else 0
    j = np.arange(ip, min(ip + int(6 * DEG), len(seg)))
    D = np.polyfit(j / DEG, lseg[j], 1)[0] if len(j) > 5 else np.nan
    xs = x[idx]; X = np.abs(np.fft.rfft(xs * np.hanning(len(xs)))) ** 2; f = np.fft.rfftfreq(len(xs), 1 / FSE)
    fc = (f * X).sum() / X.sum() / 1000
    return dict(L=L, thp=(i0 + ip) / DEG, tho=(i0 + io) / DEG, D=D, fc=fc, pre=pre)
rows = []
for fn in sorted(glob.glob('revs/P*.npz')):
    z = np.load(fn); X = z['X'][~z['sat']]; p, r = int(z['P']), int(z['R'])
    for c in (0, 1):
        for k, th in enumerate(EV[c]):
            P = pd.DataFrame([params(X[i, c], th) for i in range(len(X))])
            d = P.median().to_dict(); d.update(P=p, R=r, ch=f'A{c+1}', ev=th, nrev=len(X)); rows.append(d)
df = pd.DataFrame(rows); df.to_csv('res/events_rec.csv', index=False); print(df.shape)
