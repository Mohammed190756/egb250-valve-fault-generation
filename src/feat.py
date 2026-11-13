"""Per-revolution crank-angle x frequency energy maps and angle-resolved RMS for A1, A2."""
import numpy as np, glob, os
from scipy.signal import butter, sosfiltfilt
REV = '../revs/'; OUT = '../maps/'; os.makedirs(OUT, exist_ok=True)
NA = 4096; NB = 360; FSH = 12.975; FSE = NA * FSH          # equivalent sampling rate in the angle domain
EDGES = np.geomspace(250, 24000, 17)                        # 16 log-spaced bands, Hz (at nominal speed)
SOS = [butter(4, [EDGES[i], min(EDGES[i + 1], 0.49 * FSE)], btype='band', fs=FSE, output='sos') for i in range(16)]
def maps(X):   # X: n_rev x 2 x NA  ->  n_rev x 2 x 16 x NB band energy (mean square per 1-deg bin), and broadband RMS n x 2 x NB
    n = len(X); flat = X.reshape(n * 2, NA)
    # filter on the concatenated revolutions of each channel to avoid edge effects
    out = np.empty((n, 2, 16, NB), np.float32)
    for c in range(2):
        s = X[:, c, :].reshape(-1)
        for b, sos in enumerate(SOS):
            y = sosfiltfilt(sos, s).reshape(n, NA) ** 2
            out[:, c, b] = y.reshape(n, NB, -1).mean(2) if NA % NB == 0 else np.add.reduceat(y, np.linspace(0, NA, NB + 1).astype(int)[:-1], axis=1) / np.diff(np.linspace(0, NA, NB + 1).astype(int))
    rms = np.sqrt(np.add.reduceat(X ** 2, np.linspace(0, NA, NB + 1).astype(int)[:-1], axis=2) / np.diff(np.linspace(0, NA, NB + 1).astype(int)))
    return out, rms.astype(np.float32)
if __name__ == '__main__':
    for fn in sorted(glob.glob(REV + 'P*.npz')):
        o = OUT + os.path.basename(fn)
        if os.path.exists(o): continue
        z = np.load(fn); X = z['X']; k = ~z['sat']
        M, R = maps(X); M, R = M[k], R[k]; np.savez_compressed(o, M=M, R=R, P=z['P'], R_=z['R'], nrev=int(k.sum()))
        print(os.path.basename(fn), M.shape, flush=True)
