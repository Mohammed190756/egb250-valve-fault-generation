"""Assemble the generative-model dataset: per-revolution log10 band energy maps (2 ch x 16 bands x 360 deg), float16.
Split: repetitions R1-R10 = train, R11-R15 = test (healthy test = 'fresh healthy input')."""
import numpy as np, glob
X, P, R = [], [], []
for f in sorted(glob.glob('maps/P*.npz')):
    z = np.load(f); M = np.log10(z['M'].astype(np.float64) + 1e-12).astype(np.float16)
    X.append(M); P += [int(z['P'])] * len(M); R += [int(z['R_'])] * len(M)
X = np.concatenate(X); P = np.array(P); R = np.array(R)
tr = R <= 10
mu = X[tr & (P == 1)].astype(np.float32).mean(0); sd = X[tr & (P == 1)].astype(np.float32).std(0).mean(axis=2, keepdims=True)  # per channel-band scale
np.savez('../ds.npz', X=X, P=P, R=R, mu=mu, sd=sd)
print(X.shape, X.dtype, 'train', tr.sum(), 'test', (~tr).sum(), [int((P == p).sum()) for p in range(1, 18)])
