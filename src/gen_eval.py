"""Generate faulty revolutions from fresh healthy revolutions (R11-R15) for all 16 conditions and evaluate."""
import sys, numpy as np, torch, pandas as pd, importlib; sys.path.insert(0, '..')
import gcommon as gc
from evalsuite import evaluate, X, P, R, CONDS
def run(modname, cls, ckpt, name, gen_kw=None, n_max=None):
    M = importlib.import_module(modname); net = getattr(M, cls)(); net.load_state_dict(torch.load(ckpt)); net.eval()
    Hte = gc.norm(X[(R >= 11) & (P == 1)]); Ht = torch.tensor(Hte if n_max is None else Hte[:n_max]); G = {}
    for p in CONDS:
        v, m = gc.VAL[p]; out = []
        for j in range(0, len(Ht), 256):
            xb = Ht[j:j + 256]; vv = torch.full((len(xb),), v, dtype=torch.long); mm = torch.full((len(xb),), m, dtype=torch.long)
            out.append(net.generate(xb, vv, mm, **(gen_kw or {})).numpy())
        G[p] = gc.denorm(np.concatenate(out))
    df, s = evaluate(G, name); df.to_csv(f'../res/eval_{name}.csv'); return df, s
if __name__ == '__main__':
    df, s = run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]); print(s)
    pd.DataFrame([s]).to_csv(f'../res/evalsum_{sys.argv[4]}.csv', index=False)
