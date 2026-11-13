"""Generate the 16 single faults from the 620 fresh healthy test revolutions with every model and save them (float16, log10 units)."""
import sys, os, importlib, numpy as np, torch; sys.path.insert(0, '..')
import gcommon as gc
D = np.load('../ds180.npz'); X, P, R = D['X'], D['P'], D['R']; TE = R >= 11; CONDS = list(range(2, 18))
MODELS = [('CVAE', 'm_cvae', 'CVAE', 'runs/cvae.pt', {}), ('StarGANv2', 'm_stargan', 'StarGAN', 'runs/stargan.pt', {}), ('FlowMatching', 'm_fm', 'FM', 'runs/fm.pt', {}),
          ('NormFlow', 'm_flow', 'Flow', 'runs/flow.pt', {}), ('Bridge', 'm_bridge', 'Bridge', 'runs/bridge.pt', {}), ('Diffusion', 'm_diff', 'Diff', 'runs/diff.pt', {}), ('DiT', 'm_dit', 'DiT', 'runs/dit.pt', {})]
H = X[TE & (P == 1)].astype(np.float32); muH = X[(R <= 8) & (P == 1)].astype(np.float32).mean(0)
f = 'res/gen/B0.npz'
if not os.path.exists(f): np.savez(f, **{str(p): (H + X[(R <= 8) & (P == p)].astype(np.float32).mean(0) - muH).astype(np.float16) for p in CONDS})
f = 'res/gen/Healthy.npz'
if not os.path.exists(f): np.savez(f, **{str(p): H.astype(np.float16) for p in CONDS})
for name, mod, cls, ck, kw in MODELS:
    f = f'res/gen/{name}.npz'
    if os.path.exists(f): continue
    M = importlib.import_module(mod); net = getattr(M, cls)(); net.load_state_dict(torch.load(ck)); net.eval()
    Ht = torch.tensor(gc.norm(H)); G = {}
    for p in CONDS:
        v, m = gc.VAL[p]; out = []
        for j in range(0, len(Ht), 256):
            xb = Ht[j:j + 256]
            with torch.no_grad(): out.append(net.generate(xb, torch.full((len(xb),), v), torch.full((len(xb),), m), **kw).numpy())
        G[str(p)] = gc.denorm(np.concatenate(out)).astype(np.float16)
    np.savez(f + '.tmp.npz', **G); os.replace(f + '.tmp.npz', f); print('saved', name, flush=True)
