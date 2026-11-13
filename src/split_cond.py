import numpy as np, glob, os, re, datetime
import os; S = os.environ.get('EGB_EXTRACT', '../extract/'); OUT = '../raw/'
for f in sorted(glob.glob(S + 'P*.npz')):
    p = int(re.search(r'P(\d+)\.npz', f).group(1)); z = np.load(f)
    rs = sorted({int(k[1:3]) for k in z.files})
    for r in rs:
        out = OUT + f'P{p:02d}_R{r:02d}.npz'
        if os.path.exists(out): continue
        hdr = str(z[f'R{r:02d}_hdr'])
        try: ts = datetime.datetime.strptime(hdr.split('Created on: ')[1].strip(), '%a %b %d %H:%M:%S %Y').isoformat()
        except Exception: ts = ''
        np.savez_compressed(out, A1=z[f'R{r:02d}_A1'], A2=z[f'R{r:02d}_A2'], edges=z[f'R{r:02d}_edges'], P=p, R=r, ts=ts)
    print(p, len(rs), flush=True)
    os.remove(f)
