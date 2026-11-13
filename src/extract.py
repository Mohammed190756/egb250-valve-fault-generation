"""Extract A1, A2 (51.2 kS/s) and encoder E1 rising edges (500 kS/s) from staged valve-session .mat files.
Saves one compact npz per recording to ../raw and deletes the staged .mat (keeps R1P1, R1P5 originals)."""
import os, re, glob, sys, datetime, numpy as np, scipy.io as sio
import os; SRC = os.environ.get('EGB_RAW', '../RawData/Raw_Data_Valve_Faults_DB_011V0'); OUT = '../raw/'
KEEP = {'R1P1.mat', 'R1P5.mat'}
def one(path):
    fn = os.path.basename(path); r, p = map(int, re.match(r'R(\d+)P(\d+)\.mat$', fn).groups())
    out = OUT + f'P{p:02d}_R{r:02d}.npz'
    if os.path.exists(out): return 'skip'
    M = sio.loadmat(path); d = M['data'][0, 0]
    hdr = M['__header__'].decode(errors='ignore')
    try: ts = datetime.datetime.strptime(hdr.split('Created on: ')[1].strip(), '%a %b %d %H:%M:%S %Y').isoformat()
    except Exception: ts = ''
    g = d['Analog50k'][0, 0]
    a1 = g['TDMSAnalog50kAcelerometer0'].ravel().astype(np.float32); a2 = g['TDMSAnalog50kAcelerometer1'].ravel().astype(np.float32)
    enc = d['Analog500k'][0, 0]['TDMSAnalog500kEncoder0'].ravel()
    ed = np.flatnonzero(np.diff((enc > 0).astype(np.int8)) == 1) / 500000.0
    np.savez_compressed(out, A1=a1, A2=a2, edges=ed, P=p, R=r, ts=ts)
    return f'{fn}: n={len(a1)} edges={len(ed)} f={1/np.median(np.diff(ed)):.3f}Hz ts={ts}'
if __name__ == '__main__':
    for f in sorted(glob.glob(SRC + '/P*/*.mat')):
        p = int(re.search(r'P(\d+)\.mat$', f).group(1))
        if p > 17: continue
        try:
            msg = one(f)
            if os.path.basename(f) not in KEEP: os.remove(f)
            if msg != 'skip': print(msg, flush=True)
        except Exception as e: print('ERR', f, e, flush=True)
