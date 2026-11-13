"""Diffusion transformer (Peebles and Xie, 2023) as the denoiser of the conditional diffusion translator:
patchify the crank-angle axis (patch p = 6 bins -> 30 tokens), transformer blocks with adaLN-Zero conditioning on the
sum of timestep and condition embeddings; healthy revolution concatenated to the noisy input before patchify.
Training objective, guidance and DDIM sampling identical to m_diff (Ho et al., 2020; Ho and Salimans, 2022; Song et al., 2021)."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F; sys.path.insert(0, '..')
from gcommon import *
import m_diff
from m_diff import temb, T, ABAR
PATCH = 3; NTOK = 180 // PATCH; D = 128; HEADS = 4; DEPTH = 4
class DiTBlock(nn.Module):
    def __init__(s):
        super().__init__(); s.n1 = nn.LayerNorm(D, elementwise_affine=False); s.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        s.n2 = nn.LayerNorm(D, elementwise_affine=False); s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        s.ada = nn.Sequential(nn.SiLU(), nn.Linear(64, 6 * D)); nn.init.zeros_(s.ada[1].weight); nn.init.zeros_(s.ada[1].bias)   # adaLN-Zero
    def forward(s, x, e):
        b1, g1, a1, b2, g2, a2 = s.ada(e)[:, None].chunk(6, -1)
        h = s.n1(x) * (1 + g1) + b1; x = x + a1 * s.att(h, h, h, need_weights=False)[0]
        h = s.n2(x) * (1 + g2) + b2; return x + a2 * s.mlp(h)
class DiTNet(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = CondEmb(64); s.null = nn.Parameter(torch.zeros(64)); s.tm = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 64))
        s.inp = nn.Linear(64 * PATCH, D); s.pos = nn.Parameter(torch.randn(1, NTOK, D) * 0.02)
        s.blocks = nn.ModuleList([DiTBlock() for _ in range(DEPTH)]); s.nf = nn.LayerNorm(D, elementwise_affine=False)
        s.adaf = nn.Sequential(nn.SiLU(), nn.Linear(64, 2 * D)); s.out = nn.Linear(D, 32 * PATCH)
        nn.init.zeros_(s.adaf[1].weight); nn.init.zeros_(s.adaf[1].bias); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, xt, t, xh, v, m, drop=None):
        e = s.emb(v, m)
        if drop is not None: e = torch.where(drop[:, None], s.null.expand_as(e), e)
        e = e + s.tm(temb(t)); B = len(xt)
        tok = torch.cat([xt, xh], 1).reshape(B, 64, NTOK, PATCH).permute(0, 2, 1, 3).reshape(B, NTOK, 64 * PATCH)
        x = s.inp(tok) + s.pos
        for bl in s.blocks: x = bl(x, e)
        b, g = s.adaf(e)[:, None].chunk(2, -1); x = s.out(s.nf(x) * (1 + g) + b)
        return x.reshape(B, NTOK, 32, PATCH).permute(0, 2, 1, 3).reshape(B, 32, 180)
class DiT(m_diff.Diff):
    def __init__(s): nn.Module.__init__(s); s.net = DiTNet()
def train(name='dit', epochs=40, patience=6, **kw):
    m_diff.Diff_orig = m_diff.Diff; m_diff.Diff = DiT
    try: m_diff.train(name=name, epochs=epochs, patience=patience, **kw)
    finally: m_diff.Diff = m_diff.Diff_orig
if __name__ == '__main__':
    train()
