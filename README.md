# egb250-valve-fault-generation

Code and result files for the study *What a valve fault does to compressor vibration: a crank-angle mathematical model and a comparison of generative models that translate healthy data into specified faults*.

The study models how each single valve fault of a two-stage reciprocating compressor changes the vibration on the crank-angle axis. It then compares seven generative models that translate fresh healthy vibration into a requested fault (valve, mode): CVAE, DDPM, DiT, flow matching, diffusion bridge, conditional normalising flow and StarGAN v2.

## Data

The EGB-250 reciprocating-compressor dataset is the property of GIDTEC, Universidad Politécnica Salesiana (Cuenca, Ecuador), and was used under a data-sharing agreement. It is **not** included in this repository. Requests should be directed to GIDTEC.

## Layout

| Folder | Content |
|---|---|
| `src/` | All code (flat folder; scripts import each other directly) |
| `results/` | Result tables (CSV, Markdown) behind every number, table and figure of the paper |
| `diagrams/` | TikZ sources of the schematic and architecture figures |

Folders are set by environment variables: `EGB_RAW` (raw `.mat` recordings) and `EGB_FEAT` (pressure features used only for the crank-angle origin and checks). Run the scripts from `src/`; outputs are written to the parent folder (`../ds.npz`, `../runs/`, `../res/`, `../fig/`).

## Order of execution

1. `extract.py`, `revs.py`, `revs2.py`, `offsets.py`, `offsets_win.py`, `offset_r8.py`: angular resampling and crank-angle origin.
2. `dataset.py`, `split_cond.py`: band-energy maps (Eq. 4.8) and the R1–R8 / R9–R10 / R11–R15 split.
3. `events.py`, `stats1.py`, `stats_all.py`, `anova_vm.py`, `state.py`, `opmodel.py`, `loo.py`: mathematical model (Section 4).
4. `m_cvae.py`, `m_diff.py`, `m_dit.py`, `m_fm.py`, `m_bridge.py`, `m_flow.py`, `m_stargan.py` (shared code in `gcommon.py`), run through `queue3.sh`: training.
5. `gen_eval.py`, `evalsuite.py`, `baselines.py`, `results.py`: metrics of Section 6.
6. `gen_save.py`, `judge.py`, `judge_eval.py`, `tstr_all.py`: judge and TSTR analyses.
7. `fig_*.py`: figures.

All models were trained on two CPU cores with seed 0 (Python 3.13, PyTorch 2.15, NumPy 2.5, SciPy 1.18, scikit-learn 1.9).

## Licence

MIT. See `LICENSE`.
