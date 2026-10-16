"""Generate notebooks 01-03: thin views over the pipeline outputs (run run_all.sh first)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
HEAD = """import sys; sys.path.insert(0, '..')
import pandas as pd, numpy as np
from IPython.display import Image, display
from msdvv import load_config
cfg = load_config(); OUT = cfg['output_dir']; FIG = OUT + '/figs/'
pd.set_option('display.width', 200)"""

NB = {
    "01_noise.ipynb": [
        ("md", "# 01 · Data, geometry and noise\nInventory, repeater picks / channel locations, PSD, coherence vs offset, f–k, and the physical/non-physical channel classification (`scripts/s00`–`s05`)."),
        ("code", HEAD),
        ("code", "inv = pd.read_csv(OUT + '/inventory.csv'); inv[['run','cable','start','end','duration_h','nch','setup','gap_before_s']]"),
        ("code", "display(Image(FIG + 'geometry.png'))\nfor c in ('north','south'): display(pd.read_csv(f'{OUT}/repeater_picks_{c}.csv'))"),
        ("code", "pd.read_csv(OUT + '/events.csv')[['run','time','mag','place','dist_km','win0','win1']]"),
        ("code", "for c in ('north','south'):\n    for k in ('psd','coherence','fk'): display(Image(f'{FIG}{k}_{c}.png'))"),
        ("md", "Channel quality: median magnitude-squared coherence at 2-channel offset (0.12–0.6 Hz). Threshold 0.15 reproduces the readme's good/bad stretches on the south cable (258–260 km bad, 261–263 and 265–268 km good, 290+ km bad)."),
        ("code", "display(Image(FIG + 'channel_quality.png'))\nfor c in ('north','south'):\n    q = pd.read_csv(f'{OUT}/channel_quality_{c}.csv'); display(q.groupby(['block','span']).good.agg(['sum','size','mean']).round(2))"),
    ],
    "02_convergence.ipynb": [
        ("md", "# 02 · Cross-correlation gathers and convergence\nReference gathers (offset × lag), and how fast 10–120-min stacks converge to a leave-one-run-out reference for single pairs vs 1/2/5-km spatial stacks (`s06`–`s08b`)."),
        ("code", HEAD),
        ("code", "for c in ('north','south'): display(Image(f'{FIG}gathers_{c}_all.png'))"),
        ("code", "display(Image(FIG + 'convergence.png'))\ns = pd.read_csv(OUT + '/convergence/summary.csv')\ns[s.T_min.isin([30,60])].pivot_table(index=['cable','block','span','band'], columns=['window','level','T_min'], values='cc_median').round(2)"),
        ("md", "Late coda (≥15 s) does not converge within 2 h; direct and early-coda windows do. Tested remedies:"),
        ("code", "pd.read_csv(OUT + '/convergence/coda_fixes.csv').pivot_table(index=['cable','band'], columns='method', values='cc_median_60min').round(3)"),
    ],
    "03_dvv.ipynb": [
        ("md", "# 03 · dv/v, noise floor, tides, run offsets\n30-min stacks (moving 10 min), stretching vs all-January reference, inverse-variance averaged over offsets and lag sides, then over each repeater span (`s09`–`s10`)."),
        ("code", HEAD),
        ("code", "for c in ('north','south'): display(Image(f'{FIG}dvv_timeseries_{c}.png'))"),
        ("code", "display(Image(FIG + 'dvv_precision_map.png'))\nfl = pd.read_csv(OUT + '/dvv/noise_floor.csv')\nfl[fl.level=='5km'].pivot_table(index=['cable','seg_id'], columns=['band','window'], values='floor_span').round(4)"),
        ("code", "t = pd.read_csv(OUT + '/dvv/tide_regression.csv')\nprint('tests:', len(t), ' |z|>2:', (t.z.abs()>2).sum(), ' (≈', round(0.0455*len(t),1), 'expected by chance)')\nt[(t.level=='5km') & (t.window=='early')][['cable','band','seg_id','beta_per_m','se_per_m','z']].round(5)"),
        ("code", "r = pd.read_csv(OUT + '/dvv/run_means.csv')\nr[(r.level=='5km') & (r.window=='early')].pivot_table(index=['cable','seg_id','band'], columns='run', values='dv').round(4)"),
        ("code", "pd.read_csv(OUT + '/dvv/method_agreement.csv').round(3)"),
    ],
}

for name, cells in NB.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)
print("notebooks written")
