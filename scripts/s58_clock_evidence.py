"""Item 2 (docs/NEXT_STEPS.md): evidence for the DAS clock note (docs/DAS_clock_note.md).

For every raw file on disk (100-m stream Dec 23 - Jan 28, both cables; 8-Hz stream samples) the per-chunk
stream.proc_start / stream.gpu_idx of the OFDR trailer are compared with continuous sampling from the header time:
number of dropped chunks (gpu_idx step > 1.5 x nominal), the accumulated drift by the end of the file.
Figure: figs/clock_note_evidence.png (drops per file vs time; drift inside the Jan 8 south file; colocated HYSB1 lag
per file before and after using the per-chunk times). Table: outputs_100m/clock_drops_per_file.csv
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from msdvv.raw import chunk_start_times, is_complete, is_raw, read_chunk_meta, read_header

B = Path("/data/wsd02/multispan_dvv")
paths = sorted(glob.glob(str(B / "raw/**/nokia*"), recursive=True)) + sorted(glob.glob(str(B / "raw_hf/**/nokia*"), recursive=True))
rows = []
for p in paths:
    if not Path(p).is_file() or not is_raw(p) or not is_complete(p):
        continue
    h = read_header(p)
    fs, m, n = h["attrs"]["new_fs"], h["m"], h["nchunk"]
    step = m / fs
    ct = chunk_start_times(p, h)
    _, cm = read_chunk_meta(p)
    gi = np.array([c.get("stream.gpu_idx", np.nan) for c in cm[1:n]], float)
    dgi = np.diff(gi[np.isfinite(gi)]) if np.isfinite(gi).sum() > 2 else np.array([])
    nom_gi = np.median(dgi) if len(dgi) else np.nan
    dts = np.diff(ct)
    rows.append(dict(path=p, cable="north" if "north" in Path(p).name else "south", stream="100-m" if "-100m" in Path(p).name else "8-Hz",
                     t0=float(h["attrs"]["stream.proc_start"]), dx=h["attrs"]["dx"], fs=fs, D=int(h["attrs"]["D"]), chunk_s=step, nchunk=n,
                     duration_h=n * step / 3600, drops=int(np.sum(dts > 1.5 * step)), drift_s=float(ct[-1] - (ct[0] + (n - 1) * step)),
                     gpu_idx_step=nom_gi, max_gpu_jump=float(dgi.max()) if len(dgi) else np.nan))
R = pd.DataFrame(rows)
R.to_csv(B / "outputs_100m/clock_drops_per_file.csv", index=False)
pd.set_option("display.width", 200)
print(R.groupby(["stream", "cable", "D"]).agg(files=("drops", "size"), with_drops=("drops", lambda x: int((x > 0).sum())), drops=("drops", "sum"),
                                              hours=("duration_h", "sum"), max_drift_s=("drift_s", "max"), chunk_s=("chunk_s", "first")).round(2).to_string())

# ---- figure
fig, axs = plt.subplots(3, 1, figsize=(13, 11))
ax = axs[0]
for (st, cab), g in R.groupby(["stream", "cable"]):
    ax.scatter(pd.to_datetime(g.t0, unit="s"), g.drops / g.duration_h, s=14, label=f"{st} {cab}")
ax.set(ylabel="dropped chunks per hour", title="Dropped acquisition chunks per file (per-chunk proc_start / gpu_idx in the OFDR trailer)")
ax.legend(fontsize=8)
ax = axs[1]
ex = R[(R.stream == "100-m") & (R.cable == "south")].copy()
ex = ex.iloc[(np.abs(ex.t0 - pd.Timestamp("2026-01-08 00:30", tz="UTC").timestamp())).argmin()]
h = read_header(ex.path)
ct = chunk_start_times(ex.path, h)
step = h["m"] / h["attrs"]["new_fs"]
nom = ct[0] + np.arange(len(ct)) * step
ax.plot((nom - nom[0]) / 3600, ct - nom, "k-", lw=1.2)
ax.set(xlabel="hours into file (nominal sample time)", ylabel="true - nominal time (s)",
       title=f"South 100-m file starting {pd.Timestamp(ex.t0, unit='s'):%Y-%m-%d %H:%M}: {int(ex.drops)} dropped chunks of {step:.2f} s -> {ex.drift_s:.0f} s by the end")
ax = axs[2]
qc_old = B / "outputs_100m/obsdas_nominal_times/file_timing_qc_T.csv"
qc_new = B / "outputs_100m/obsdas_combined/file_timing_qc_T.csv"
for f_, lab, col in ((qc_old, "nominal times (t0 + i/fs)", "tab:red"), (qc_new, "per-chunk times (fixed)", "tab:blue")):
    if f_.exists():
        q = pd.read_csv(f_)
        ax.scatter(pd.to_datetime(q.t0, unit="s"), q.cc0, s=14, color=col, label=lab)
ax.axhline(0.6, color="0.5", ls="--", lw=0.8)
ax.set(ylabel="CC at zero shift vs reference", ylim=(-0.4, 1),
       title="Colocated HYSB1 x DAS correlation per file vs the Dec 26 - Jan 6 reference (T component, 0.12-0.3 Hz)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(B / "outputs_100m/figs/clock_note_evidence.png", dpi=100)
print("done")
