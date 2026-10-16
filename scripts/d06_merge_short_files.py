"""Item 7 fix: join the short Nov 26 - Dec 2 2025 reduced files (~9.6 min each, raw 52.9-m full-fiber period) into
~38-min pieces so that the cross-span correlation (600-s windows, s20) can use them; within-span (240-s windows) is not
affected. Up to MAX_FILES consecutive files of one cable are placed on one true-time grid (sample positions from each
file's proc_start; gaps < MAX_GAP_S between files are zero-filled and flagged invalid; larger gaps start a new piece).
Data are stored as float32 strain rate (no 'scale' dataset) with a 'valid' mask; metadata = the first file's attributes
with stream.proc_start of the piece. Files are written to raw_nov/merged_short/<cable>/ with the original naming
pattern (first file's name), so msdvv readers treat them like any reduced file.
usage: python scripts/d06_merge_short_files.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import pandas as pd

OUT = Path("/data/wsd02/multispan_dvv/raw_nov/merged_short")
INV = pd.read_csv("/data/wsd02/multispan_dvv/outputs_nov_ram/inventory.csv")
MAX_FILES, MAX_GAP_S, SHORT_S = 4, 5.0, 700.0
inv = INV[(INV.t1 - INV.t0) < SHORT_S].sort_values(["cable", "t0"]).reset_index(drop=True)
print(f"{len(inv)} short files ({inv.groupby('cable').size().to_dict()}), {pd.to_datetime(inv.t0.min(), unit='s')} - {pd.to_datetime(inv.t1.max(), unit='s')}")
n_out = 0
for cab, g in inv.groupby("cable"):
    od = OUT / cab
    od.mkdir(parents=True, exist_ok=True)
    rows = g.to_dict("records")
    i = 0
    while i < len(rows):
        grp = [rows[i]]
        while len(grp) < MAX_FILES and i + len(grp) < len(rows) and rows[i + len(grp)]["t0"] - grp[-1]["t1"] <= MAX_GAP_S:
            grp.append(rows[i + len(grp)])
        i += len(grp)
        with h5py.File(grp[0]["path"], "r") as h:
            attrs = dict(h["data"].attrs)
            nch = h["data"].shape[0]
        fs = float(attrs["new_fs"])
        t0 = grp[0]["t0"]
        n = int(round((grp[-1]["t1"] - t0) * fs)) + 1
        data = np.zeros((nch, n), np.float32)
        valid = np.zeros(n, bool)
        for r in grp:
            with h5py.File(r["path"], "r") as h:
                x = h["data"][:].astype(np.float32)
                if "scale" in h:
                    x *= h["scale"][:][:, None]
                v = h["valid"][:] if "valid" in h else np.ones(x.shape[1], bool)
            p0 = int(round((r["t0"] - t0) * fs))
            m = min(x.shape[1], n - p0)
            data[:, p0:p0 + m] = x[:, :m]
            valid[p0:p0 + m] = v[:m]
        attrs["stream.proc_start"] = float(t0)
        name = od / Path(grp[0]["path"]).name
        with h5py.File(name, "w") as h:
            d = h.create_dataset("data", data=data, compression="lzf")
            for k, v in attrs.items():
                d.attrs[k] = v
            h["valid"] = valid
        n_out += 1
print(f"wrote {n_out} merged pieces to {OUT}")
