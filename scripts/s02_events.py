"""Step 0c: earthquake exclusion windows overlapping the recorded data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from msdvv import load_config
from msdvv.events import event_windows

cfg = load_config()
out = Path(cfg["output_dir"])
inv = pd.read_csv(out / "inventory.csv", parse_dates=["start", "end"])
evs = []
for run, g in inv.groupby("run"):
    ev = event_windows(g.start.min(), g.end.max())
    if len(ev):
        ev = ev[(ev.win1 > g.start.min()) & (ev.win0 < g.end.max())]
        ev["run"] = run
        evs.append(ev)
cols = ["id", "time", "latitude", "longitude", "depth", "mag", "place", "dist_km", "win0", "win1", "run"]
ev = pd.concat(evs, ignore_index=True) if evs else pd.DataFrame(columns=cols)
if ev.empty:
    print("no catalogue events pass the exclusion criteria in these windows")
ev.to_csv(out / "events.csv", index=False)
print(ev[["run", "time", "mag", "place", "dist_km", "win0", "win1"]].to_string())
