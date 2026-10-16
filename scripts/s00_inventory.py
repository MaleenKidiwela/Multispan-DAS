"""Step 0a: inventory of January files, grouped into runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msdvv import load_config
from msdvv.io import build_inventory, read_meta

cfg = load_config()
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

inv = build_inventory(cfg["data_dirs"], cfg["date_range"], cfg["file_glob"], cfg["require"])
inv.to_csv(out / "inventory.csv", index=False)
cols = ["run", "cable", "start", "end", "duration_h", "nch", "setup", "dtype", "gap_before_s"]
print(inv[cols].to_string())
for cab, g in inv.groupby("cable"):
    print(f"{cab}: {len(g)} files, {g.duration_h.sum():.1f} h, runs {sorted(g.run.unique())}")
    print(read_meta(g.path.iloc[0]).blocks().to_string())
