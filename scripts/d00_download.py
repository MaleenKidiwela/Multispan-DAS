"""Mirror the archive for the configured date range and streams.

usage: MSDVV_CONFIG=config_100m.yaml python scripts/d00_download.py [START END]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.download import crawl, fetch

cfg = load_config()
dl = cfg["download"]
start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else cfg["date_range"]
dest = Path(cfg["data_dirs"][0])
files = crawl(start, end, dl["streams"])
print(f"{len(files)} files listed for {start}..{end} {dl['streams']}")
files["dest"] = [dest / f"{d:%Y/%m/%d}" / s / n for d, s, n in zip(pd.to_datetime(files.day), files.stream, files.name)]
res = Parallel(n_jobs=dl["parallel"], prefer="threads")(delayed(fetch)(u, p) for u, p in zip(files.url, files.dest))
man = pd.DataFrame(res)
mpath = dest / "manifest.csv"
if mpath.exists():
    man = pd.concat([pd.read_csv(mpath), man]).drop_duplicates("path", keep="last")
man.to_csv(mpath, index=False)
new = man[man.path.isin(map(str, files.dest))]
print(new.status.str.split(":").str[0].value_counts().to_string())
ok = new[new.status == "ok"]
if len(ok):
    print(f"downloaded {ok.bytes.sum() / 1e9:.1f} GB at {ok.bytes.sum() / max(ok.seconds.sum(), 1e-9) / 1e6 * dl['parallel']:.0f} MB/s (aggregate)")
bad = new[new.status.str.startswith("failed")]
if len(bad):
    print(bad.to_string())
