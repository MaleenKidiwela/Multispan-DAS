"""Stream a large archive section: download -> correlate -> write one ccf/<run>.h5 per file -> delete raw.

usage: MSDVV_CONFIG=config_hf.yaml python scripts/d02_stream_xcorr.py sample   # keep a spread sample (for s00-s05)
       MSDVV_CONFIG=config_hf.yaml python scripts/d02_stream_xcorr.py xcorr    # stream everything (needs s05 output)
Resumable: files whose output exists are skipped. A manifest is kept in <output_dir>/stream_manifest.csv.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed

from msdvv import load_config
from msdvv.download import crawl, fetch
from msdvv.io import read_meta
from msdvv.xcorr import correlate_file, write_run

cfg = load_config()
dl = cfg["download"]
out = Path(cfg["output_dir"])
(out / "ccf").mkdir(parents=True, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "xcorr"
files = crawl(*cfg["date_range"], dl["streams"]).sort_values(["stream", "day", "name"]).reset_index(drop=True)
files["cable"] = files.stream.str.replace("-100m", "")
# file names carry a (rough) write time; good enough for spreading the sample and naming
files["ftime"] = pd.to_datetime(files.name.str.extract(r"_(\d{8}_\d{6})_")[0], format="%Y%m%d_%H%M%S", utc=True)
print(f"{len(files)} files listed ({files.groupby('cable').size().to_dict()})", flush=True)

if mode == "sample":
    dest = Path(cfg["data_dirs"][0])
    keep = []
    for cab, g in files.groupby("cable"):
        last = None
        for r in g.itertuples():
            if last is None or (r.ftime - last).total_seconds() >= dl["sample_every_h"] * 3600:
                keep.append(r.Index)
                last = r.ftime
    sel = files.loc[keep]
    paths = [dest / f"{d:%Y/%m/%d}" / s / n for d, s, n in zip(pd.to_datetime(sel.day), sel.stream, sel.name)]
    with ThreadPoolExecutor(dl["parallel"]) as ex:
        res = list(ex.map(fetch, sel.url, paths))
    print(pd.DataFrame(res).status.str.split(":").str[0].value_counts().to_string())
    sys.exit(0)

# ---- streaming correlation
ev = pd.read_csv(out / "events.csv", parse_dates=["win0", "win1"])
good = {c: pd.read_csv(out / f"channel_quality_{c}.csv").good.values for c in files.cable.unique()}
files["run"] = [f"{c[0].upper()}{i:04d}" for c, i in zip(files.cable, files.groupby("cable").cumcount())]
todo = files[[not (out / "ccf" / f"{r}.h5").exists() for r in files.run]].reset_index(drop=True)
print(f"{len(todo)} files to process", flush=True)
tmp = Path(cfg["data_dirs"][0]).parent / "tmp"
tmp.mkdir(parents=True, exist_ok=True)
n_jobs = cfg["xcorr"].get("n_jobs", 12)
workers = max(1, 48 // n_jobs)
B = dl.get("batch_files", 36)
batches = [todo.iloc[i : i + B] for i in range(0, len(todo), B)]
mpath = out / "stream_manifest.csv"
t_start = time.time()


def download_batch(b):
    return list(ThreadPoolExecutor(dl["parallel"]).map(fetch, b.url, [tmp / n for n in b.name]))


def process(path, cable, run):
    try:
        f = read_meta(path)
        ok = any(abs(f.dx - v) < 1e-3 for v in cfg["require"]["dx"]) and abs(f.fs - cfg["require"]["fs"]) < 1e-3
        if not ok:
            return dict(run=run, status=f"skipped setup dx={f.dx:.3f} fs={f.fs:.3f}")
        r = correlate_file(str(path), cfg, ev, workers, good[cable])
        if len(r["t_sub"]) == 0:
            return dict(run=run, status="no substacks", t0=f.t0, setup=f.setup)
        write_run(out / "ccf" / f"{run}.h5", [r], cfg, run)
        return dict(run=run, status="ok", t0=f.t0, t1=f.t1, setup=f.setup, nsub=len(r["t_sub"]),
                    nwin=int(r["nwin"].sum()), rejected=r["n_rejected"])
    except Exception as e:  # noqa: BLE001 - keep streaming, report in manifest
        return dict(run=run, status=f"failed: {type(e).__name__}: {e}")


with ThreadPoolExecutor(1) as pre, Parallel(n_jobs=n_jobs) as par:
    nxt = pre.submit(download_batch, batches[0]) if batches else None
    for i, b in enumerate(batches):
        dres = nxt.result()
        nxt = pre.submit(download_batch, batches[i + 1]) if i + 1 < len(batches) else None  # prefetch
        okd = [(Path(d["path"]), c, r) for d, c, r in zip(dres, b.cable, b.run) if d["status"] in ("ok", "exists")]
        res = par(delayed(process)(p, c, r) for p, c, r in okd)
        for p, _, _ in okd:
            p.unlink(missing_ok=True)
        man = pd.DataFrame(res).merge(b[["run", "url", "cable"]], on="run", how="right")
        man.loc[man.status.isna(), "status"] = "download failed"
        man.to_csv(mpath, mode="a", header=not mpath.exists(), index=False)
        el = time.time() - t_start
        print(f"batch {i + 1}/{len(batches)}: {man.status.str.split(':').str[0].value_counts().to_dict()} "
              f"elapsed {el / 60:.1f} min, ETA {el / (i + 1) * (len(batches) - i - 1) / 60:.0f} min", flush=True)
