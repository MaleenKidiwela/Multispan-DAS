"""Fill gaps in the daily OBS files from d04 (whole-day FDSN requests sometimes come back truncated).

For every day/station whose Z coverage is < 99.5 %, re-request the missing intervals in 2-h chunks, merge with
what is on disk, decimate 200 -> 20 Hz like d04 and rewrite the file. Real outages stay gaps (FDSN no-data).
usage: python scripts/d04b_fill_obs_gaps.py [START END]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

OUT = Path("/data/wsd02/multispan_dvv/obs/continuous")
START, END = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2025-12-23", "2026-01-28")
REQ = [("HYS14", "HH?"), ("HYSB1", "HH?"), ("HYS11", "EH?"), ("HYS12", "EH?"), ("HYS13", "EH?")]
C = Client("https://service.earthscope.org", timeout=300)
CHUNK = 7200.0


def covered(st, t0, t1):
    """Merged [a, b] intervals of the Z traces inside [t0, t1]."""
    iv = sorted((max(tr.stats.starttime.timestamp, t0), min(tr.stats.endtime.timestamp, t1)) for tr in st.select(component="Z"))
    out = []
    for a, b in iv:
        if b <= a:
            continue
        if out and a <= out[-1][1] + 1.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def fill(d, sta, cha):
    fn = OUT / f"{d:%Y%m%d}" / f"OO.{sta}.{cha[:2]}.mseed"
    t0 = UTCDateTime(d.strftime("%Y-%m-%d")).timestamp
    t1 = t0 + 86400
    old = read(str(fn)) if fn.exists() else Stream()
    cov = covered(old, t0, t1)
    frac = sum(b - a for a, b in cov) / 86400
    if frac >= 0.995:
        return dict(day=f"{d:%Y%m%d}", sta=sta, before=frac, after=frac, status="complete")
    # missing intervals, requested in 2-h chunks
    edges, prev = [], t0
    for a, b in cov + [[t1, t1]]:
        if a - prev > 1.0:
            edges.append((prev, a))
        prev = max(prev, b)
    new = Stream()
    nodata = 0
    for a, b in edges:
        for c0 in np.arange(a, b, CHUNK):
            c1 = min(c0 + CHUNK, b)
            for attempt in range(3):
                try:
                    new += C.get_waveforms("OO", sta, "*", cha, UTCDateTime(c0 - 60), UTCDateTime(c1 + 60))
                    break
                except FDSNNoDataException:
                    nodata += 1
                    break
                except Exception:  # noqa: BLE001 - transient: retry
                    if attempt == 2:
                        return dict(day=f"{d:%Y%m%d}", sta=sta, before=frac, after=frac, status="request failed")
    if not len(new):
        return dict(day=f"{d:%Y%m%d}", sta=sta, before=frac, after=frac, status=f"server has no data ({nodata} chunks)")
    for tr in new:
        tr.stats.sampling_rate = round(tr.stats.sampling_rate, 2)
    new.merge(method=1, fill_value=None)
    new = new.split()
    for tr in new:
        tr.detrend("demean")
        if tr.stats.sampling_rate == 200 and tr.stats.npts > 200:
            tr.decimate(10, no_filter=False)
        tr.data = tr.data.astype("float32")
    new.trim(UTCDateTime(t0), UTCDateTime(t1))
    # keep old traces, add new samples only where the old file had none (no double coverage)
    old_cov = covered(old, t0, t1)
    add = Stream()
    for tr in new:
        a, b = tr.stats.starttime.timestamp, tr.stats.endtime.timestamp
        pieces, cur = [], a
        for oa, ob in old_cov:
            if ob <= cur or oa >= b:
                continue
            if oa > cur:
                pieces.append((cur, oa - 1.0 / tr.stats.sampling_rate))
            cur = max(cur, ob + 1.0 / tr.stats.sampling_rate)
        if cur < b:
            pieces.append((cur, b))
        for pa, pb in pieces:
            if pb - pa > 10:
                add += tr.slice(UTCDateTime(pa), UTCDateTime(pb))
    st = (old + add)
    st = Stream([tr for tr in st if tr.stats.npts > 0])
    for tr in st:
        tr.data = tr.data.astype("float32")
    st.sort()
    st.write(str(fn), format="MSEED", encoding="FLOAT32")
    after = sum(b - a for a, b in covered(st, t0, t1)) / 86400
    return dict(day=f"{d:%Y%m%d}", sta=sta, before=frac, after=after, status="filled")


jobs = [(d, s, c) for d in pd.date_range(START, END, freq="D") for s, c in REQ]
with ThreadPoolExecutor(6) as ex:
    res = pd.DataFrame(list(ex.map(lambda a: fill(*a), jobs)))
res.to_csv(OUT / "fill_report.csv", index=False)
pd.set_option("display.width", 200)
print(res[res.status != "complete"].round(3).to_string(index=False))
print(res.groupby("sta")[["before", "after"]].mean().round(4).to_string())
