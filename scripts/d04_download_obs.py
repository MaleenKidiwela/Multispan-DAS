"""Download continuous OOI seafloor seismometer data (EarthScope FDSN), decimate 200 -> 20 Hz, save per day.

usage: python scripts/d04_download_obs.py [START END]   (default 2025-12-23 2026-01-28)
Output: /data/wsd02/multispan_dvv/obs/continuous/<YYYYMMDD>/OO.<STA>.<CHA>.mseed, plus OO_response.xml
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

OUT = Path("/data/wsd02/multispan_dvv/obs/continuous")
OUT.mkdir(parents=True, exist_ok=True)
START, END = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2025-12-23", "2026-01-28")
REQ = [("HYS14", "HH?"), ("HYSB1", "HH?"), ("HYS11", "EH?"), ("HYS12", "EH?"), ("HYS13", "EH?")]
C = Client("https://service.earthscope.org", timeout=300)
inv = C.get_stations(network="OO", station=",".join(s for s, _ in REQ), starttime=UTCDateTime(START), endtime=UTCDateTime(END) + 86400, level="response")
inv.write(str(OUT / "OO_response.xml"), format="STATIONXML")


def day(d, sta, cha):
    try:
        return _day(d, sta, cha)
    except Exception as e:  # noqa: BLE001 - keep going, report
        return f"{d:%Y%m%d} {sta} failed ({type(e).__name__}: {e})"


def _day(d, sta, cha):
    dd = OUT / f"{d:%Y%m%d}"
    dd.mkdir(exist_ok=True)
    fn = dd / f"OO.{sta}.{cha[:2]}.mseed"
    if fn.exists():
        return f"{d:%Y%m%d} {sta} exists"
    t0 = UTCDateTime(d.strftime("%Y-%m-%d"))
    try:
        st = C.get_waveforms("OO", sta, "*", cha, t0 - 60, t0 + 86400 + 60)
    except Exception as e:  # noqa: BLE001 - no data that day
        return f"{d:%Y%m%d} {sta} none ({type(e).__name__})"
    for tr in st:  # some records report e.g. 200.0000153 Hz: round so they merge
        tr.stats.sampling_rate = round(tr.stats.sampling_rate, 2)
    try:
        st.merge(method=1, fill_value=None)
    except Exception as e:  # noqa: BLE001
        return f"{d:%Y%m%d} {sta} merge failed ({e})"
    st = st.split()  # gaps stay gaps
    for tr in st:
        tr.detrend("demean")
        if tr.stats.sampling_rate == 200 and tr.stats.npts > 200:
            tr.decimate(10, no_filter=False)  # 200 -> 20 Hz, anti-alias FIR
    st.trim(t0, t0 + 86400)
    for tr in st:
        tr.data = tr.data.astype("float32")
    st.write(str(fn), format="MSEED", encoding="FLOAT32")
    return f"{d:%Y%m%d} {sta} ok {len(st)} traces"


jobs = [(d, s, c) for d in pd.date_range(START, END, freq="D") for s, c in REQ]
with ThreadPoolExecutor(6) as ex:
    for msg in ex.map(lambda a: day(*a), jobs):
        print(msg, flush=True)
