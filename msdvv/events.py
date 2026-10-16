"""Earthquake windows to exclude from noise processing (USGS ComCat)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
from pyproj import Geod

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
CENTER = (45.3, -125.5)  # lat, lon roughly between the two cables
GEOD = Geod(ellps="WGS84")


def query(t0: pd.Timestamp, t1: pd.Timestamp, **kw) -> pd.DataFrame:
    p = dict(format="csv", starttime=t0.strftime("%Y-%m-%dT%H:%M:%S"), endtime=t1.strftime("%Y-%m-%dT%H:%M:%S"), orderby="time-asc")
    p.update(kw)
    r = requests.get(URL, params=p, timeout=60)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text))


def event_windows(t0: pd.Timestamp, t1: pd.Timestamp, pad_s: float = 3 * 3600) -> pd.DataFrame:
    """M>=6.5 anywhere, M>=5.5 within 3000 km, M>=2.5 within 500 km, with exclusion windows [win0, win1] (UTC).

    Window: from predicted P (8 km/s) to surface-wave arrival (2.5 km/s) plus a magnitude-dependent coda.
    Events up to ``pad_s`` before t0 are included because long-period surface waves arrive late.
    """
    q0 = t0 - pd.Timedelta(seconds=pad_s)
    glob = query(q0, t1, minmagnitude=5.5)
    reg = query(q0, t1, minmagnitude=2.5, latitude=CENTER[0], longitude=CENTER[1], maxradiuskm=500)
    ev = pd.concat([glob, reg]).drop_duplicates("id") if len(glob) or len(reg) else pd.DataFrame()
    if ev.empty:
        return ev
    ev["time"] = pd.to_datetime(ev.time, utc=True)
    _, _, d = GEOD.inv(np.full(len(ev), CENTER[1]), np.full(len(ev), CENTER[0]), ev.longitude.values, ev.latitude.values)
    ev["dist_km"] = d / 1e3
    # teleseisms only matter if large; amplitude-based window rejection handles the rest
    ev = ev[(ev.mag >= 6.5) | ((ev.mag >= 5.5) & (ev.dist_km < 3000)) | ((ev.mag >= 2.5) & (ev.dist_km < 500))].copy()
    if ev.empty:
        return ev
    coda = 120 + 600 * np.clip(ev.mag - 4, 0, None)
    ev["win0"] = ev.time + pd.to_timedelta(ev.dist_km / 8.0, unit="s")
    ev["win1"] = ev.time + pd.to_timedelta(ev.dist_km / 2.5 + coda, unit="s")
    return ev[["id", "time", "latitude", "longitude", "depth", "mag", "place", "dist_km", "win0", "win1"]].sort_values("time")
