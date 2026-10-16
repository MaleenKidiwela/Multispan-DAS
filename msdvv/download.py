"""Mirror files from the public OOI DAS 2025 archive (piweb.ooirsn.uw.edu/das25).

Layout on the server: data/MultiDAS/<YYYY>/<MM>/<DD>/<stream>/8Hz/<file>, stream in
{north, south, north-100m, south-100m}. Files are mirrored to <dest>/<YYYY>/<MM>/<DD>/<stream>/.
Downloads go to <name>.part and are renamed only after the file ends with the OFDR magic
(or, for h5, after the transfer completes), so an interrupted run never leaves a truncated file.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from .raw import MAGIC, is_complete

BASE = "http://piweb.ooirsn.uw.edu/das25/data/MultiDAS"
HREF = re.compile(r'href="([^"?/][^"]*)"')


def list_files(day: pd.Timestamp, stream: str, session: requests.Session) -> list[str]:
    url = f"{BASE}/{day:%Y/%m/%d}/{stream}/8Hz/"
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return [f"{url}{h}" for h in HREF.findall(r.text) if h.startswith(("nokia", "downsampled"))]


def crawl(start: str, end: str, streams) -> pd.DataFrame:
    rows = []
    with requests.Session() as s:
        for day in pd.date_range(start, end, freq="D"):
            for st in streams:
                for url in list_files(day, st, s):
                    rows.append(dict(day=day.date(), stream=st, url=url, name=url.rsplit("/", 1)[1]))
    return pd.DataFrame(rows)


def _complete(path: Path) -> bool:
    return path.exists() and (path.suffix == ".h5" or is_complete(str(path)))


def fetch(url: str, dest: Path, retries: int = 3) -> dict:
    if _complete(dest):
        return dict(url=url, path=str(dest), status="exists", bytes=dest.stat().st_size, seconds=0.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        t = time.time()
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        fh.write(chunk)
            if dest.suffix != ".h5":
                with open(part, "rb") as fh:
                    fh.seek(-len(MAGIC), 2)
                    if fh.read(len(MAGIC)) != MAGIC:
                        raise IOError("downloaded file has no trailing OFDR magic (truncated on server?)")
            part.rename(dest)
            return dict(url=url, path=str(dest), status="ok", bytes=dest.stat().st_size, seconds=time.time() - t)
        except Exception as e:  # noqa: BLE001 - report every failure in the manifest
            err = f"{type(e).__name__}: {e}"
            time.sleep(5 * attempt)
    part.unlink(missing_ok=True)
    return dict(url=url, path=str(dest), status=f"failed: {err}", bytes=0, seconds=0.0)
