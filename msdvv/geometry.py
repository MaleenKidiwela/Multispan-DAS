"""Channel geometry: repeater picks in optical distance -> route distance -> lat/lon/depth.

Repeaters appear in the data as a single-channel amplitude spike followed by a
drop in median amplitude (the official notebook pins on the amplitude drop).
Route files and repeater coordinates are from github.com/uwfiberlab/OOI_DAS_2025.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from . import ROOT

RES = ROOT / "resources"
GEOD = Geod(ellps="WGS84")
OPT_TO_ROUTE_GUESS = 0.96  # route km per optical km, first guess (fiber over-length)


def read_route(cable: str) -> pd.DataFrame:
    r = pd.read_csv(
        RES / f"RSNcableroute_{cable}_nostubs_withdepths.xy",
        sep=r"\s+",
        header=None,
        names=["lat", "lon", "depth"],
        comment=">",
    )
    r["depth"] = r.depth.abs()  # last south segment uses negative-down convention
    _, _, d = GEOD.inv(r.lon.values[:-1], r.lat.values[:-1], r.lon.values[1:], r.lat.values[1:])
    r["route_km"] = np.r_[0.0, np.cumsum(d)] / 1e3
    return r


def repeaters(cable: str) -> pd.DataFrame:
    rep = pd.read_csv(RES / "RSNrepeaterlocations_latlon.txt", sep=r"\s+", header=None, usecols=[0, 1], names=["lat", "lon"]).dropna()
    rep = rep.iloc[3:] if cable == "north" else rep.iloc[:3]  # order as in the official notebook
    route = read_route(cable)
    km = []
    for la, lo in zip(rep.lat, rep.lon):
        _, _, d = GEOD.inv(np.full(len(route), lo), np.full(len(route), la), route.lon.values, route.lat.values)
        km.append(route.route_km.values[np.argmin(d)])
    rep = rep.reset_index(drop=True)
    rep["route_km"] = km
    rep["rep_id"] = np.arange(1, len(rep) + 1)
    return rep


def amplitude_profile(data: np.ndarray) -> np.ndarray:
    x = data - data.mean(1, keepdims=True)
    return np.log10(np.median(np.abs(x), axis=1) + 1e-30)


def pick_repeaters(lm: np.ndarray, chans: pd.DataFrame, cable: str, search_km: float = 8.0) -> pd.DataFrame:
    """Find the spike and/or drop signature near each route-predicted repeater."""
    rep = repeaters(cable)
    out = []
    for r in rep.itertuples():
        pred = r.route_km / OPT_TO_ROUTE_GUESS
        cand = chans[(np.abs(chans.opt_km - pred) < search_km) & (chans.pos_in_block >= 4) & (chans.pos_in_block < chans.block_len - 4)]
        best, best_score = None, 0.0
        for c in cand.ch.values:
            before = np.median(lm[c - 4 : c])
            after = np.median(lm[c + 1 : c + 5])
            spike = lm[c] - max(before, after)
            drop = before - after
            score = max(spike, 0) + max(drop, 0)
            # 211 m spacing: one-channel spike then drop; ~106 m spacing: often only the drop
            if (spike > 0.25 or drop > 0.4) and score > best_score:
                best, best_score = c, score
        if best is not None:
            row = chans.loc[chans.ch == best].iloc[0]
            out.append(dict(rep_id=r.rep_id, route_km=r.route_km, ch=int(best), opt_km=row.opt_km, block=row.block, score=best_score))
    return pd.DataFrame(out)


def fixed_picks(chans: pd.DataFrame, cable: str, opt_km) -> pd.DataFrame:
    """Repeater picks at known optical distances (e.g. from a coarser stream of the same fiber)."""
    rep = repeaters(cable)
    rows = []
    for r, o in zip(rep.itertuples(), opt_km):
        c = chans.iloc[int(np.argmin(np.abs(chans.opt_km.values - o)))]
        if abs(c.opt_km - o) < 0.5:
            rows.append(dict(rep_id=r.rep_id, route_km=r.route_km, ch=int(c.ch), opt_km=c.opt_km, block=c.block, score=np.nan))
    return pd.DataFrame(rows)


def build_geometry(lm: np.ndarray, chans: pd.DataFrame, cable: str, edge_pad: int = 3, search_km: float = 8.0,
                   fixed_opt_km=None):
    picks = fixed_picks(chans, cable, fixed_opt_km) if fixed_opt_km else pick_repeaters(lm, chans, cable, search_km)
    if len(picks) >= 2:
        a, b = np.polyfit(picks.opt_km, picks.route_km, 1)
    else:
        a, b = OPT_TO_ROUTE_GUESS, 0.0
    route = read_route(cable)
    g = chans.copy()
    g["route_km"] = np.clip(a * g.opt_km + b, 0, route.route_km.max())
    for k in ("lat", "lon", "depth"):
        g[k] = np.interp(g.route_km, route.route_km, route[k])
    rep = repeaters(cable)
    # repeater span id: 1 before the first repeater, etc. (span boundaries from picks when available)
    rep_opt = rep.route_km.map(lambda km: (km - b) / a).values
    if len(picks):
        rep_opt[picks.rep_id.values - 1] = picks.opt_km.values
    g["span"] = 1 + np.searchsorted(rep_opt, g.opt_km.values, side="right")
    near_edge = (g.pos_in_block < edge_pad) | (g.pos_in_block >= g.block_len - edge_pad)
    near_rep = np.zeros(len(g), bool)
    for c in picks.ch.values if len(picks) else []:
        near_rep |= np.abs(g.ch.values - c) <= edge_pad
    g["near_edge"] = near_edge.values
    g["near_repeater"] = near_rep
    g.loc[g.ch == 0, "near_edge"] = True  # dead first channel
    g["usable_geom"] = ~(g.near_edge | g.near_repeater)
    return g, picks, (a, b)
