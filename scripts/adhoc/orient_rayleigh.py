import sys, glob; sys.path.insert(0, "/home/seismic/Multispan_DV")
import numpy as np, pandas as pd
from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.geodetics import gps2dist_azimuth
from scipy.signal import hilbert
ev = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/events.csv")
ev = ev[ev.mag >= 5.8].drop_duplicates("id").sort_values("time")
inv = read_inventory("/data/wsd02/multispan_dvv/obs/continuous/OO_response.xml")
crd = {s.code: (s.latitude, s.longitude) for n in inv for s in n}
rows = []
for e in ev.itertuples():
    t0 = UTCDateTime(pd.Timestamp(e.time).to_pydatetime())
    for sta in ("HYSB1", "HYS14"):
        dist, az, baz = gps2dist_azimuth(crd[sta][0], crd[sta][1], e.latitude, e.longitude)
        tR = t0 + dist / 1e3 / 3.9
        st = Stream()
        for d in sorted({(tR - 600).strftime("%Y%m%d"), (tR + 1200).strftime("%Y%m%d")}):
            for fn in glob.glob(f"/data/wsd02/multispan_dvv/obs/continuous/{d}/OO.{sta}.HH.mseed"):
                st += read(fn)
        st.merge(fill_value=None)
        try:
            st.trim(tR - 300, tR + 900)
            st = st.split().merge(fill_value=0)
            st.detrend("demean"); st.filter("bandpass", freqmin=0.02, freqmax=0.05, corners=4, zerophase=True)
            Z, N, E = (st.select(component=c)[0].data.astype(float) for c in "ZNE")
            n = min(len(Z), len(N), len(E)); Z, N, E = Z[:n], N[:n], E[:n]
        except Exception as ex:
            continue
        Zh = np.imag(hilbert(Z))  # Z shifted by 90 deg
        best = []
        for th in range(0, 360, 2):
            R = np.cos(np.radians(th)) * N + np.sin(np.radians(th)) * E
            best.append(np.dot(Zh, R) / np.sqrt(np.dot(Zh, Zh) * np.dot(R, R)))
        best = np.array(best); k = int(np.argmax(best))
        est_baz = (2 * k + 180) % 360  # radial points away from the source: R(theta) maximal along propagation -> baz = theta + 180
        rows.append(dict(event=f"{str(e.time)[:16]} M{e.mag:.1f}", sta=sta, dist_deg=round(dist / 111195, 1), baz=round(baz), est_baz=est_baz,
                         cc=round(best[k], 2), corr=((est_baz - baz + 180) % 360) - 180))
R = pd.DataFrame(rows)
pd.set_option("display.width", 200); print(R.to_string(index=False))
for s, g in R[R.cc > 0.6].groupby("sta"):
    print(f"{s}: orientation offset (estimated - true back-azimuth), events with CC > 0.6: median {g['corr'].median():+.0f} deg, spread {g['corr'].std():.0f} deg, n={len(g)}")
