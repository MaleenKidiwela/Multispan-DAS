import sys; sys.path.insert(0, "/home/seismic/Multispan_DV")
import struct, os, msgpack, numpy as np, pandas as pd
from msdvv.raw import MAGIC, _decoder, read_header
def chunk_meta(p):
    with open(p, "rb") as f:
        f.seek(-(len(MAGIC) + 4), os.SEEK_END); (back,) = struct.unpack("<I", f.read(4))
        f.seek(-(len(MAGIC) + 4 + back), os.SEEK_END); buf = f.read(back)
    u = msgpack.Unpacker(raw=False, object_hook=_decoder, max_buffer_size=10**8); u.feed(buf)
    objs = list(u)
    return objs[1], objs[2]
inv = pd.read_csv("/data/wsd02/multispan_dvv/outputs_100m/inventory.csv"); inv = inv[inv.cable == "south"]
for when in sys.argv[1:]:
    r = inv.iloc[(np.abs(inv.t0 - pd.Timestamp(when, tz="UTC").timestamp())).argmin()]
    h = read_header(r.path); fs, m = h["attrs"]["new_fs"], h["m"]
    w, cm = chunk_meta(r.path)
    t0 = float(h["attrs"]["stream.proc_start"])
    ps = np.array([t0] + [c.get("stream.proc_start", np.nan) for c in cm[1:]])
    gi = np.array([w[0]["idx"]] + [c.get("stream.gpu_idx", np.nan) for c in cm[1:]], float)
    nominal = t0 + np.arange(len(ps)) * m / fs
    dps, dgi = np.diff(ps), np.diff(gi)
    lag = ps - nominal
    print(f"{pd.Timestamp(t0, unit='s'):%m-%d %H:%M} file: {len(ps)} chunks of {m} samples ({m / fs:.3f} s); "
          f"proc_start step median {np.nanmedian(dps):.3f} s, gpu_idx step median {np.nanmedian(dgi):.0f}")
    big = np.where(dgi > 300)[0]
    print(f"   gpu_idx jumps > 300: {len(big)}; at h into file / idx step / proc_start step:")
    print("   " + "; ".join(f"{(nominal[i] - t0) / 3600:.2f}h: +{dgi[i]:.0f} / {dps[i]:.1f}s" for i in big[:30]))
    print(f"   proc_start - nominal at end of file: {lag[-1]:+.1f} s; at 1, 2, 3, 4, 5, 6, 7 h: " +
          ", ".join(f"{lag[int(k * 3600 * fs / m)]:+.0f}" for k in range(1, 8) if int(k * 3600 * fs / m) < len(lag)))
