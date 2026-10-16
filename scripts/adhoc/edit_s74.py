p="/home/seismic/Multispan_DV/scripts/s74_amity_locate.py"; s=open(p).read()
def R(a,b):
    global s
    assert a in s, a[:60]; s=s.replace(a,b)
R('SNR_SEG, MAXSH, DMIS = 1.5, 10.0, 0.3','SNR_SEG, MAXSH, DMIS, CC_STR = 1.5, 10.0, 0.3, 0.8\nimport os\nMODE = os.environ.get("MODE", "wavefront")  # wavefront: one free offset per stretch (shape only); common: one offset for all')
R('''D = pd.DataFrame(BINS)
S = pd.DataFrame(STR)''','''D = pd.DataFrame(BINS)
S = pd.DataFrame(STR)
S["used"] = S.cc_med >= CC_STR  # coherent stretches only (near-shore stretches: CC ~0.6, one 'onset' before the origin)
D = D[D.stretch.isin(S.stretch[S.used])].reset_index(drop=True)''')
R('''                res = Q.t.values[None] - r / v
                res = res - np.median(res, axis=1, keepdims=True)''','''                res = Q.t.values[None] - r / v
                if MODE == "wavefront":  # free offset per stretch: only the wavefront shape inside each stretch is used
                    for gm in grp:
                        res[:, gm] -= np.median(res[:, gm], axis=1, keepdims=True)
                else:
                    res = res - np.median(res, axis=1, keepdims=True)''')
R('''RUNS.update({f"{k} (single stretch)": (D.stretch == k).values for k in S.stretch})''','''RUNS.update({f"{k} (single stretch)": (D.stretch == k).values for k in S.stretch[S.used]})''')
R('open(f"{OUT}/amity_location.json", "w")','open(f"{OUT}/amity_location_{MODE}.json", "w")')
R('json.dump(dict(event=EV, stretches=STR','json.dump(dict(event=EV, mode=MODE, stretches=STR')
R('fig.savefig(f"{OUT}/figs/amity_location.png"','fig.savefig(f"{OUT}/figs/amity_location_{MODE}.png"')
R('fig.suptitle("Nov 23 2025 M3.2 Amity (ground truth, onshore): DAS-only location from S wavefronts on the lit stretches, one free time offset")',
  'fig.suptitle("Nov 23 2025 M3.2 Amity (ground truth, onshore): DAS-only location from S wavefronts on the coherent stretches (" + ("wavefront shape only, free offset per stretch" if MODE == "wavefront" else "one common time offset") + ")")')
R("""   ONE free time offset shared by both cables (= origin time + the unknown latency of this setup; both cables share
   the latency, s67). Each stretch weighted equally.""","""   MODE=wavefront (default): one free time offset per stretch, i.e. only the wavefront shape inside each stretch is used
   (robust to clock offsets between files and to which phase sets a stretch's onset) - multi-span triangulation;
   MODE=common: one time offset for all stretches (origin + latency; both cables share the latency, s67).
   Only stretches with median envelope CC >= 0.8 are used. Each stretch weighted equally.""")
open(p,"w").write(s); print("edited")
