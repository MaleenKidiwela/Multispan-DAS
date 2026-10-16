from embed_helpers import fig
H = '''
  <h3>8g++ · Can the spans see the rupture (fault) of the M6.0? No</h3>
  <p>Time-resolved back-projection images large ruptures by tracking where high-frequency energy comes from second by second. It was run with all spans of both cables (188 bins), 0.4–0.9 Hz energy envelopes with equal weight per bin, the same timing and model as the location in 8g, and the reference hypocentre from that location (43.75°N 128.35°W). The beam maximum was tracked every 1 s from −4 to +24 s and compared with 20 synthetic point sources: the same array, the observed per-bin timing scatter (4.3 s) and real pre-event noise. Script: <code>s72_rupture_backproject_m6.py</code>.</p>
  <ul>
    <li><b>Observation:</b> the beam maximum does not stay near the hypocentre. It moves to the edges of the search grid and slides along the range arc. The apparent along-strike speed (+7.8 km/s) lies outside the synthetic point-source range (−6.2 to +1.7 km/s), but it is faster than any rupture on a transform fault and comes from the maximum hopping between grid edges. It is not a rupture signal.</li>
    <li><b>Why:</b> with energy envelopes at 0.4–0.9 Hz the array resolves distance (an arc) far better than position along the arc. A 4.3-s timing scatter blurs the image by about 35 km at P velocity, and even a synthetic point source appears to move by up to 76 km (95th percentile), while the M6.0 rupture is only about 15–25 km long.</li>
    <li><b>What it would take:</b> a larger rupture (M7+, 50–100+ km) within a few hundred km; higher frequencies with timing scatter of about 1 s (the 8-Hz stream, archived here only in snapshots) and station corrections; or a denser, better-calibrated array. A first version that took the reference hypocentre from the energy maximum instead was biased 233 km away, because the energy peaks 5–15 s after the onset, and was rejected.</li>
  </ul>
'''
H += fig("rupture_m6.png", "Time-resolved back-projection of the M6.0: beam maximum positions, positions versus time, and migration speed against synthetic point sources",
         "<b>Rupture test for the Jan 16 M6.0</b> (all spans, both cables). Left: beam maximum every 1 s (colour = time, size = power) over the summed 0–6 s image, with the Blanco Transform strike. Middle: beam-maximum position along (solid) and across (dotted) strike against 20 synthetic point sources (red). Right: along-strike migration speed, observed vs synthetic. The jumps follow grid edges and the range arc, not the fault.")
s = open("report/index.html").read()
anchor = "  <h3>8h · Where this leaves multi-span beamforming</h3>"
s = s.replace(anchor, H + anchor)
s = s.replace("<li>time-resolved back-projection of the Jan 16 M6.0, to test whether the array sees the rupture direction along the Blanco Transform;</li>",
              "<li>rupture imaging needs a larger or closer earthquake, or 8-Hz data with station corrections (the M6.0 test in 8g++ was negative);</li>")
s = s.replace("s71 (joint DAS + seismometer back-projection).</li>", "s71 (joint DAS + seismometer back-projection), s72 (rupture test).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
