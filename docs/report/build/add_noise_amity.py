from embed_helpers import fig
N = '''
  <p><b>Noise direction vs within-span dv/v</b> (<code>s76_noise_direction_vs_dvv.py</code>). Each hour's beam maximum (back-azimuth as east and north components, beam power, apparent velocity at 0.16, 0.20 and 0.25 Hz) was averaged over the same trailing 24 or 48 h as the dv/v and correlated with the within-span dv/v (b1 and b2; cable averages and south spans), using an AR(1) effective sample size as in item 4.</p>
  <ul>
    <li><b>No relation:</b> no correlation with the cable-average dv/v is significant; 5 of 1,008 tests pass overall, which is what chance alone would give.</li>
    <li><b>The after-Jan 13 variability is not a noise-source effect:</b> after Jan 13 the noise direction became <i>more</i> stable (the east–west swings at 0.16–0.20 Hz fell to 0.4–0.5 of their earlier standard deviation; beam power and apparent velocity unchanged), while dv/v became more variable.</li>
    <li><b>Limit:</b> 48-h stacks give only about 5–8 independent samples per period, so only strong relations (|r| &gt; ~0.7) could be detected.</li>
  </ul>
'''
N += fig("noise_direction_vs_dvv.png", "Within-span dv/v with the east component of the noise direction and the beam power over time",
         "<b>Within-span dv/v (b2, 48 h) and the noise field</b> from the south-loop beam: east component of the noise direction (−1 = from the west) and beam maximum, 48-h trailing means. Dotted line: Jan 13 18:00.")
A = '''
  <p><b>A ground-truth local test: Nov 23 2025 M3.2 Amity, Oregon</b> (PNSN, 75 stations, azimuthal gap 22°; <code>s73</code>, <code>s74</code>). The original 7.45-Hz files were re-downloaded for this event. It is recorded on many spans, including the curved southern loop: north A2 (65–95 km), south A2 (100–130 km), south B3, C4 and D4, at 120–190 km from the event; near-shore stretches are incoherent.</p>
  <ul>
    <li><b>Location not robust:</b> using only the wavefront shape in each coherent stretch (one free offset per stretch), the uncertainty region covers about 60,000 km² with the best point on the grid edge. With one common time offset the best point is 94 km from PNSN, with a 4-s misfit.</li>
    <li><b>Why:</b> at 120–190 km the 2–3.6 Hz envelopes on the far stretches are long, emergent wave trains (probably Lg and coda, not a clean direct S), and the stretch onsets are mutually inconsistent (the stretch nearest the event triggers last). These November files also carry no per-chunk timing, so dropped chunks cannot be corrected.</li>
    <li><b>Conclusion:</b> for small earthquakes the multi-span array gives detections on many spans but not a reliable location with envelope-based times. A direct-phase picker with waveform correlation between stretches, or a larger local event, would be needed.</li>
  </ul>
'''
A += fig("amity_m32_section.png", "Envelope sections of the Amity M3.2 on both cables at two frequency bands",
         "<b>Nov 23 M3.2 Amity on the 8-Hz stream</b>: log envelope along both cables (1–2 and 2–3.6 Hz) with predicted P (cyan) and S (green) without latency; the event is recorded on north A2, south A2 and the southern-loop spans B3, C4, D4.", w=1300)
s = open("report/index.html").read()
a1 = "  <h3>8c · Regional P waves: the Jan 16 M6.0 Blanco earthquake</h3>"
assert a1 in s
s = s.replace(a1, N + a1)
s = s.replace("    <li><b>Still to do:</b> compare the hour-to-hour noise direction with the within-span dv/v (open question: the more variable dv/v after Jan 13).</li>\n", "")
a2 = "  <h3>8g · Locating the Jan 16 M6.0"
assert a2 in s
s = s.replace(a2, A + a2)
s = s.replace("<li>compare the hourly noise direction with the within-span dv/v;</li>", "")
s = s.replace("s72 (rupture test).</li>", "s72 (rupture test), s73–s75 (Amity M3.2: sections, location, fault crossings), s76 (noise direction vs dv/v).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
