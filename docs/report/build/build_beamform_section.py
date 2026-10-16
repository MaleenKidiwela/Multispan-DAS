import base64, io, re
from PIL import Image
F = "/data/wsd02/multispan_dvv/outputs_100m/figs/"
def uri(name, w=1500, q=82):
    im = Image.open(F + name).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def fig(name, alt, cap, w=1500):
    return f'  <figure><img src="{uri(name, w)}" alt="{alt}"><figcaption>{cap}</figcaption></figure>\n'

S = '''
  <h3>8 · Multi-span beamforming</h3>
  <p><b>The idea.</b> A conventional DAS stops at the first optical repeater, so it covers one span (here 20–60 km of usable fiber). The Nokia multispan system reads every span of a cable through the repeaters with one laser and one clock. All spans of a cable are therefore one array with consistent timing, 120–300 km across and bent in several directions. The question is what that array can do that a single span cannot: find the direction of incoming waves (beamforming) and the position of nearby earthquakes (back-projection, which is beamforming steered to points instead of directions).</p>
  <ul>
    <li><b>Array elements:</b> the 1-km bins of good channels used for the cross-span work, on true per-chunk times with dropped chunks as gaps. South C3 is excluded (tonal noise).
      <ul>
        <li>Single spans: north A1 26, A2 32, B3 20, C4 11 bins; south A1 23, A2 32, B3 16, C4 14, D4 14 bins.</li>
        <li>One cable's spans together: north 89 bins (a nearly straight line), south loop 99 bins (a 2-D loop, about 120 km E–W by 95 km N–S).</li>
        <li>Both cables: 188 bins.</li>
      </ul></li>
    <li><b>Beam types used:</b>
      <ul>
        <li><b>Bartlett</b> (frequency domain, cross-spectral matrix without autocorrelations, spectra whitened per bin, Rayleigh-wave axial-strain weighting cos²(back-azimuth − cable azimuth)) for ambient noise and teleseismic Rayleigh waves;</li>
        <li><b>delay-and-sum of an onset function</b> (STA/LTA 2 s / 20 s, 0.4–0.9 Hz, each bin normalised to equal weight) for regional P waves;</li>
        <li><b>moveout and back-projection of S envelopes</b> (2–3.6 Hz, 8-Hz stream) for the local Jan 13 earthquake.</li>
      </ul></li>
    <li><b>Scripts:</b> s64 (noise, event Bartlett), s66 (array response), s67 (8-Hz latency), s68 / s68b (regional P), s69 (Rayleigh waves), s65 → s65h (Jan 13).</li>
  </ul>

  <h3>8a · Synthetic test: single span vs multi-span</h3>
  <p>A plane Rayleigh wave (0.35 s/km) with the fiber's axial-strain sensitivity was beamformed with each geometry, for every back-azimuth in 4° steps, with independent noise per bin (amplitude signal-to-noise 0.5, 8 windows).</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Configuration</th><th class="num">Bins</th><th class="num">Mirror peak<br>(1 = as strong as truth)</th><th class="num">Beam width<br>0.2 Hz</th><th class="num">Wrong by &gt; 10°<br>0.1 / 0.2 / 0.5 Hz</th></tr></thead>
    <tbody>
      <tr><td>Single span (south A1, south A2, north A1)</td><td class="num">23–32</td><td class="num">0.97–1.00</td><td class="num">44–76°</td><td class="num">69–88 / 62–84 / 42–77 %</td></tr>
      <tr><td>North cable, all spans</td><td class="num">89</td><td class="num">0.77–0.95</td><td class="num">11°</td><td class="num">30 / 30 / 21 %</td></tr>
      <tr><td>South loop, all spans</td><td class="num">99</td><td class="num">0.57–0.80</td><td class="num">12°</td><td class="num">14 / 20 / 2 %</td></tr>
      <tr><td>Both cables, all spans</td><td class="num">188</td><td class="num">0.45–0.69</td><td class="num">6°</td><td class="num">0 / 0 / 0 %</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>A single span is a line array.</b> It measures only the angle between the wave and the fiber, so a wave from either side gives the same beam (mirror peak at full strength), and broadside waves are poorly resolved. The beam is 30–120° wide.</li>
    <li><b>The spans together</b> break the mirror symmetry because they point in different directions. The south loop does better than the nearly straight north cable.</li>
    <li><b>Both cables</b> never got the direction wrong in this test, provided their relative timing and polarity are right (see 8d).</li>
  </ul>
'''
S += fig("array_response_maps.png", "Array response maps for single spans, the south loop and both cables",
         "<b>Array response</b> for a plane wave from 250° (red +): single spans (south A1, north A1) give a long ridge with an equally strong mirror side; the south loop and both cables give one compact peak. Left: the bins of each span.")
S += fig("array_response_metrics.png", "Back-azimuth error against true back-azimuth for each configuration",
         "<b>Back-azimuth error vs true direction</b> (synthetic, SNR 0.5) for each configuration at 0.1, 0.2 and 0.5 Hz.")
S += '''
  <h3>8b · Ambient-noise direction (south loop)</h3>
  <ul>
    <li><b>Method:</b> hourly Bartlett beams over 982 hours (Dec 23 – Jan 28), 300-s windows, at 0.12, 0.16, 0.20 and 0.25 Hz.</li>
    <li><b>Result:</b> the 0.16–0.20 Hz secondary microseism arrives from the western half-plane (open Pacific) throughout, peaking near 256° at about 2.2 km/s (0.16 Hz) and 1.4 km/s (0.20 Hz). At 0.12 Hz a broad episode on Jan 7–12 also has energy from the east.</li>
    <li><b>Caveat:</b> the loop resolves E–W better than N–S, so the energy smears across 180–340°. The sharp edges at 180° and 360° are array-response artefacts, not source boundaries. Vertical stripes are earthquakes and transients.</li>
    <li><b>Still to do:</b> compare the hour-to-hour noise direction with the within-span dv/v (open question: the more variable dv/v after Jan 13).</li>
  </ul>
'''
S += fig("beamform_hourly.png", "Hourly back-azimuth of ambient noise at four frequencies and apparent velocity of the beam peak",
         "<b>Hourly noise beamforming</b>, south loop: beam power vs back-azimuth (max over slowness) at four frequencies, and the apparent velocity of the beam peak.")
S += '''
  <h3>8c · Regional P waves: the Jan 16 M6.0 Blanco earthquake</h3>
  <ul>
    <li><b>Data:</b> continuous 100-m stream (1.86 Hz), both cables, 0.4–0.9 Hz (above the microseism peak). Times converted to UTC with the 25.2-s latency.</li>
    <li><b>Two methods failed first:</b>
      <ul>
        <li>frequency-domain Bartlett beams looked the same during and before the event: the microseism dominates below 0.25 Hz;</li>
        <li>a delay-and-sum of log envelopes (window P − 20 s to P + 40 s) was 43° off with an impossible slowness (0.04 s/km): the later S and Lg energy and strongly amplified sites dominate the sum.</li>
      </ul></li>
    <li><b>What works:</b> delay-and-sum of the STA/LTA onset function, every bin normalised to equal weight, searching the origin of the plane wave within ±15 s of the predicted first P. This aligns the arrival onsets instead of the energy.</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Configuration</th><th class="num">Bins</th><th class="num">Back-azimuth error</th><th class="num">Slowness (s/km)</th></tr></thead>
    <tbody>
      <tr><td><b>South loop, all spans</b></td><td class="num">99</td><td class="num"><b>−1°</b></td><td class="num">0.16</td></tr>
      <tr><td><b>Both cables</b></td><td class="num">188</td><td class="num"><b>−3°</b></td><td class="num">0.14</td></tr>
      <tr><td><b>North, all spans</b></td><td class="num">89</td><td class="num"><b>−6°</b></td><td class="num">0.20</td></tr>
      <tr><td>Single spans (9)</td><td class="num">11–32</td><td class="num">+2 (S A1), +10 (N C4), +15 (N A1), +27 (S B3), +28 (N A2), −46 (S C4), −75 (S D4), +101 (S A2), +142 (N B3)</td><td class="num">0.08–0.20</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Result:</b> every multi-span configuration is within 6°; 7 of the 9 single spans are more than 10° off, two by more than 100°. The P slowness of the multi-span beams (0.14–0.16 s/km) is the expected Pn/Pg value; single spans often sit on the search limits.</li>
    <li><b>Only this event is recorded in the P band.</b> All M4–5 regional events (4–13°) and all M6+ teleseisms have beam SNR ≈ 1 (beam peak / pre-event beam peak). Teleseismic P arrives almost vertically, which barely strains a horizontal fiber, and small regional events have little energy below the 0.93-Hz Nyquist frequency of this stream.</li>
  </ul>
'''
S += fig("beam_m6_single_vs_multi.png", "Map of beam directions from each span and each multi-span array for the Jan 16 M6.0, with error bars",
         "<b>Jan 16 M6.0: direction per array.</b> Map: arrows from each span's centroid (coloured) and each multi-span array (blue, thick) to the estimated back-azimuth; grey dashed = true direction. Right: |error| per configuration (blue multi-span, red single span; dotted line 10°).")
S += '''
  <h3>8d · Teleseismic Rayleigh waves, and the north-cable polarity</h3>
  <p>Large distant earthquakes are recorded mainly as long-period surface waves. At 0.04–0.07 Hz a Rayleigh wave is about 55–100 km long, longer than any single span, so a single span cannot resolve it at all, while the multi-span array (120–300 km) can. This is the cleanest test of the idea with many events.</p>
  <ul>
    <li><b>Events:</b> all ComCat M ≥ 6.0 at 20–150° plus M ≥ 5.0 within 20°, Dec 23 – Jan 28 (15 events). Six fell in stretches where gaps from dropped chunks left too few clean windows, and one had no file. Nine were beamformed; eight are recorded (both-cable beam SNR ≥ 1.3).</li>
    <li><b>Method:</b> Bartlett beams at 0.04, 0.05, 0.06 and 0.07 Hz, averaged; 300-s sub-windows (50 % overlap, ≥ 90 % valid samples) in the Rayleigh group window 3.8 → 3.0 km/s. A first pass used 4.2 → 3.0 km/s, which includes Love waves, whose along-fiber strain changes sign with cable orientation and spoils a coherent beam.</li>
    <li><b>The north cable has the opposite polarity to the south cable.</b> Each cable alone pointed roughly right, but both together were systematically 40–66° off. Flipping the sign of the north data fixed it:</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Event</th><th class="num">Distance</th><th class="num">Both cables, north as recorded<br>error / beam SNR</th><th class="num">Both cables, north flipped<br>error / beam SNR</th></tr></thead>
    <tbody>
      <tr><td>Dec 27 M6.6</td><td class="num">88°</td><td class="num">+41° / 2.0</td><td class="num">+5° / 2.6</td></tr>
      <tr><td>Dec 28 M6.2</td><td class="num">68°</td><td class="num">+44° / 1.8</td><td class="num">−132° / 1.4</td></tr>
      <tr><td>Dec 31 M6.0</td><td class="num">64°</td><td class="num">+60° / 1.2</td><td class="num">−18° / 1.4</td></tr>
      <tr><td>Jan 7 M6.4</td><td class="num">97°</td><td class="num">−66° / 3.2</td><td class="num">−14° / 2.8</td></tr>
      <tr><td>Jan 10 M6.4</td><td class="num">100°</td><td class="num">+52° / 2.0</td><td class="num">−10° / 5.3</td></tr>
      <tr><td>Jan 13 M6.2</td><td class="num">58°</td><td class="num">+15° / 1.9</td><td class="num">−9° / 4.2</td></tr>
      <tr><td>Jan 21 M6.1</td><td class="num">76°</td><td class="num">+46° / 4.1</td><td class="num">−6° / 6.9</td></tr>
      <tr><td>Jan 22 M6.2</td><td class="num">49°</td><td class="num">+45° / 2.5</td><td class="num">−3° / 3.5</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Polarity:</b> with the flip the two cables add coherently (higher beam SNR in 7 of 9 events) and the direction is right. Dec 28 is the exception (weakest event, beam SNR 1.4). The likely cause is a sign convention in the phase-to-strain conversion or in the optical setup of one interrogator. It does not affect any dv/v product, because no product correlates the north cable with the south cable. It must be applied whenever the two cables are combined, and should be confirmed with the data providers.</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Configuration (north flipped)</th><th class="num">Median |error|</th><th class="num">Within 10°</th><th class="num">Within 20°</th></tr></thead>
    <tbody>
      <tr><td><b>Both cables, all spans</b></td><td class="num"><b>9°</b></td><td class="num"><b>62 %</b></td><td class="num"><b>88 %</b></td></tr>
      <tr><td>South loop, all spans</td><td class="num">20°</td><td class="num">38 %</td><td class="num">50 %</td></tr>
      <tr><td>North cable, all spans (nearly straight)</td><td class="num">37°</td><td class="num">0 %</td><td class="num">0 %</td></tr>
      <tr><td>Best single span (north C4)</td><td class="num">10°</td><td class="num">50 %</td><td class="num">62 %</td></tr>
      <tr><td>Other single spans (8)</td><td class="num">24–124°</td><td class="num">0–12 %</td><td class="num">0–25 %</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Result (8 events):</b> combining all spans of both cables gives the right direction for 7 of 8 earthquakes. Single spans mostly fail, as the synthetic test predicted. The north cable alone, being nearly straight, behaves like a long line array and is no better than a single span. North C4, the farthest-offshore span, is the one good single span; it is also the quietest site.</li>
    <li><b>Caveats:</b>
      <ul>
        <li>an event counts as recorded when the both-cable beam rises above its pre-event level, which favours that configuration; the single-span and one-cable rows use the same 8 events;</li>
        <li>only 8 events, all teleseisms from the west and south-west;</li>
        <li>gaps from dropped chunks removed six events; the November data and a gap-tolerant beam would add more.</li>
      </ul></li>
  </ul>
'''
S += fig("surface_wave_beam_examples_sign-1.png", "Rayleigh-wave beams from a single span and from both cables for four teleseisms",
         "<b>Rayleigh-wave beams, 0.04–0.07 Hz</b> (north polarity corrected). Top: south A2 alone, a broad ridge with a mirror side. Bottom: both cables, a compact peak. Red dashed: true back-azimuth.")
S += fig("surface_wave_beam_sign-1.png", "Back-azimuth error per configuration for the recorded teleseisms",
         "<b>Rayleigh waves: back-azimuth error per configuration</b> for the recorded teleseisms (black bar = median) and the share of events more than 20° off.")
S += '''
  <h3>8e · Latency of the 8-Hz stream, measured without an earthquake</h3>
  <ul>
    <li><b>Why it matters:</b> back-projection across both cables, and combining DAS with seismometers, needs absolute DAS times.</li>
    <li><b>Method:</b> on Jan 13 the 8-Hz stream (7.45 Hz) and the 100-m stream (1.86 Hz, latency 25.2 s from the colocated seismometer HYSB1) recorded the same fibers at the same time: north 01:43–08:03, south 01:49–05:33 UTC. The same fiber positions were cross-correlated between the streams in the microseism band (0.1–0.5 Hz), in 20-min windows.</li>
    <li><b>Result:</b> the 8-Hz time stamps are 19.408 s (north, 29 positions, median CC 0.91, scatter 5 ms) and 19.397 s (south, 25 positions, CC 0.81, scatter 8 ms) earlier than the 100-m stamps. <b>8-Hz latency = 5.8 s on both cables</b>, equal to within 10 ms. The absolute value inherits the ±~3 s uncertainty of the 25.2-s calibration.</li>
    <li><b>Correction:</b> the earlier "13–14 s" came from aligning the Jan 13 earthquake on its PNSN origin and location, which is circular. It is superseded (also in item 3 and the clock note).</li>
  </ul>

  <h3>8f · Locating the Jan 13 ML 2.0 with the spans</h3>
  <ul>
    <li><b>Reference location is not ground truth:</b> PNSN located this event with 14 onshore stations, nearest 39 km, azimuthal gap 250°, depth fixed near the surface. Its position is a comparison, not the answer.</li>
    <li><b>Where the event is seen:</b> stacking each 10-km stretch of every span shows the S wave on only four stretches, all on the first two spans near the coast: north 20–30 km and 67–100 km, south 20–50 km and 98–120 km (stack SNR 50 to more than 600). The curved southern loop (south B3, C4, D4, 60–90 km away) and the far north spans show nothing; a delay-and-sum of all 113 southern-loop bins reaches only 3.5σ, the same as for a deliberately wrong source position. At most they record a faint water-borne T-phase about 50 s after the origin.</li>
  </ul>
'''
S += fig("das_location_jan13_span_stacks.png", "Median S envelopes per 10-km stretch of every span on both cables for the Jan 13 earthquake",
         "<b>Jan 13 ML 2.0 per 10-km stretch</b>: median 2–3.6 Hz envelope of each stretch (red: SNR &gt; 8). Only stretches on the first two spans near the coast record the S wave.")
S += '''
  <ul>
    <li><b>Direction from the spans (beamforming per stretch):</b> cross-correlating the envelopes of neighbouring channels gives the moveout along each lit stretch. North 67–100 km: +0.224 ± 0.014 s/km (envelope CC 0.92); south 98–120 km: +0.30 ± 0.02 s/km (CC 0.93), nearly along the fiber. Both put the source east of about 124.75°W. The two stretches near the coast give incoherent envelopes (CC 0.40–0.45) and were not usable.</li>
    <li><b>Distance needs time differences between stretches</b>, which require the latency (8e). Joint fit of the per-channel S times (UTC via 5.8 s) with the Hydrate Ridge seismometer picks (PhaseNet: HYS11 P; HYS11, HYS12, HYS13, HYSB1 S):
      <ul>
        <li>DAS only: unconstrained N–S (region about 4000 km²);</li>
        <li>seismometers only: a distance arc around Hydrate Ridge (about 6500 km²), origin about 4 s after the PNSN origin;</li>
        <li><b>joint: 45.06°N 124.53°W, 16 km from PNSN, region about 1300 km²</b>; depth and velocity sit on the grid limits, and shifting the latency by ±2 s moves the solution 17–21 km.</li>
      </ul></li>
    <li><b>Conclusion for this event:</b> the multi-span beam gives a robust direction, but only two far stretches record a coherent S wavefront, so distance and depth are not robust. Larger local events, recorded on more spans, are needed to test location properly.</li>
    <li><b>Approaches rejected along the way</b> (kept in the scripts for the record): STA/LTA onset picks (they saturate on noise bursts, and the first version searched around the catalogue-predicted S, which is circular) and envelope back-projection over all bins (the many noise-only bins dominate the stack).</li>
  </ul>
'''
S += fig("das_location_jan13_moveout.png", "Envelope images and moveout fits along four fiber stretches",
         "<b>S-envelope moveout along each lit stretch</b>: envelopes (top) and per-channel delays with line fits (bottom). The two far stretches give clean moveouts; the near stretches do not.")
S += fig("das_location_jan13_final.png", "Location misfit maps from DAS only, seismometers only and joint",
         "<b>Jan 13 location</b>: DAS only (S times on the coherent stretches, UTC via the measured 5.8-s latency), Hydrate Ridge seismometers only, and joint. White contour: misfit within 0.3 s of the minimum. Grey star: PNSN (onshore stations only).")
S += '''
  <h3>8g · Where this leaves multi-span beamforming</h3>
  <ul>
    <li><b>Direction:</b> shown on real data. For the regional M6.0 (P waves) and for 8 teleseisms (Rayleigh waves), combining spans gives the right direction where single spans mostly fail, as the synthetic array response predicts. Both cables together are best, once the north polarity is corrected.</li>
    <li><b>Location:</b> not yet shown. The only local event with 8-Hz data, an ML 2.0, is recorded coherently on just two far stretches.</li>
    <li><b>Next:</b>
      <ul>
        <li>download 8-Hz data for every local M ≥ 2.5 event and the uncatalogued ones found by the seismometer scan, and repeat the multi-span back-projection;</li>
        <li>time-resolved back-projection of the Jan 16 M6.0, to test whether the array sees the rupture direction along the Blanco Transform;</li>
        <li>compare the hourly noise direction with the within-span dv/v;</li>
        <li>repeat the teleseismic test on the November data, with a gap-tolerant beam.</li>
      </ul></li>
  </ul>
'''
s = open("report/index.html").read()
# corrections of the circular D512 latency
s = s.replace("<li>The detection gives that setup's latency: about 13–14 s.</li>",
              "<li>An alignment on the PNSN origin suggested a latency of about 13–14 s for this setup. That estimate was circular and is superseded: the latency measured against the simultaneous 100-m stream is 5.8 s (item 8e).</li>")
s = s.replace("<li>about 13–14 s for the D512 8-Hz stream on Jan 12–13, from the Jan 13 earthquake.</li>",
              "<li>+5.8 s for the 8-Hz stream (D512) on Jan 13, the same on both cables, measured against the simultaneous 100-m stream (item 8e); an earlier 13–14 s from the Jan 13 earthquake was circular and is superseded.</li>")
s = s.replace("(211.7 m channels, 2–3.6 Hz envelope, true times, 13.5 s latency removed)", "(211.7 m channels, 2–3.6 Hz envelope, true times; 13.5 s was removed here, but the measured latency is 5.8 s, item 8e)")
assert s.count("13–14 s") >= 2
# insert section 8 at the end of the follow-ups section
anchor = "Full note: <code>docs/DAS_clock_note.md</code>.</figcaption></figure>\n</section>"
assert anchor in s
s = s.replace(anchor, anchor[:-len("</section>")] + S + "</section>")
s = s.replace("After the main dv/v products, seven follow-up items were run", "After the main dv/v products, eight follow-up items were run")
# summary bullet
s = s.replace('''    <li>Within-span dv/v is very stable on Dec 23 – Jan 13''', '''    <li><b>Multi-span beamforming</b> (all spans on one laser and clock) gets earthquake directions right where single spans mostly fail: the Jan 16 M6.0 within 1–6° (single spans 2–142°), and 7 of 8 teleseisms within 20° with both cables (median 9°). Combining the cables required flipping the polarity of the north cable. Locating a small local earthquake was not yet robust (follow-up 8).</li>
    <li>Within-span dv/v is very stable on Dec 23 – Jan 13''')
# limits
s = s.replace('''    <li>November data (8-Hz stream''', '''    <li>The north cable's polarity is opposite to the south cable's (follow-up 8d); this must be applied whenever the two cables are combined, and should be confirmed with the data providers.</li>
    <li>DAS-only location of small local earthquakes is limited to direction: the Jan 13 ML 2.0 gives a coherent S wavefront on only two stretches.</li>
    <li>November data (8-Hz stream''')
# repro scripts
s = s.replace("s38/s42/s47 (correlation figures).</li>", "s38/s42/s47 (correlation figures); beamforming s64 (noise), s66 (array response), s67 (8-Hz latency), s68 / s68b (regional P), s69 (Rayleigh waves), s65–s65h (Jan 13 location).</li>")
open("report/index.html", "w").write(s)
print("size MB", len(s) / 1e6)
