from embed_helpers import fig
H = '''
  <h3>8g+ · Adding the seismometers to the back-projection (origin time free)</h3>
  <p>The seismometers carry absolute UTC time and sit in other directions, so in principle they remove the need for a catalogue origin time. All elements were stacked together with each group weighted equally: north cable (89 bins), south cable (99), Hydrate Ridge seismometers (HYSB1, HYS11–14), Axial Seamount seismometers (AXAS1/2, AXBA1, AXCC1, AXEC1–3, AXID1) and 7 coastal and Coast Range land stations (JEDS, MPO, UMPQ, SEAS2, WPO, RADR, NEMA; HEBO had no data). The origin time was left free, so no catalogue information is used. Script: <code>s71_backproject_m6_joint.py</code>.</p>
  <ul>
    <li><b>DAS timing checked on the event itself:</b> the P onset on the 16 south-cable bins within 10 km of the colocated seismometer HYSB1 comes 3.75 s after HYSB1 (spread 3.3–5.0 s). This independent, catalogue-free check says the DAS needs 3.75 s more correction than the 11.8-s noise-based latency, within its ±3-s uncertainty (or the fiber picks the steep P slightly later). It was applied to both cables.</li>
    <li><b>Travel-time errors differ by path:</b> against iasp91 the onsets are 7 s early at Axial (purely oceanic path), 3 s early at Hydrate Ridge, about 1 s early on land, and close to zero on the fiber. The thin oceanic crust explains most of this. An oceanic 1-D model was used for offshore receivers and iasp91 for land.</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Data (origin time free)</th><th class="num">Elements</th><th class="num">Distance from catalogue</th><th class="num">90 % region</th><th>What it constrains</th></tr></thead>
    <tbody>
      <tr><td>DAS, all spans, both cables</td><td class="num">188</td><td class="num">178 km</td><td class="num">64,000 km²</td><td>a wedge pointing at the epicentre: direction</td></tr>
      <tr><td>Offshore seismometers (Axial + Hydrate Ridge)</td><td class="num">13</td><td class="num">52 km</td><td class="num">3,500 km²</td><td>a line (two sites give one arrival-time difference)</td></tr>
      <tr><td>All seismometers (+ land)</td><td class="num">20</td><td class="num">68 km</td><td class="num">340 km²</td><td>a small but displaced patch</td></tr>
      <tr><td>DAS + offshore seismometers</td><td class="num">201</td><td class="num">102 km</td><td class="num">3,600 km²</td><td>a point on the seismometer line inside the DAS wedge</td></tr>
      <tr><td>DAS + all seismometers</td><td class="num">208</td><td class="num">120 km</td><td class="num">1,900 km²</td><td>same</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Result:</b> combining does not yet beat the separate data sets. Without a known origin time, path-dependent travel-time errors of several seconds between groups shift the solution along the seismometer line, and the DAS wedge (emergent P onsets at 0.4–0.9 Hz) is too broad to fix the point on it. With absolute time the multi-span DAS alone does better (13–34 km, section 8g).</li>
    <li><b>What would make it work:</b> station or group delay corrections calibrated on other events with known locations, to absorb the path errors. That needs more events recorded by the DAS in the P band. In Dec 23 – Jan 28 only this M6.0 qualifies, so this waits for the November data.</li>
  </ul>
'''
H += fig("backproject_m6_joint.png", "Joint back-projection maps for DAS only, seismometers only and combinations, origin time free",
         "<b>Joint back-projection, origin time free.</b> DAS spans give a direction wedge; the two offshore seismometer clusters give a line through the epicentre; land stations add a patch; the combinations land on the seismometer line about 100–120 km from the catalogue epicentre (white star). Triangles: Axial and Hydrate Ridge seismometers; squares: land stations; green: fiber bins.")
s = open("report/index.html").read()
anchor = "  <h3>8h · Where this leaves multi-span beamforming</h3>"
assert anchor in s
s = s.replace(anchor, H + anchor)
s = s.replace("s70 (M6.0 back-projection).</li>", "s70 (M6.0 back-projection), s71 (joint DAS + seismometer back-projection).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
