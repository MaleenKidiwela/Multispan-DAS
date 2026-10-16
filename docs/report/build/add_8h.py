from embed_helpers import fig
H = '''
  <h3>8h · Locating the Jan 16 M6.0 with all spans (back-projection)</h3>
  <p>Back-projection is beamforming steered to points instead of directions: for every point on a grid (40–47°N, 131–123.5°W, 0.05°, 10 km depth) the P travel time to every 1-km bin is computed (iasp91), each bin's onset function (STA/LTA, 0.4–0.9 Hz, equal weight per bin) is shifted by that time, and the bins are averaged. The location is where the spans line up best. The catalogue epicentre (USGS, 43.69°N 128.06°W, from global and regional networks, error about 5–15 km) is used only for comparison.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Configuration</th><th class="num">Bins</th><th class="num">Origin known (±3 s)<br>distance from catalogue</th><th class="num">90 % region</th><th class="num">Origin free<br>distance from catalogue</th></tr></thead>
    <tbody>
      <tr><td><b>South loop, all spans</b></td><td class="num">99</td><td class="num"><b>13 km</b></td><td class="num">10,500 km²</td><td class="num">198 km</td></tr>
      <tr><td><b>Both cables</b></td><td class="num">188</td><td class="num"><b>25 km</b></td><td class="num">10,300 km²</td><td class="num">188 km</td></tr>
      <tr><td><b>North cable, all spans</b></td><td class="num">89</td><td class="num"><b>34 km</b></td><td class="num">9,600 km²</td><td class="num">230 km</td></tr>
      <tr><td>Single spans (9)</td><td class="num">11–32</td><td class="num">50–310 km (median 141)</td><td class="num">27,000–60,000 km²</td><td class="num">136–481 km</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Origin time free</b> (no outside information): every multi-span array produces a narrow bright ridge from the array toward the true epicentre, so the direction is right, but the distance along the ridge trades off against the origin time, and the maximum sits near the array (origin 26–33 s late). Single spans give bright regions of 130,000–430,000 km² with no usable direction. For a source about two array widths away this trade-off is expected.</li>
    <li><b>Origin time known</b> (from any seismometer or catalogue, with the DAS latency of the file's setup): the arrival time fixes the distance. A single span then gives only an arc of possible positions, because it has no direction (its maximum lands anywhere on the arc, 50–310 km off). The multi-span arrays collapse the arc to one patch around the epicentre: <b>south loop 13 km, both cables 25 km, north cable 34 km from the catalogue position</b>.</li>
    <li><b>The latency matters:</b> Jan 16 falls in the Jan 13–21 period recorded at 3.73 Hz (latency 11.8 s). A first run with the 1.86-Hz latency (25.2 s) put every solution 80–120 km too close to the array. Checked independently: with 11.8 s the observed P onsets fall about 1 s after the iasp91 prediction, as expected for emergent onsets.</li>
    <li><b>Limits:</b> one event; iasp91 has continental crust whereas the path is oceanic; the 90 % regions (about 100 × 100 km) are large because the usable band is only 0.4–0.9 Hz and the event is 280 km away. Scripts: <code>s70_backproject_m6.py</code> (<code>ORIGIN=free|known</code>).</li>
  </ul>
'''
H += fig("backproject_m6_origin_known.png", "Back-projection location maps for single spans and multi-span arrays with known origin time",
         "<b>Jan 16 M6.0 located by back-projection, origin time known (±3 s).</b> Single spans (south A1, north A2) give an arc of possible positions; the south loop, the north cable and both cables give a patch at the epicentre. White star: catalogue; cyan cross: back-projection maximum; cyan contour: 90 % of the maximum; green: bins used.")
H += fig("backproject_m6.png", "Back-projection maps with origin time free",
         "<b>Same, origin time free</b> (no outside information): multi-span arrays give a narrow ridge pointing at the epicentre (direction), single spans do not; the distance along the ridge is not resolved without absolute time.")
s = open("report/index.html").read()
anchor = "  <h3>8g · Where this leaves multi-span beamforming</h3>"
assert anchor in s
s = s.replace(anchor, H + anchor)
s = s.replace("<li><b>Location:</b> not yet shown. The only local event with 8-Hz data, an ML 2.0, is recorded coherently on just two far stretches.</li>",
              "<li><b>Location:</b> shown for the regional M6.0 once absolute time is used (8h): the south loop puts it 13 km from the catalogue epicentre, single spans 50–310 km. For the small local ML 2.0 (8f) only two far stretches record a coherent wavefront, so its location is not robust.</li>")
s = s.replace("Locating a small local earthquake was not yet robust (follow-up 8).",
              "With absolute timing, back-projection over all spans located the M6.0 within 13–34 km of the catalogue epicentre (single spans 50–310 km). Locating a small local earthquake was not yet robust (follow-up 8).")
s = s.replace("s65–s65h (Jan 13 location).</li>", "s65–s65h (Jan 13 location), s70 (M6.0 back-projection).</li>")
# move section 8g heading text to keep order: 8h inserted before 8g -> rename to keep sequence
s = s.replace("<h3>8h · Locating the Jan 16 M6.0", "<h3>8g · Locating the Jan 16 M6.0").replace("<h3>8g · Where this leaves", "<h3>8h · Where this leaves")
s = s.replace("(8h): the south loop puts it", "(8g): the south loop puts it")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
