from embed_helpers import fig
H = '''
  <p><b>Direction of small earthquakes</b> (<code>s77_small_event_direction.py</code>). Using only the S-wavefront delays inside each coherent stretch (a free time offset per stretch, so no latency or origin time), a plane-wave fit gives the arrival direction. Ground truth: the Amity M3.2, true back-azimuth 88° from the array.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Configuration</th><th class="num">Estimated</th><th class="num">Error</th><th class="num">Range within 10 % of best</th></tr></thead>
    <tbody>
      <tr><td><b>All 6 coherent stretches (north A2, south A2, A 145–150, B3, C4, D4)</b></td><td class="num"><b>64°</b></td><td class="num"><b>−23°</b></td><td class="num">65°</td></tr>
      <tr><td>North A2 alone</td><td class="num">33°</td><td class="num">−65°</td><td class="num">slowness at search limit</td></tr>
      <tr><td>South A2 alone</td><td class="num">135°</td><td class="num">+44°</td><td class="num">82°</td></tr>
      <tr><td>South B3 / C4 alone (8 / 4 bins)</td><td class="num">89° / 74°</td><td class="num">+20° / +14°</td><td class="num">~19°</td></tr>
      <tr><td>South A 145–150 / D4 alone (4 / 3 bins)</td><td class="num">122° / 343°</td><td class="num">+38° / −89°</td><td class="num">degenerate</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Result:</b> the spans together give the right general direction for a small earthquake (east-northeast vs true east, about 25° off, uncertainty about ±30°). Single stretches give errors of 14–89° that vary from stretch to stretch, and without the others there is no way to tell which are right. For the Jan 13 ML 2.0 the two coherent stretches together give 73° (east-northeast), consistent with the source lying east of about 124.75°W.</li>
    <li><b>Limit:</b> envelope delays blur the wavefront; coherent waveform beams on 2–5 km pieces of each stretch would be the way to sharpen it.</li>
  </ul>
'''
H += fig("small_event_direction.png", "Misfit versus back-azimuth for all stretches and single stretches, Amity and Jan 13",
         "<b>Direction of small earthquakes from the S wavefront in each stretch.</b> Misfit vs back-azimuth for all coherent stretches (black) and single stretches (colours). Left: Amity M3.2 with the true back-azimuth (red dashed). Right: Jan 13 ML 2.0.")
s = open("report/index.html").read()
a = "  <h3>8g · Locating the Jan 16 M6.0"
assert a in s
s = s.replace(a, H + a)
s = s.replace("s76 (noise direction vs dv/v).</li>", "s76 (noise direction vs dv/v), s77 (small-event direction).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
