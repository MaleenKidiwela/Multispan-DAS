from embed_helpers import fig
H = '''
  <p><b>Third product: span-to-seismometer (HYSB1 and HYS14 × south cable), November.</b> Same pipeline as January (<code>config_nov_obsdas.yaml</code>, chains <code>run_nov_obsdas*.sh</code>, comparison <code>s79_nov_obsdas_compare.py</code>), with three November-specific steps:</p>
  <ul>
    <li><b>DAS latency per setup</b> (colocated HYSB1 paths, 0.12–0.3 Hz): +11.27 s for Nov 6–26 (downsampled, gauge 256) and +15.57 s for Nov 26 – Dec 17 (full fiber, gauge 128). They differ by 4.30 s, one 4.29-s data chunk at this sampling, the same kind of whole-unit step as January's 13.42 s.</li>
    <li><b>Timing check on 24-h blocks:</b> the per-file check rejected every November file, because a single 17-min file is too short for a stable colocated correlation. Scored in 24-h blocks, 32 of 38 days pass (median CC 0.69) and <b>no block shows a timing jump</b> (no shift above 1 s). The 6 weak days are left out under January's rule.</li>
    <li><b>Injection on Dec 12</b>, so that both comparison windows lie inside one setup.</li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Cluster (48 h)</th><th class="num">Stretching CC<br>Nov / Jan</th><th class="num">Noise floor %<br>Nov / Jan</th><th class="num">Injection recovered<br>Nov / Jan</th></tr></thead>
    <tbody>
      <tr><td>HYS14 × Alvin Canyon fault → SB3</td><td class="num">0.63 / 0.68</td><td class="num">0.066 / 0.060</td><td class="num">93 % / 94 %</td></tr>
      <tr><td>HYS14 × Alvin Canyon fault → SC4 west</td><td class="num">0.58 / 0.66</td><td class="num">0.082 / 0.053</td><td class="num">49 % / 94 %</td></tr>
      <tr><td>HYS14 × Alvin Canyon fault → SC4 east</td><td class="num">0.54 / 0.61</td><td class="num">0.096 / 0.065</td><td class="num">66 % / 95 %</td></tr>
      <tr><td>HYSB1 control → SB3</td><td class="num">0.76 / 0.78</td><td class="num">0.090 / 0.077</td><td class="num">82 % / 94 %</td></tr>
      <tr><td>HYSB1 control → SC4</td><td class="num">0.48 / 0.56</td><td class="num">0.098 / 0.080</td><td class="num">56 % / 84 %</td></tr>
      <tr><td>HYS14 × Daisy Bank fault</td><td class="num">0.42 / 0.50</td><td class="num">0.082 / 0.108</td><td class="num">68 % / 61 %</td></tr>
      <tr><td><b>Median</b></td><td class="num"><b>0.56 / 0.64</b></td><td class="num"><b>0.086 / 0.071</b></td><td class="num"><b>67 % / 94 %</b></td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Result:</b> the span-to-seismometer product works in November but is weaker than in January. The best cluster (HYSB1 → SB3) holds up; the far clusters lose most.</li>
    <li><b>Likely reasons:</b>
      <ul>
        <li>The 17-min files fit only one 10-min window each, so each hour has fewer windows.</li>
        <li>Six days are excluded by the timing check.</li>
        <li>The Dec 12 injection's "before" window contains the Dec 8–9 outage, which makes recovery noisier.</li>
        <li>Visible jumps at outage edges and around the Nov 26 setup change: as in January, this is the product most sensitive to setup changes.</li>
      </ul></li>
    <li><b>Use:</b> treat November span-to-seismometer dv/v near Nov 26 and next to outages with caution. No fault-specific signal: fault-crossing clusters vary like the controls.</li>
  </ul>
'''
H += fig("nov_vs_jan_obsdas.png", "Seismometer-DAS dv/v per path cluster, November and January",
         "<b>Span-to-seismometer dv/v per path cluster</b> (48 h, back-stacked): November (blue) and Dec 23 – Jan 28 (black), with each period's stretching CC and noise floor.", w=1300)
s = open("report/index.html").read()
anchor = "    <li><b>Still to do:</b> the DAS–seismometer product for November. The seismometer data are downloaded; the DAS latency of the November setups must be calibrated first, because the DAS–seismometer correlation depends on absolute time.</li>\n  </ul>"
assert anchor in s
s = s.replace(anchor, "  </ul>")
a2 = "  <h3>8 · Multi-span beamforming</h3>"
# insert the third product right after the November figure (before section 8)
s = s.replace(a2, H + a2, 1)
s = s.replace("Item 7, the November extension, is done for the DAS–DAS products; the DAS–seismometer product for November is pending.",
              "Item 7, the November extension, is done for all three products.")
s = s.replace("    <li>November: the within-span and cross-span products are done (follow-up 7). The November DAS–seismometer product still needs the DAS latency of the November setups. The Nov 6–26 files carry no per-chunk times, so dropped chunks there cannot be corrected.</li>",
              "    <li>November: all three products are done (follow-up 7). The span-to-seismometer product is weaker than in January (median CC 0.56 vs 0.64) and jumps at outage edges and the Nov 26 setup change. The Nov 6–26 files carry no per-chunk times, so dropped chunks there cannot be corrected (the 24-h timing check found no jumps).</li>")
s = s.replace("<li><b>November (Nov 6 – Dec 17) reproduces January's quality</b> (CC 0.78 within-span, 0.53 cross-span at 48 h; same noise floors), and shows no episode",
              "<li><b>November (Nov 6 – Dec 17) reproduces January's quality</b> for within-span and cross-span (CC 0.78 and 0.53 at 48 h; same noise floors; span-to-seismometer somewhat weaker, CC 0.56 vs 0.64), and shows no episode")
s = s.replace("s77 (small-event direction).</li>", "s77 (small-event direction); November: config_nov_ram.yaml, config_nov_obsdas.yaml, run_nov*.sh, d06 (joining short files), s20b (compaction), s78 / s79 (Nov vs Jan).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6, s.count("Third product: span-to-seismometer"))
