from embed_helpers import fig
H = '''
  <h3>7 · November extension (Nov 6 – Dec 17, 2025)</h3>
  <p>The within-span and cross-span products were rerun on Nov 6 – Dec 17 with exactly the final January settings (running-absolute-mean normalisation, true times, gap handling, coda-only windows, 24-h and 48-h back-stacks, 0.5 % injection test). Configuration <code>config_nov_ram.yaml</code>; chain <code>run_nov.sh</code>; comparison <code>s78_nov_compare.py</code>. Outages: Nov 12–18, Dec 8–9, and the Dec 17–23 power cut.</p>
  <ul>
    <li><b>Data:</b> the 8-Hz archive stream, reduced to the 100-m-stream channels (105.9 m, 1.86 Hz). The acquisition setup changed twice: Nov 6–26 downsampled files (gauge 256, no per-chunk times, so dropped chunks cannot be corrected), Nov 26 – Dec 2 full-fiber raw files (gauge 128), and Dec 2–17 masked raw files. November has more usable channels than January: south 1,149 vs 825, north 870 vs 828.</li>
    <li><b>Processing fixes needed for November:</b>
      <ul>
        <li>The Nov 6–26 files are only 17 min long, so each hour was split into several partial substacks (6× the storage; the first attempt filled the disk). Partial substacks are now summed per hour.</li>
        <li>The Nov 26 – Dec 2 files are 9.6 min long, shorter than the 10-min cross-span window, so cross-span initially lost those 6 days. Consecutive files were joined into 38-min pieces on one true-time grid (<code>d06_merge_short_files.py</code>).</li>
        <li>Per-file cross-span outputs were compacted into daily files, which is lossless for the dv/v step (<code>s20b_compact_interspan.py</code>).</li>
      </ul></li>
  </ul>
  <div class="tablewrap"><table>
    <thead><tr><th>Product (48 h)</th><th class="num">Stretching CC<br>Nov / Jan</th><th class="num">Noise floor %<br>Nov / Jan</th><th class="num">Injection recovered<br>Nov / Jan</th></tr></thead>
    <tbody>
      <tr><td>Within-span north, b2</td><td class="num">0.78 / 0.80</td><td class="num">0.012 / 0.011</td><td class="num">97 % / 99 %</td></tr>
      <tr><td>Within-span south, b2</td><td class="num">0.78 / 0.80</td><td class="num">0.010 / 0.011</td><td class="num">99 % / 98 %</td></tr>
      <tr><td>Within-span north, b1</td><td class="num">0.77 / 0.79</td><td class="num">0.008 / 0.015</td><td class="num">69 % / 91 %*</td></tr>
      <tr><td>Within-span south, b1</td><td class="num">0.76 / 0.77</td><td class="num">0.011 / 0.013</td><td class="num">67 % / 91 %*</td></tr>
      <tr><td>Cross-span south, all paths</td><td class="num">0.53 / 0.53</td><td class="num">0.010 / 0.013</td><td class="num">67 % / 81 %*</td></tr>
      <tr><td>Cross-span south, Alvin Canyon / Daisy Bank / control</td><td class="num">0.54 / 0.53 / 0.53</td><td class="num">0.023 / 0.018 / 0.013</td><td class="num">70 / 66 / 68 %*</td></tr>
      <tr><td>Cross-span north, all paths</td><td class="num">0.50 / 0.53</td><td class="num">0.014 / 0.011</td><td class="num">66 % / 81 %*</td></tr>
    </tbody>
  </table></div>
  <ul>
    <li><b>Quality:</b> November matches January. Stretching CC and noise floors are the same within a few hundredths; the 24-h products behave likewise (full table in <code>outputs_nov_ram/nov_vs_jan.csv</code>).</li>
    <li><b>*Lower injection recovery in b1 is a test artefact:</b> the injection date (Nov 27) falls just after the Nov 26 setup change, so the before/after comparison spans two setups. Repeating the within-span injection at Nov 20, inside one setup, gives 81–83 % for b1 (24 h), close to January (86–87 %). Only about 8 hours exist before Nov 20 because of the outage, so this check is indicative. b2 recovers fully either way.</li>
    <li><b>The larger variability after Jan 13 is unusual:</b> the within-span 48-h dv/v standard deviation is 0.018–0.019 % (b2) over Nov 6 – Dec 17, similar to Dec 23 – Jan 13 (0.009–0.012 %) and well below Jan 13–28 (0.027–0.035 %). Six weeks of November show no comparable episode, so Jan 13–28 is a genuinely different period, not the normal behaviour of the cables. Its cause remains open; it is not a noise-source effect (item 8, noise direction) or an ocean driver (item 4).</li>
    <li><b>Fault paths:</b> as in January, cross-span paths across the Alvin Canyon and Daisy Bank faults behave like the control paths.</li>
    <li><b>Still to do:</b> the DAS–seismometer product for November. The seismometer data are downloaded; the DAS latency of the November setups must be calibrated first, because the DAS–seismometer correlation depends on absolute time.</li>
  </ul>
'''
H += fig("nov_vs_jan_dvv.png", "November and January dv/v time series for within-span and cross-span products on both cables",
         "<b>November (blue) and Dec 23 – Jan 28 (black)</b>, same settings, coda only, back-stacked (thin 24 h, thick 48 h). Gaps are outages. Dotted line: Jan 13 18:00. The short excursion right after the Dec 8–9 outage (cross-span south, 24 h) is an edge effect of stacks with few hours of data.")
s = open("report/index.html").read()
anchor = "  <h3>8 · Multi-span beamforming</h3>"
assert anchor in s
s = s.replace(anchor, H + anchor)
s = s.replace("After the main dv/v products, eight follow-up items were run (tracker: <code>docs/NEXT_STEPS.md</code>). Item 2 has its own note (<code>docs/DAS_clock_note.md</code>, summarised in Lessons and pitfalls). Item 7, the November extension, is still in progress.",
              "After the main dv/v products, eight follow-up items were run (tracker: <code>docs/NEXT_STEPS.md</code>). Item 2 has its own note (<code>docs/DAS_clock_note.md</code>, summarised in Lessons and pitfalls). Item 7, the November extension, is done for the DAS–DAS products; the DAS–seismometer product for November is pending.")
s = s.replace("    <li>November data (8-Hz stream", "    <li><s>November data not processed</s>: done for within-span and cross-span (follow-up 7); the November DAS–seismometer product still needs the latency of the November setups.</li>\n    <li>November data (8-Hz stream", 1)
s = s.replace('''    <li><b>Multi-span beamforming</b>''', '''    <li><b>November (Nov 6 – Dec 17) reproduces January's quality</b> (CC 0.78 within-span, 0.53 cross-span at 48 h; same noise floors), and shows no episode like the larger dv/v variability of Jan 13–28, which is therefore unusual (follow-up 7).</li>
    <li><b>Multi-span beamforming</b>''', 1)
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
