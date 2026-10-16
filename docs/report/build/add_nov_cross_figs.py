from embed_helpers import fig
import shutil
F = "/data/wsd02/multispan_dvv/outputs_100m/figs/"
shutil.copy("/data/wsd02/multispan_dvv/outputs_nov_ram/interspan_coda/ram_b1_48h/windows_check_b1_48h.png", F + "nov_crossspan_ccf_south_48h.png")
shutil.copy("/data/wsd02/multispan_dvv/outputs_nov_ram/interspan_coda/north_b1_48h/windows_check_b1_48h.png", F + "nov_crossspan_ccf_north_48h.png")
H = '''
  <p><b>Span-to-span (cross-span), November in detail</b> (<code>s43</code> windows check, <code>s80_nov_crossspan_figs.py</code>).</p>
  <ul>
    <li><b>Correlation functions:</b> 188 converging pairs across the Alvin Canyon fault, 141 across Daisy Bank, 12 at Southern Hydrate Ridge and 461 controls (south); 637 on the north cable. The coda windows start after the end of the ballistic packet of every segment pair, as in January, and the best pair of each group is stable day to day.</li>
    <li><b>dv/v per group:</b> all groups behave alike (48 h: CC 0.49–0.54, floors 0.010–0.023 % except SHR 0.063 % with only 12 pairs), as in January. No fault-specific change. Short excursions follow the Nov 12–18 and Dec 8–9 outages, where the stacks have few hours of data.</li>
  </ul>
'''
H += fig("nov_crossspan_ccf_south_48h.png", "November cross-span correlation gathers with ballistic ends and coda windows, south cable groups, and best-pair daily stacks",
         "<b>November span-to-span correlation functions, south cable</b> (b1, 48 h): gathers of the converging pairs per group sorted by distance, with the end of the ballistic packet (black), coda start (green) and coda end (magenta); below, the best pair of each group: reference and daily 48-h stacks with the coda windows (green).", w=1600)
H += fig("nov_crossspan_ccf_north_48h.png", "November cross-span correlation gathers, north cable",
         "<b>November span-to-span correlation functions, north cable</b> (b1, 48 h), same layout.", w=1400)
H += fig("nov_crossspan_dvv.png", "Cross-span dv/v per path group, November and January",
         "<b>Span-to-span dv/v per path group</b>, November (left) and January (right), same settings: thin 24 h, thick 48 h, with each group's stretching CC, noise floor and number of pairs.", w=1500)
H += fig("nov_crossspan_pairs_map.png", "Map of the November cross-span pairs used per group on both cables",
         "<b>November span-to-span pairs with a dv/v measurement</b> (48 h), coloured by group: control (blue), Alvin Canyon fault (red), Daisy Bank fault (orange), Southern Hydrate Ridge (purple); north cable right.", w=1500)
s = open("report/index.html").read()
anchor = "  <p><b>Third product: span-to-seismometer"
assert anchor in s
s = s.replace(anchor, H + anchor, 1)
s = s.replace("s78 / s79 (Nov vs Jan).</li>", "s78 / s79 / s80 (Nov vs Jan, cross-span figures and pair maps).</li>")
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
