import re
from embed_helpers import fig
s = open("report/index.html").read()
# 1) November section: replace the side-by-side figure with the November-only figure in the common layout
m = re.search(r'  <figure><img src="data:image/jpeg;base64,[^"]+" alt="Cross-span dv/v per path group, November and January"><figcaption>.*?</figcaption></figure>\n', s, re.S)
assert m, "side-by-side figure not found"
nov = fig("nov_crossspan_dvv_groups.png", "Cross-span dv/v per path group, November",
          "<b>Span-to-span dv/v per path group, November</b> (thin 24 h, thick 48 h; labels: stretching CC, noise floor, number of pairs). The January version in the same layout is in the Cross-span section.", w=1150)
s = s.replace(m.group(0), nov)
# 2) January cross-span section: add the January figures in the same layout, at the end of the section
sec = s.find('<section id="cross">'); end = s.find("</section>", sec)
jan = '''
  <h3>Same layout as the November figures</h3>
  <p>For direct comparison with November (follow-up 7), the January span-to-span dv/v per path group and the map of the pairs used, drawn exactly like the November figures (<code>s80_nov_crossspan_figs.py</code>).</p>
'''
jan += fig("jan_crossspan_dvv_groups.png", "Cross-span dv/v per path group, January, November layout",
           "<b>Span-to-span dv/v per path group, Dec 23 – Jan 28</b> (thin 24 h, thick 48 h; labels: stretching CC, noise floor, number of pairs), in the same layout as the November figure.", w=1150)
jan += fig("jan_crossspan_pairs_map.png", "Map of the January cross-span pairs used per group on both cables",
           "<b>January span-to-span pairs with a dv/v measurement</b> (48 h), coloured by group: control (blue), Alvin Canyon fault (red), Daisy Bank fault (orange), Southern Hydrate Ridge (purple); north cable right.", w=1300)
s = s[:end] + jan + s[end:]
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
