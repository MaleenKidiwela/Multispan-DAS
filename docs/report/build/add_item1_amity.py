from embed_helpers import fig
H = '''
  <p><b>Earthquake test (Nov 23 2025 M3.2 Amity, Oregon; 8-Hz stream, full fiber):</b> could the waves of a real earthquake show the faults where the south cable crosses them? A fault zone would appear as a step in arrival time, amplification and longer shaking over the damage zone, a change in apparent velocity along the fiber, or scattered waves starting at the crossing.</p>
  <ul>
    <li><b>Result:</b> the fiber from about route km 272 to 300, which contains both crossings, does not record the earthquake. It is clearly seen at km 252–262, 265–270 and 305–320.</li>
    <li><b>Why:</b> the channels at the crossings are non-physical. Pre-event noise coherence between neighbours is 0.09 at Alvin Canyon (0.29 on control stretches) and about 0 at Daisy Bank, against a threshold of 0.15. Arrival-time, amplitude and slowness profiles there are therefore noise.</li>
    <li><b>Conclusion:</b> after noise coherence, f–k and now an earthquake, the crossing channels are consistently non-physical; the absence of a fault signal reflects the fiber there, not the faults. The remaining chance is the Nov 27 – Dec 2 52.9-m cut-outs (borderline f–k at Alvin Canyon) with the Nov 27 M6.4 Indonesia teleseism, whose long wavelengths could at best show whether those channels record anything. Script: <code>s75_amity_fault_crossings.py</code>.</li>
  </ul>
'''
H += fig("amity_fault_crossings.png", "Envelope record section and amplitude, coherence and onset profiles along the south cable across the two fault crossings",
         "<b>Nov 23 M3.2 across the fault crossings.</b> Left: log envelope along the south cable (route km 240–330) with the Alvin Canyon and Daisy Bank crossings (cyan); the earthquake (bright, 50–100 s) appears only away from the crossings. Middle: peak amplitude, SNR and noise coherence. Right: onset, local slowness and duration (noise where the channels are non-physical).")
s = open("report/index.html").read()
anchor = '''    <li><b>Final test:</b> the full Nov 26 – Dec 2 record at 52.9 m, cut out around both crossings, is being saved during the November download.</li>
  </ul>'''
assert anchor in s
s = s.replace(anchor, anchor.replace("is being saved during the November download.", "was saved during the November download (857 files).") + H)
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
