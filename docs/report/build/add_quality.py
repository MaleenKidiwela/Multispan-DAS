S = '''
<section id="quality">
  <div class="rule"></div>
  <h2>Data quality: known issues and how to correct them</h2>
  <p>A checklist for anyone using the OOI multispan DAS records (Nov 5, 2025 – Jan 28, 2026). Each row gives the issue, how to spot it, what it does if ignored, and the correction used here, with the code that implements it. The project copy is <code>docs/DATA_QUALITY.md</code>.</p>
  <div class="callout warn"><p><b>The five that matter most.</b> (1) Dropped data chunks make sample times wrong by up to 16 min: always time samples from the per-chunk metadata. (2) Every acquisition setup has its own time offset (latency) relative to UTC: calibrate it per setup before combining DAS with any other instrument. (3) Masked recordings still contain non-physical channels, including at both fault crossings: test channel quality before use. (4) Acquisition-setup changes can cause steps: never compare or inject across a setup change. (5) The north cable's polarity is opposite to the south cable's: flip it whenever the two are combined.</p></div>

  <h3>Timing</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Issue</th><th>How to spot it</th><th>Effect if ignored</th><th>Correction</th></tr></thead>
    <tbody>
      <tr><td><b>Dropped chunks</b>: whole data chunks (32 samples: 17.18 s on the 100-m stream, 4.29 s on the 8-Hz stream) are missing but the stored array has no gap</td>
          <td>Raw-file trailer: <code>stream.gpu_idx</code> steps by twice the normal amount; <code>stream.proc_start</code> jumps by two chunk lengths. About 450 drops per cable on the 100-m stream, mostly Jan 6 18:00 – 13 and Jan 21 18:00 – 28</td>
          <td>Sample times wrong by up to 958 s by the end of a file; DAS–seismometer correlations collapse; windows splice two records (a fake setup step and a fake correlation with swell period appeared)</td>
          <td>Time every chunk from its own <code>proc_start</code> or the sample counter (<code>msdvv.raw.sample_times</code>, <code>chunk_start_times_counter</code>); put samples on a true-time grid with dropped chunks as flagged zero gaps (<code>load_file(regrid=True)</code>); reject windows more than 10 % empty (<code>xcorr.window_times</code>)</td></tr>
      <tr><td><b>No chunk metadata</b> in the hand-downsampled Nov 6–26 h5 files</td>
          <td>The h5 files carry only <code>data</code> (and no trailer)</td>
          <td>Drops inside a file cannot be found or corrected</td>
          <td>Files are only 17 min long and each has its own start time, so errors cannot accumulate; a 24-h block check against the colocated seismometer HYSB1 found no timing jump (<code>file_timing_qc</code>, <code>MSDVV_QC_BLOCK_H=24</code>). Ask the providers for the original raw files' chunk metadata</td></tr>
      <tr><td><b>Latency</b>: DAS time stamps lag UTC by a setup-dependent amount</td>
          <td>Colocated DAS × HYSB1 correlation peaks away from zero lag; arrivals early or late against catalogue times</td>
          <td>Wrong absolute times; wrong locations (a wrong latency moved the M6.0 back-projection 80–120 km); DAS–seismometer correlations mis-aligned</td>
          <td>Calibrate per setup: +25.2 s (100-m stream, 1.86 Hz), +11.8 s (Jan 13–21, 3.73 Hz), +5.8 s (8-Hz stream, Jan 13; from the simultaneous 100-m stream, <code>s67</code>), +11.27 s (Nov 6–26), +15.57 s (Nov 26 – Dec 17) (<code>s33</code>, colocated HYSB1). Setup differences are whole data units (13.42 s; 4.30 s ≈ one 4.29-s chunk). Absolute values carry about ±3 s; an event-based check (Jan 16 M6.0) put the fiber 3.75 s after HYSB1</td></tr>
      <tr><td><b>Filename times</b> are not the start times</td>
          <td>File names are offset by about one file from <code>proc_start</code></td>
          <td>The wrong file is chosen for an event</td>
          <td>Use <code>stream.proc_start</code> (or the inventory built from it, <code>s00</code>), never the file name</td></tr>
    </tbody>
  </table></div>

  <h3>Channels and geometry</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Issue</th><th>How to spot it</th><th>Effect if ignored</th><th>Correction</th></tr></thead>
    <tbody>
      <tr><td><b>Non-physical channels</b> inside the kept (masked) blocks; only 26–65 % of channels are usable per block (Jan), 40–60 % (Nov)</td>
          <td>Low noise coherence with neighbouring channels (&lt; 0.15 at 0.42 km); f–k energy outside physical velocities; single channels with spikes or tones</td>
          <td>Noise-only pairs dilute correlations and stacks; in back-projection they dominate</td>
          <td>Channel-quality test per block and setup (<code>s05</code>; <code>good</code> column), 1-km bins of good channels only (<code>interspan.make_bins</code>), pairs with a bad channel set to zero</td></tr>
      <tr><td><b>Fault crossings are non-physical</b> (Alvin Canyon route km 274.5, Daisy Bank 295.6)</td>
          <td>Masked from Dec 5; in the unmasked Nov files the channels there fail the coherence test and do not record the Nov 23 M3.2</td>
          <td>Any "fault signal" at the crossings would be noise</td>
          <td>Do not interpret channels at the crossings; study faults with paths that cross them (cross-span, DAS–seismometer), which showed no fault-specific change</td></tr>
      <tr><td><b>Tonal or noisy segments</b> (south C3; near-shore shelf stretches)</td>
          <td>Peaky long-term spectra; incoherent envelopes for earthquakes</td>
          <td>Spurious coherence; unstable dv/v and event picks</td>
          <td>Exclude C3 (<code>exclude_segs</code>); weight or drop incoherent stretches</td></tr>
      <tr><td><b>Block edges, joins and repeaters</b></td>
          <td>Amplitude drops at repeaters; edge effects at mask boundaries</td>
          <td>Wrong geometry and spurious signals near joins</td>
          <td>Drop ±3–6 channels around joins and repeaters (<code>geometry.edge_pad</code>); repeater picks from the amplitude drop (<code>s01</code>)</td></tr>
      <tr><td><b>Polarity</b>: the north cable is sign-inverted relative to the south cable</td>
          <td>Each cable alone gives the right direction; combined coherent beams are 40–66° off</td>
          <td>Combined (two-cable) coherent processing fails</td>
          <td>Multiply north-cable data by −1 whenever the cables are combined coherently (<code>NORTH_SIGN=-1</code> in <code>s69</code>); single-cable products are unaffected. Confirm with the providers</td></tr>
      <tr><td><b>Single-component axial strain</b></td>
          <td>Sensitivity cos²(angle between wave and fiber) for P/Rayleigh; sign flips with orientation for Love/SH</td>
          <td>Broadside waves and steep teleseismic P are barely recorded; Love waves spoil coherent Rayleigh beams</td>
          <td>Use cos² weighting in beams; use Rayleigh windows after the Love waves; use onset or envelope functions when polarity is uncertain</td></tr>
    </tbody>
  </table></div>

  <h3>Acquisition setups, files and coverage</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Issue</th><th>How to spot it</th><th>Effect if ignored</th><th>Correction</th></tr></thead>
    <tbody>
      <tr><td><b>Setup changes</b> (sampling, gauge, mask): Jan 13–21 at 3.73 Hz; Nov 26 (gauge 256 → 128, full fiber); Dec 2 (masked)</td>
          <td>File metadata (<code>D</code>, gauge, dx, mask); the inventory's <code>setup</code> label</td>
          <td>Steps in dv/v, especially DAS–seismometer (cluster HYSB1 → SC4); a biased injection test when the injection date sits at a change (b1 recovery 62–69 % vs 81–83 % inside one setup)</td>
          <td>Decimate to a common rate; label every substack with its setup; test for steps (<code>s48</code>); never place injections or before/after comparisons across a change</td></tr>
      <tr><td><b>Short files</b>: 17 min (Nov 6–26) and 9.6 min (Nov 26 – Dec 2)</td>
          <td>File durations in the inventory</td>
          <td>Several partial substacks per hour (6× storage; filled the disk once); files shorter than the 10-min cross-span window are skipped; per-file timing checks become meaningless</td>
          <td>Merge same-hour substacks (<code>merge_substacks</code>); join consecutive short files onto one true-time grid (<code>d06_merge_short_files.py</code>); compact per-file outputs (<code>s20b</code>); do timing checks on 24-h blocks</td></tr>
      <tr><td><b>Outages and gaps</b>: Nov 12–18, Dec 8–9, Dec 17–23 (power cut; Dec 18–22 missing from the archive); seismometer day files truncated</td>
          <td>Gaps in the inventory; coverage per day</td>
          <td>Excursions at gap edges where stacks contain few hours</td>
          <td>Require ≥ 5/6 of a stack window with data; break plotted series at gaps; read results next to outages with caution; re-request seismometer data in 2-h pieces</td></tr>
      <tr><td><b>Sampling limits</b>: the continuous 100-m stream is 1.86 Hz (Nyquist 0.93 Hz); the 8-Hz stream exists continuously only for Nov (as 7.45 Hz h5) and as Jan snapshots</td>
          <td>—</td>
          <td>Small earthquakes and teleseismic P are not recorded on the 100-m stream; local location needs the 8-Hz stream</td>
          <td>Use the 8-Hz stream for events (download per event from the archive); use 0.04–0.9 Hz for noise, regional and teleseismic work</td></tr>
      <tr><td><b>Unknown fiber coupling and site effects</b></td>
          <td>Strong amplitude variation along the fiber, unrelated to distance</td>
          <td>Amplitude-based magnitudes and energy beams are biased toward well-coupled sites</td>
          <td>Normalise each channel or bin (RAM normalisation, per-bin onset functions); avoid absolute amplitudes without calibration</td></tr>
    </tbody>
  </table></div>

  <h3>External references</h3>
  <ul>
    <li><b>Offshore catalogue locations</b> (PNSN) use onshore stations only (azimuthal gap about 250°); they are not ground truth for validating DAS locations. Onshore events (e.g. the Nov 23 M3.2 Amity) and global locations of large events are.</li>
    <li><b>Seismometer horizontal orientation</b> (HYSB1, HYS14) is unresolved; it does not bias dv/v because each component is measured against its own reference.</li>
  </ul>

  <h3>Recommended processing recipe</h3>
  <ol>
    <li>Build the inventory from <code>proc_start</code>; label each file's setup.</li>
    <li>Time samples per chunk; regrid with flagged gaps; reject windows &gt; 10 % empty.</li>
    <li>Test channel quality per block and setup; use 1-km bins of good channels; exclude C3 and noisy near-shore stretches.</li>
    <li>Normalise with running absolute mean (20 s); whiten in band.</li>
    <li>For absolute time, calibrate the latency of each setup against HYSB1 (or a simultaneous stream with a known latency).</li>
    <li>Use coda-only windows that start after the measured end of the ballistic packet; back-stack 24/48 h with a 1-h step; check with even/odd floors and an injection test inside one setup.</li>
    <li>Flip the north-cable polarity when combining the cables; never compare across setup changes without a step test.</li>
  </ol>

  <h3>Recommendations for the data providers</h3>
  <ul>
    <li>Document dropped chunks and the trailer fields; update the example notebook to time samples from <code>gpu_idx</code> or the per-chunk <code>proc_start</code>.</li>
    <li>Publish the latency of each acquisition setup and the polarity convention of each interrogator.</li>
    <li>Export the chunk metadata for the hand-downsampled Nov 6–26 files, if the raw files still exist.</li>
    <li>Flag non-physical channels (including the fault crossings) in the metadata, and keep file lengths of at least 30 min.</li>
  </ul>
</section>
'''
s = open("report/index.html").read()
anchor = '<section id="limits">'
assert anchor in s and 'id="quality"' not in s
s = s.replace(anchor, S + "\n\n" + anchor, 1)
s = s.replace('    <li><a href="#lessons">Lessons and pitfalls</a></li>', '    <li><a href="#lessons">Lessons and pitfalls</a></li>\n    <li><a href="#quality">Data quality and corrections</a></li>', 1)
open("report/index.html", "w").write(s)
print(len(s) / 1e6)
