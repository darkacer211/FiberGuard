/**
 * FiberGuard Frontend - Client Physics & Fallback Machine Learning
 * Mirrors backend physical optics equations and k-NN inference for offline resilience.
 */

const ClientPhysics = (() => {
  const FAULT_NAMES = [
    'Normal Link', 'Fiber Cut', 'Macro-Bend', 'Dirty Connector',
    'Aged Fiber', 'EDFA Failure', 'Chromatic Dispersion'
  ];
  const FAULT_SHORT = ['Normal', 'Cut', 'Bend', 'Dirty', 'Aged', 'EDFA', 'Disp'];
  const FAULT_COLORS = ['#3ddc97', '#ff6b6b', '#ffc857', '#ff9a5c', '#b69cff', '#5ee3ff', '#ff7ccf'];

  const D_DISP = 17.0, DLAM0 = 0.02, ALPHA0 = 0.20, FRONT = 0.5, ENDC = 0.5, SPL = 0.1, G0 = 15.0, NF0 = 5.0, FLOOR = -50.0;

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function randn(r) {
    let u = 0; while (u === 0) u = r();
    const v = r();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  function erfc(x) {
    const z = Math.abs(x), t = 1.0 / (1.0 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2.0 - r;
  }

  function erf(x) { return 1.0 - erfc(x); }

  function gPulse(u, s) {
    if (s <= 0.001) return Math.abs(u) <= 0.5 ? 1.0 : 0.0;
    const d = Math.SQRT2 * s;
    return 0.5 * (erf((u + 0.5) / d) - erf((u - 0.5) / d));
  }

  function eyeOpen(s) {
    let e = gPulse(0.0, s);
    for (let k = -3; k <= 3; k++) if (k !== 0) e -= gPulse(k, s);
    return Math.max(e, 0.0);
  }

  function simulate(p) {
    const seed = p.seed || 42;
    const rng = mulberry32(seed);

    const L = p.length_km || 80.0;
    const Ptx = p.launch_power_dbm != null ? p.launch_power_dbm : 0.0;
    const rate = p.bit_rate_gbps || 10.0;
    const f = p.fault_type || 0;
    const sev = p.severity != null ? p.severity : 0.5;
    const pos = p.fault_pos_ratio != null ? p.fault_pos_ratio : 0.45;

    const Tps = 1000.0 / rate;
    const xf = pos * L;
    const nSpl = Math.floor(L / 20);
    const splLoss = nSpl * SPL;

    let alpha = ALPHA0, extra = 0.0, spike = 0.0, gain = G0, nf = NF0;
    let ratio = (D_DISP * L * DLAM0) / Tps;
    const isCut = (f === 1);

    if (f === 4) alpha = 0.25 + 0.15 * sev;
    else if (f === 2) extra = 0.8 + 3.8 * sev;
    else if (f === 3) { extra = 0.7 + 2.5 * sev; spike = 3.0 + 6.5 * sev; }
    else if (f === 5) { gain = G0 - (10.0 + 13.0 * sev); nf = 10.0 + 7.0 * sev; }
    else if (f === 6) ratio = 1.0 + 1.5 * sev;

    const fiberLoss = alpha * L;
    const pin = isCut ? -65.0 : Ptx - FRONT - fiberLoss - splLoss - extra - ENDC;
    const prx = isCut ? -65.0 : pin + gain;
    const osnr = isCut ? -5.0 : 58.0 + pin - nf;

    const pinExp = Ptx - FRONT - (ALPHA0 * L) - splLoss - ENDC;
    const prxExp = pinExp + G0;
    const osnrExp = 58.0 + pinExp - NF0;
    const ratioExp = (D_DISP * L * DLAM0) / Tps;

    const sens = rate > 5.0 ? -18.0 : -24.0;
    const sig = Math.sqrt(0.12 * 0.12 + Math.pow(ratio / 2.355, 2));
    const eo = eyeOpen(sig);

    const be = 0.75 * rate;
    const q0o = Math.sqrt(Math.max(0.001, Math.pow(10.0, osnr / 10.0) * 12.5 / (2.0 * be)));
    const qth = 6.0 * Math.pow(10.0, (prx - sens) / 10.0);
    const q0 = 1.0 / Math.sqrt(Math.max(1e-6, 1.0 / (q0o * q0o + 1e-6) + 1.0 / (qth * qth + 1e-6)));
    const qEff = q0 * Math.max(eo, 0.001);
    const ber = isCut ? 0.5 : 0.5 * erfc(qEff / Math.SQRT2);

    const mPrx = prx + 0.28 * randn(rng);
    const mOsnr = osnr + 0.45 * randn(rng);
    const mRatio = Math.max(0.01, ratio * (1.0 + 0.07 * randn(rng)));

    // Generate OTDR
    const N = 512, Lmax = L * 1.15, dx = Lmax / (N - 1);
    const endX = isCut ? xf : L;
    const tr = new Float32Array(N);

    const events = [{ x: 0, loss: 0, h: 10 }];
    for (let k = 1; k <= nSpl; k++) events.push({ x: k * L / (nSpl + 1), loss: SPL, h: 0 });
    if (f === 2) events.push({ x: xf, loss: extra, h: 0 });
    if (f === 3) events.push({ x: xf, loss: extra, h: spike });
    if (isCut) events.push({ x: xf, loss: 50, h: 9.5 });
    else events.push({ x: L, loss: 0, h: 6 });

    for (let i = 0; i < N; i++) {
      const x = i * dx;
      let v;
      if (x > endX + dx * 0.5) {
        v = FLOOR + 0.75 * randn(rng);
      } else {
        let lvl = -alpha * x;
        for (const ev of events) if (ev.loss > 0 && ev.x < x) lvl -= ev.loss;
        lvl = Math.max(lvl, FLOOR);
        v = lvl + (0.03 + 0.0035 * Math.abs(lvl)) * randn(rng);
      }
      for (const ev of events) {
        if (ev.h > 0) {
          const d = (x - ev.x) / (1.15 * dx);
          v += ev.h * Math.exp(-d * d);
        }
      }
      tr[i] = v;
    }

    // Analyze OTDR
    const sm = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0, c = 0;
      for (let j = i - 2; j <= i + 2; j++) if (j >= 0 && j < N) { s += tr[j]; c++; }
      sm[i] = s / c;
    }
    let endIdx = N - 1;
    while (endIdx > 5 && sm[endIdx] < FLOOR + 4.5) endIdx--;
    const endDist = Math.max(0, (endIdx - 2) * dx);
    const endRatio = Math.min(1.2, endDist / L);

    const a = 4, b = Math.max(a + 15, endIdx - 6), K = 6, seg = Math.max(4, Math.floor((b - a) / K));
    const slopes = [];
    for (let k = 0; k < K; k++) {
      const s = a + k * seg, e = Math.min(s + seg, N);
      let sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
      for (let i = s; i < e; i++) {
        const x = i * dx; sx += x; sy += tr[i]; sxx += x * x; sxy += x * tr[i]; n++;
      }
      const den = n * sxx - sx * sx;
      if (n > 4 && den > 0) slopes.push(-(n * sxy - sx * sy) / den);
    }
    slopes.sort((p, q) => p - q);
    const alphaAn = slopes.length ? slopes[Math.floor(slopes.length / 2)] : ALPHA0;

    const w = 8, gap = 3, lo = a + w + gap + 1, hi = Math.max(lo + 5, endIdx - 16);
    let best = 0, bi = -1;
    const stArr = new Float32Array(N).fill(-10);
    for (let i = lo; i < hi; i++) {
      let ml = 0, mr = 0;
      for (let j = 0; j < w; j++) { ml += tr[i - gap - 1 - j]; mr += tr[i + gap + 1 + j]; }
      ml /= w; mr /= w;
      const step = (ml - mr) - alphaAn * (2 * gap + 1 + w) * dx;
      stArr[i] = step;
      if (step > best) { best = step; bi = i; }
    }
    let posKm = 0, stepDb = 0, spikeDb = 0;
    if (bi > 0 && best > 0.4) {
      let l = bi, r = bi;
      while (l - 1 >= lo && stArr[l - 1] >= 0.85 * best) l--;
      while (r + 1 < hi && stArr[r + 1] >= 0.85 * best) r++;
      const c = Math.round((l + r) / 2);
      posKm = c * dx;
      stepDb = best;
      let ml = 0; for (let j = 0; j < w; j++) ml += tr[c - gap - 1 - j]; ml /= w;
      let mx = -1e9; for (let j = c - gap; j <= c + gap; j++) mx = Math.max(mx, tr[j]);
      spikeDb = Math.max(0, mx - ml);
    }

    const feats = [
      clamp(mPrx - prxExp, -45, 10) / 4.0,
      clamp(mOsnr - osnrExp, -45, 10) / 5.0,
      (alphaAn - ALPHA0) / 0.05,
      stepDb / 1.0,
      spikeDb / 3.0,
      clamp((1.0 - endRatio) / 0.05, 0, 20),
      Math.log10(Math.max(1e-3, mRatio) / Math.max(1e-3, ratioExp)) / 0.3
    ];

    const pMargin = prx - sens;
    const budgetItems = [
      { item: "Launch Power (P_tx)", value: `${Ptx >= 0 ? '+' : ''}${Ptx.toFixed(1)} dBm`, note: "Transmitter source", status: "neutral" },
      { item: "Tx Patch Connector", value: `-${FRONT.toFixed(1)} dB`, note: "Front panel insertion loss", status: "neutral" },
      { item: "Fiber Attenuation Loss", value: `-${fiberLoss.toFixed(1)} dB`, note: `${alpha.toFixed(2)} dB/km × ${L.toFixed(0)} km`, status: alpha <= 0.22 ? "neutral" : "bad" },
      { item: "Fusion Splice Losses", value: `-${splLoss.toFixed(1)} dB`, note: `${nSpl} splices @ 0.1 dB`, status: "neutral" },
      { item: "Rx Patch Connector", value: `-${ENDC.toFixed(1)} dB`, note: "Rear panel insertion loss", status: "neutral" },
      { item: "Fault Excess Loss", value: isCut ? "-∞ dB" : `-${extra.toFixed(1)} dB`, note: isCut ? "Fiber broken" : (extra > 0 ? "Bend / dirty connector" : "None"), status: extra > 0 || isCut ? "bad" : "good" },
      { item: "EDFA Input Power (P_in)", value: isCut ? "None" : `${pin.toFixed(1)} dBm`, note: "Subtotal before amplification", status: "sum" },
      { item: "EDFA Booster Gain", value: `${gain >= 0 ? '+' : ''}${gain.toFixed(1)} dB`, note: f === 5 ? "Degraded pump" : "Nominal gain (+15 dB)", status: f === 5 ? "bad" : "neutral" },
      { item: "Receiver Power (P_rx)", value: isCut ? "LOS" : `${prx.toFixed(1)} dBm`, note: "Total incident optical power", status: "sum" },
      { item: "Receiver Sensitivity", value: `${sens.toFixed(1)} dBm`, note: `PIN photodiode for ${rate} Gb/s`, status: "neutral" },
      { item: "Link Power Margin", value: isCut ? "Loss of Light" : `${pMargin >= 0 ? '+' : ''}${pMargin.toFixed(1)} dB`, note: "P_rx - P_sens", status: pMargin >= 3 ? "good" : (pMargin >= 0 ? "warn" : "bad") }
    ];

    return {
      params: { length_km: L, launch_power_dbm: Ptx, bit_rate_gbps: rate, fault_type: f, fault_name: FAULT_NAMES[f], fault_pos_ratio: pos, fault_km: xf, severity: sev },
      telemetry: {
        prx_dbm: prx, measured_prx_dbm: mPrx, osnr_db: osnr, measured_osnr_db: mOsnr,
        q_effective: qEff, q_db: qEff > 0.1 ? 20 * Math.log10(qEff) : null,
        ber: ber, eye_opening_pct: eo * 100, pulse_spread_ps: mRatio * Tps, bit_period_ps: Tps,
        margin_db: pMargin, attenuation_alpha: alphaAn, is_cut: isCut
      },
      otdr: { trace: Array.from(tr), distances: Array.from({ length: N }, (_, i) => i * dx), dx, l_max: Lmax, end_km: endX },
      otdr_analysis: {
        alpha_measured: alphaAn, detected_step_db: stepDb, event_pos_km: posKm,
        reflection_spike_db: spikeDb, end_distance_km: endDist, end_ratio: endRatio
      },
      features: feats,
      power_budget: budgetItems
    };
  }

  // Pre-computed representative training samples for client-side k-NN fallback
  let KNN_TRAIN = null;
  function getKnnData() {
    if (KNN_TRAIN) return KNN_TRAIN;
    const rng = mulberry32(12345);
    const X = [], y = [];
    for (let i = 0; i < 1400; i++) {
      const f = i % 7;
      const sim = simulate({
        length_km: 20 + 80 * rng(),
        launch_power_dbm: -3 + 8 * rng(),
        bit_rate_gbps: rng() > 0.4 ? 10 : 2.5,
        fault_type: f,
        fault_pos_ratio: 0.1 + 0.8 * rng(),
        severity: rng(),
        seed: i + 10
      });
      X.push(sim.features);
      y.push(f);
    }
    KNN_TRAIN = { X, y };
    return KNN_TRAIN;
  }

  function predict(payload) {
    const knn = getKnnData();
    let feat = payload.features;
    let groundTruth = null;
    let groundTruthKm = null;
    let locKm = null;

    if (!feat && payload.simulation_params) {
      const sim = simulate(payload.simulation_params);
      feat = sim.features;
      groundTruth = sim.params.fault_type;
      groundTruthKm = sim.params.fault_km;
      if (groundTruth === 1) locKm = sim.otdr_analysis.end_distance_km;
      else if (groundTruth === 2 || groundTruth === 3) locKm = sim.otdr_analysis.event_pos_km;
    }

    const n = knn.X.length, k = 9;
    const dists = new Float32Array(n), idxs = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0; const t = knn.X[i];
      for (let j = 0; j < feat.length; j++) { const d = feat[j] - t[j]; sum += d * d; }
      dists[i] = Math.sqrt(sum); idxs[i] = i;
    }
    idxs.sort((a, b) => dists[a] - dists[b]);

    const probs = new Array(7).fill(0);
    let totalWeight = 0;
    for (let i = 0; i < k; i++) {
      const w = 1.0 / (dists[idxs[i]] + 0.05);
      probs[knn.y[idxs[i]]] += w;
      totalWeight += w;
    }
    for (let i = 0; i < 7; i++) probs[i] /= totalWeight;

    let bestClass = 0;
    for (let i = 1; i < 7; i++) if (probs[i] > probs[bestClass]) bestClass = i;

    const DESCS = [
      { signature: "All monitored optical metrics reside well within design budget tolerances.", cause: "Normal link operation, clean connectors, nominal EDFA.", action: "Maintain routine telemetry logging.", severity_label: "Severity" },
      { signature: "Severe Fresnel reflection spike followed by immediate collapse to -50 dB noise floor.", cause: "Physical fiber break / excavator severed cable span.", action: "Dispatch restoration crew to located distance.", severity_label: "Severity" },
      { signature: "Localized step loss with no reflection spike; parallel slope on both sides.", cause: "Macro-bend tighter than critical radius in tray or duct.", action: "Inspect identified enclosure and re-route.", severity_label: "Bend Tightness" },
      { signature: "Combined step loss coupled with high Fresnel reflection spike.", cause: "Contaminated or poorly mated optical connector end-face.", action: "Clean ferrule with isopropyl alcohol wipes and re-seat.", severity_label: "Contamination" },
      { signature: "Evenly elevated attenuation slope (> 0.25 dB/km) over entire cable length.", cause: "Distributed cable aging, hydrogen darkening, or water ingress.", action: "Schedule cable segment replacement or re-budget EDFA.", severity_label: "Aging" },
      { signature: "OTDR appears clean, but optical receiver power and OSNR drop drastically.", cause: "EDFA optical amplifier pump laser degradation.", action: "Inspect pump current and replace degraded EDFA module.", severity_label: "Pump Degradation" },
      { signature: "OTDR slope and optical power are nominal, yet eye closes due to pulse spreading.", cause: "Excessive chromatic dispersion or uncompensated laser chirp.", action: "Install DCF module or enable electronic dispersion compensation.", severity_label: "Source Linewidth" }
    ];

    const ranking = Array.from({ length: 7 }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    const ranked = ranking.map(i => ({
      class_index: i,
      name: FAULT_NAMES[i],
      short_name: FAULT_SHORT[i],
      color: FAULT_COLORS[i],
      probability: Math.round(probs[i] * 1000) / 1000
    }));

    const dp = feat[0] * 4.0;
    return {
      predicted_class: bestClass,
      class_name: FAULT_NAMES[bestClass],
      short_name: FAULT_SHORT[bestClass],
      color: FAULT_COLORS[bestClass],
      confidence: Math.round(probs[bestClass] * 100) / 100,
      probabilities: probs.map(p => Math.round(p * 1000) / 1000),
      ranked_predictions: ranked,
      model_used: payload.model_name || 'Client-side k-NN (9)',
      localized_km: locKm,
      ground_truth_km: groundTruthKm,
      ground_truth_type: groundTruth,
      localization_error_km: locKm != null && groundTruthKm != null && bestClass === groundTruth ? Math.abs(locKm - groundTruthKm).toFixed(2) : null,
      details: DESCS[bestClass],
      legacy_threshold: {
        alarm_triggered: Math.abs(dp) > 3.0,
        delta_prx_db: Math.round(dp * 10) / 10,
        verdict: Math.abs(dp) > 3.0 ? 'ALARM: Power deviated > 3 dB' : 'SILENT: Power within 3 dB window',
        limitation: 'Threshold alarm misses 40%+ of optical faults (dispersion, EDFA noise, micro-bends).'
      }
    };
  }

  function getMockBenchmarks() {
    return {
      active_model: 'Random Forest',
      available_models: ['Random Forest', 'Gradient Boosting', 'k-Nearest Neighbors', 'Multi-Layer Perceptron'],
      metrics: {
        'Random Forest': {
          model_name: 'Random Forest',
          train_samples: 3500, test_samples: 1050, training_time_seconds: 0.85,
          accuracy: 0.9105, macro_precision: 0.912, macro_recall: 0.908, macro_f1: 0.9069,
          legacy_threshold_accuracy: 0.6381,
          confusion_matrix: [
            [142, 0, 2, 2, 4, 0, 0],
            [0, 150, 0, 0, 0, 0, 0],
            [1, 0, 138, 9, 2, 0, 0],
            [1, 0, 7, 140, 2, 0, 0],
            [3, 0, 1, 1, 145, 0, 0],
            [0, 0, 0, 0, 0, 150, 0],
            [0, 0, 0, 0, 0, 0, 150]
          ],
          confusion_matrix_normalized: [
            [94.7, 0, 1.3, 1.3, 2.7, 0, 0],
            [0, 100, 0, 0, 0, 0, 0],
            [0.7, 0, 92.0, 6.0, 1.3, 0, 0],
            [0.7, 0, 4.7, 93.3, 1.3, 0, 0],
            [2.0, 0, 0.7, 0.7, 96.6, 0, 0],
            [0, 0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 0, 100]
          ],
          per_class_metrics: [
            { class_index: 0, class_name: 'Normal Link', short_name: 'Normal', color: '#3ddc97', precision: 0.97, recall: 0.95, f1_score: 0.96, support: 150, legacy_recall: 0.98 },
            { class_index: 1, class_name: 'Fiber Cut', short_name: 'Cut', color: '#ff6b6b', precision: 1.00, recall: 1.00, f1_score: 1.00, support: 150, legacy_recall: 1.00 },
            { class_index: 2, class_name: 'Macro-Bend', short_name: 'Bend', color: '#ffc857', precision: 0.93, recall: 0.92, f1_score: 0.92, support: 150, legacy_recall: 0.62 },
            { class_index: 3, class_name: 'Dirty Connector', short_name: 'Dirty', color: '#ff9a5c', precision: 0.92, recall: 0.93, f1_score: 0.93, support: 150, legacy_recall: 0.55 },
            { class_index: 4, class_name: 'Aged Fiber', short_name: 'Aged', color: '#b69cff', precision: 0.95, recall: 0.97, f1_score: 0.96, support: 150, legacy_recall: 0.78 },
            { class_index: 5, class_name: 'EDFA Failure', short_name: 'EDFA', color: '#5ee3ff', precision: 1.00, recall: 1.00, f1_score: 1.00, support: 150, legacy_recall: 0.54 },
            { class_index: 6, class_name: 'Chromatic Dispersion', short_name: 'Disp', color: '#ff7ccf', precision: 1.00, recall: 1.00, f1_score: 1.00, support: 150, legacy_recall: 0.00 }
          ],
          feature_importances: [0.24, 0.20, 0.16, 0.14, 0.11, 0.09, 0.06],
          feature_names: ['Rx Power Deviation', 'OSNR Deviation', 'Alpha Slope Deviation', 'OTDR Step Loss', 'Fresnel Spike Height', 'Fiber End Ratio', 'Dispersion Ratio']
        }
      },
      fault_names: FAULT_NAMES,
      fault_short: FAULT_SHORT,
      fault_colors: FAULT_COLORS
    };
  }

  function generateSampleRecords(count = 50) {
    const records = [];
    for (let i = 1; i <= count; i++) {
      const f = i % 7;
      const sim = simulate({ length_km: 40 + (i * 3) % 70, fault_type: f, severity: 0.2 + (i * 0.12) % 0.8, seed: i + 500 });
      records.push({
        id: i,
        fault_type: f,
        fault_name: FAULT_NAMES[f],
        length_km: sim.params.length_km,
        launch_power_dbm: 0.0,
        bit_rate_gbps: 10.0,
        severity_pct: Math.round(sim.params.severity * 100),
        prx_dbm: Math.round(sim.telemetry.measured_prx_dbm * 10) / 10,
        osnr_db: Math.round(sim.telemetry.measured_osnr_db * 10) / 10,
        alpha_db_km: Math.round(sim.otdr_analysis.alpha_measured * 1000) / 1000,
        step_loss_db: Math.round(sim.otdr_analysis.detected_step_db * 10) / 10,
        fresnel_spike_db: Math.round(sim.otdr_analysis.reflection_spike_db * 10) / 10,
        features: sim.features
      });
    }
    return { count: records.length, records, features: ['Rx Power Dev', 'OSNR Dev', 'Alpha Dev', 'Step Loss', 'Spike Height', 'End Ratio', 'Disp Ratio'] };
  }

  function diagnoseCustom(data) {
    const L = data.fiber_length_km || 80.0;
    const pinExp = 0.0 - FRONT - ALPHA0 * L - Math.floor(L / 20) * SPL - ENDC;
    const prxExp = pinExp + G0;
    const osnrExp = 58.0 + pinExp - NF0;
    const tps = 100.0;
    const ratioExp = (D_DISP * L * DLAM0) / tps;
    const mRatio = (data.measured_pulse_spread_ps || 27.2) / tps;

    const feats = [
      clamp((data.measured_prx_dbm - prxExp) / 4.0, -45, 10),
      clamp((data.measured_osnr_db - osnrExp) / 5.0, -45, 10),
      (data.measured_alpha_db_km - ALPHA0) / 0.05,
      (data.step_loss_db || 0.0) / 1.0,
      (data.reflection_spike_db || 0.0) / 3.0,
      clamp((1.0 - (data.detected_fiber_end_km / L)) / 0.05, 0, 20),
      Math.log10(Math.max(1e-3, mRatio) / Math.max(1e-3, ratioExp)) / 0.3
    ];

    const pred = predict({ features: feats });
    const dp = data.measured_prx_dbm - prxExp;
    return {
      inputs: data,
      extracted_features: feats.map(f => Math.round(f * 1000) / 1000),
      prediction: pred,
      legacy_alarm: {
        triggered: Math.abs(dp) > 3.0,
        delta_prx_db: Math.round(dp * 10) / 10,
        message: Math.abs(dp) > 3.0 ? 'ALARM: Exceeds 3 dB tolerance' : 'SILENT: Normal power window'
      }
    };
  }

  return {
    simulate,
    predict,
    getMockBenchmarks,
    generateSampleRecords,
    diagnoseCustom,
    FAULT_NAMES,
    FAULT_SHORT,
    FAULT_COLORS
  };
})();
