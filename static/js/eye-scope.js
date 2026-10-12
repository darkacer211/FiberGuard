/**
 * FiberGuard Frontend - Eye Diagram Oscilloscope Scope
 * Simulates high-speed optical receiver eye patterns with PRBS transitions,
 * dispersion pulse broadening, ASE noise rail thickening, and eye opening metrics.
 */

class EyeScope {
  constructor(canvasId, noteId) {
    this.canvas = document.getElementById(canvasId);
    this.noteEl = document.getElementById(noteId);
    this.ctx = this.canvas.getContext('2d');
    this.currentData = null;
    this.init();
  }

  init() {
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  resize() {
    if (!this.canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 420;
    const h = 280;

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = `${h}px`;
    }
  }

  updateData(simData) {
    this.currentData = simData;
    this.render();
    this.updateNote();
  }

  render() {
    if (!this.canvas || !this.ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;

    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    if (!this.currentData) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting eye pattern waveform...', w / 2, h / 2);
      return;
    }

    const sim = this.currentData;
    const tel = sim.telemetry;
    const params = sim.params;
    const isCut = tel.is_cut;

    const ml = 16, mr = 16, mt = 16, mb = 26;
    const pw = w - ml - mr;
    const ph = h - mt - mb;
    const yMin = -0.35, yMax = 1.35;

    const tToPx = t => ml + (pw * (t + 0.5) / 2.0);
    const vToPx = v => mt + (ph * (yMax - v) / (yMax - yMin));

    // Oscilloscope Reticle Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;

    // Voltage levels: Logic 0, Decision Threshold 0.5, Logic 1
    for (const v of [0.0, 0.5, 1.0]) {
      ctx.beginPath();
      ctx.moveTo(ml, vToPx(v));
      ctx.lineTo(w - mr, vToPx(v));
      ctx.stroke();
    }

    // Time ticks: -0.5T, 0T, 0.5T, 1.0T, 1.5T
    for (const t of [-0.5, 0.0, 0.5, 1.0, 1.5]) {
      ctx.beginPath();
      ctx.moveTo(tToPx(t), mt);
      ctx.lineTo(tToPx(t), mt + ph);
      ctx.stroke();
    }

    // PRBS Eye Trajectories
    // Gaussian pulse filter function
    function gPulse(u, s) {
      if (s <= 0.001) return Math.abs(u) <= 0.5 ? 1.0 : 0.0;
      const d = Math.SQRT2 * s;
      function erf(x) {
        const z = Math.abs(x), t = 1.0 / (1.0 + 0.5 * z);
        const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1.0 - r : r - 1.0;
      }
      return 0.5 * (erf((u + 0.5) / d) - erf((u - 0.5) / d));
    }

    // Seeded generator for smooth persistent waveform
    let s = 12345;
    function rand() {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    }
    function randn() {
      let u = 0; while (u === 0) u = rand();
      const v = rand();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    const sigma = Math.sqrt(0.12 * 0.12 + Math.pow((tel.pulse_spread_ps / tel.bit_period_ps) / 2.355, 2));
    const noiseSigma = Math.max(0.01, Math.min(0.42, 1.0 / (2.0 * Math.max(0.1, tel.q_effective))));

    const numTraces = 110;
    const timeStep = 0.04;
    ctx.lineWidth = 1.1;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';

    for (let tr = 0; tr < numTraces; tr++) {
      const bits = [];
      for (let k = 0; k < 6; k++) bits.push(rand() > 0.5 ? 1 : 0);

      ctx.beginPath();
      for (let t = -0.5, idx = 0; t <= 1.501; t += timeStep, idx++) {
        let val = 0.0;
        for (let k = -2; k <= 3; k++) {
          val += bits[k + 2] * gPulse(t - k, sigma);
        }
        val = (isCut ? 0.0 : val) + (noiseSigma * randn());

        const px = tToPx(t);
        const py = vToPx(val);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Decision Threshold Center Line
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(ml, vToPx(0.5));
    ctx.lineTo(w - mr, vToPx(0.5));
    ctx.stroke();
    ctx.setLineDash([]);

    // Axis Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "IBM Plex Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('2 Bit Periods (2UI)', ml + pw / 2, h - 6);

    // Eye Opening Metric Badge
    const eoPct = Math.round(tel.eye_opening_pct);
    let badgeColor = '#10b981';
    let badgeText = `Eye Opening: ${eoPct}%`;

    if (isCut) {
      badgeColor = '#ef4444';
      badgeText = 'LOSS OF SIGNAL (LOS)';
    } else if (eoPct < 30 || tel.ber > 1e-3) {
      badgeColor = '#ef4444';
      badgeText = `Eye Closed (${eoPct}%) · Severe ISI`;
    } else if (eoPct < 60 || tel.ber > 1e-9) {
      badgeColor = '#f59e0b';
      badgeText = `Marginal Eye (${eoPct}%) · FEC Active`;
    }

    ctx.fillStyle = badgeColor;
    ctx.font = '600 12.5px "Sora", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(badgeText, ml + 8, mt + 18);
  }

  updateNote() {
    if (!this.noteEl || !this.currentData) return;
    const f = this.currentData.params.fault_type;

    if (f === 1) {
      this.noteEl.innerHTML = '<strong style="color:#ef4444">No Signal:</strong> Zero optical photons reaching receiver; eye pattern collapses to baseline thermal noise floor.';
    } else if (f === 6) {
      this.noteEl.innerHTML = '<strong style="color:#ff7ccf">Severe Intersymbol Interference (ISI):</strong> Pulse broadening Δτ exceeds the bit period, causing optical pulses to smear into adjacent bit slots and squeezing the eye vertically shut.';
    } else if (f === 5) {
      this.noteEl.innerHTML = '<strong style="color:#5ee3ff">Amplified Spontaneous Emission (ASE) Noise:</strong> Degraded EDFA pump laser increases noise figure, heavily thickening the logic-0 and logic-1 rails vertically with high optical noise.';
    } else {
      this.noteEl.innerHTML = '<strong>High Eye Opening:</strong> Wide vertical opening guarantees low BER (< 10⁻¹²). Vertical height represents noise margin; horizontal width represents jitter tolerance.';
    }
  }
}
