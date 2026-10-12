/**
 * FiberGuard Frontend - OTDR Trace Oscilloscope Scope
 * Interactive canvas plotting Rayleigh backscatter curve vs distance,
 * nominal reference, localized fault indicators, and mouse cursor inspection.
 */

class OTDRScope {
  constructor(canvasId, noteId) {
    this.canvas = document.getElementById(canvasId);
    this.noteEl = document.getElementById(noteId);
    this.ctx = this.canvas.getContext('2d');
    this.currentData = null;
    this.mouseX = null;
    this.mouseY = null;
    this.init();
  }

  init() {
    window.addEventListener('resize', () => this.resize());
    this.resize();

    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      this.mouseX = e.clientX - rect.left;
      this.mouseY = e.clientY - rect.top;
      this.render();
    });

    this.canvas.addEventListener('mouseleave', () => {
      this.mouseX = null;
      this.mouseY = null;
      this.render();
    });
  }

  resize() {
    if (!this.canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 500;
    const h = 280;

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = `${h}px`;
    }
  }

  updateData(simData, diagData) {
    this.currentData = { sim: simData, diag: diagData };
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

    if (!this.currentData || !this.currentData.sim) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px "IBM Plex Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting OTDR trace telemetry...', w / 2, h / 2);
      return;
    }

    const sim = this.currentData.sim;
    const diag = this.currentData.diag;
    const otdr = sim.otdr;
    const tr = otdr.trace;
    const L = sim.params.length_km;
    const lMax = otdr.l_max;

    const ml = 48, mr = 16, mt = 16, mb = 36;
    const pw = w - ml - mr;
    const ph = h - mt - mb;
    const yMax = 12.0, yMin = -56.0;

    const kmToPx = km => ml + (pw * (km / lMax));
    const dbToPx = db => mt + (ph * (yMax - db) / (yMax - yMin));

    // Grid lines - Horizontal (dB)
    ctx.textAlign = 'right';
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.lineWidth = 1;

    for (let db = 10; db >= -50; db -= 10) {
      const y = dbToPx(db);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.beginPath();
      ctx.moveTo(ml, y);
      ctx.lineTo(w - mr, y);
      ctx.stroke();
      ctx.fillText(`${db}`, ml - 8, y + 4);
    }

    // Grid lines - Vertical (Distance km)
    ctx.textAlign = 'center';
    const distStep = lMax > 80 ? 20 : 10;
    for (let km = 0; km <= lMax; km += distStep) {
      const x = kmToPx(km);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.beginPath();
      ctx.moveTo(x, mt);
      ctx.lineTo(x, mt + ph);
      ctx.stroke();
      ctx.fillText(`${km}`, x, h - 16);
    }

    // Axis Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px "IBM Plex Sans", sans-serif';
    ctx.fillText('Distance (km)', ml + pw / 2, h - 2);

    ctx.save();
    ctx.translate(14, mt + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Backscatter Power (dB)', 0, 0);
    ctx.restore();

    // Nominal healthy link reference slope (0.20 dB/km)
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(kmToPx(0), dbToPx(0));
    ctx.lineTo(kmToPx(L), dbToPx(-0.20 * L));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.65)';
    ctx.font = '11px "IBM Plex Sans", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Healthy 0.20 dB/km design slope', kmToPx(L * 0.45), dbToPx(-0.20 * L * 0.45) - 8);
    ctx.fillText('OTDR noise floor (-50 dB)', ml + 6, dbToPx(-50) - 6);

    // Actual Simulated OTDR Trace
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < tr.length; i++) {
      const x = kmToPx(i * otdr.dx);
      const y = dbToPx(Math.max(yMin, Math.min(yMax, tr[i])));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // AI Detection Marker Overlay
    if (diag && diag.localized_km != null && diag.predicted_class >= 1 && diag.predicted_class <= 3) {
      const evX = kmToPx(diag.localized_km);
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(evX, mt);
      ctx.lineTo(evX, mt + ph);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#10b981';
      ctx.font = '600 11.5px "Sora", sans-serif';
      const an = sim.otdr_analysis;
      const tagText = diag.predicted_class === 1
        ? `Cut @ ${diag.localized_km.toFixed(1)} km`
        : `Event @ ${diag.localized_km.toFixed(1)} km, Δ${an.detected_step_db.toFixed(1)} dB`;

      ctx.textAlign = evX > w * 0.65 ? 'right' : 'left';
      ctx.fillText(tagText, evX + (evX > w * 0.65 ? -8 : 8), mt + 18);
    }

    // Aged Fiber Slope Annotation
    if (diag && diag.predicted_class === 4) {
      ctx.fillStyle = '#b69cff';
      ctx.font = '600 12px "Sora", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`Measured slope: ${sim.otdr_analysis.alpha_measured.toFixed(2)} dB/km (high attenuation)`, ml + 10, mt + 20);
    }

    // Interactive Mouse Hover Inspector Crosshair
    if (this.mouseX !== null && this.mouseX >= ml && this.mouseX <= w - mr) {
      const hoverKm = ((this.mouseX - ml) / pw) * lMax;
      const sampleIdx = Math.max(0, Math.min(tr.length - 1, Math.round(hoverKm / otdr.dx)));
      const hoverDb = tr[sampleIdx];
      const hoverY = dbToPx(hoverDb);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(this.mouseX, mt);
      ctx.lineTo(this.mouseX, mt + ph);
      ctx.moveTo(ml, hoverY);
      ctx.lineTo(w - mr, hoverY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tooltip Box
      const tipText = `${hoverKm.toFixed(1)} km | ${hoverDb.toFixed(1)} dB`;
      ctx.font = '11.5px "JetBrains Mono", monospace';
      const tipW = ctx.measureText(tipText).width + 16;
      const tipX = Math.min(w - mr - tipW, Math.max(ml, this.mouseX + 8));
      const tipY = Math.max(mt + 20, hoverY - 12);

      ctx.fillStyle = '#0e1836';
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(tipX, tipY - 16, tipW, 22, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#f1f5f9';
      ctx.textAlign = 'left';
      ctx.fillText(tipText, tipX + 8, tipY);
    }
  }

  updateNote() {
    if (!this.noteEl || !this.currentData || !this.currentData.sim) return;
    const f = this.currentData.sim.params.fault_type;

    if (f === 0) {
      this.noteEl.innerHTML = '<strong>Nominal Baseline Trace:</strong> Uniform 0.20 dB/km Rayleigh slope. Spikes at endpoints are Tx/Rx patch connectors; micro-steps at 20 km intervals represent 0.1 dB fusion splices.';
    } else if (f === 1) {
      this.noteEl.innerHTML = '<strong style="color:#ef4444">Catastrophic Fiber Break:</strong> Prominent Fresnel reflection peak occurs at fracture point, followed immediately by backscatter collapse to the -50 dB noise floor.';
    } else if (f === 2) {
      this.noteEl.innerHTML = '<strong style="color:#f59e0b">Non-Reflective Macro-Bend:</strong> Localized step insertion loss without any Fresnel reflection spike. Backscatter slope before and after the bend remains identical.';
    } else if (f === 3) {
      this.noteEl.innerHTML = '<strong style="color:#ff9a5c">Reflective Dirty Connector:</strong> Step loss combined with high Fresnel reflection spike caused by glass-to-air refractive index mismatch.';
    } else if (f === 4) {
      this.noteEl.innerHTML = '<strong style="color:#b69cff">Distributed Cable Aging:</strong> No localized step event, but the entire backscatter curve exhibits an elevated attenuation slope exceeding 0.25 dB/km.';
    } else if (f === 5) {
      this.noteEl.innerHTML = '<strong style="color:#5ee3ff">EDFA Pump Failure:</strong> <em>OTDR sees nothing wrong!</em> Because OTDR only reads passive fiber backscatter, active amplifier faults are completely invisible. That is why the AI model analyzes power & OSNR multi-modally.';
    } else if (f === 6) {
      this.noteEl.innerHTML = '<strong style="color:#ff7ccf">Excessive Chromatic Dispersion:</strong> <em>OTDR sees nothing wrong!</em> Span attenuation is nominal, but pulse broadening causes severe ISI in the eye diagram.';
    }
  }
}
