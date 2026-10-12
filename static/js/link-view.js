/**
 * FiberGuard Frontend - Optical Link Canvas Visualizer
 * Animates photon packet propagation, dispersion broadening, backscatter reflections,
 * EDFA optical amplification, and fault localization markers.
 */

class LinkVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.currentData = null;
    this.animationId = null;
    this.pulsePhase = 0;
    this.init();
  }

  init() {
    window.addEventListener('resize', () => this.resize());
    this.resize();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  resize() {
    if (!this.canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 800;
    const h = 176;

    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = `${h}px`;
    }
  }

  updateData(simData, diagData) {
    this.currentData = { sim: simData, diag: diagData };
  }

  animate(time) {
    this.render(time);
    requestAnimationFrame(this.animate);
  }

  render(time) {
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
      ctx.fillText('Initializing optical link visualizer...', w / 2, h / 2);
      return;
    }

    const sim = this.currentData.sim;
    const diag = this.currentData.diag;
    const params = sim.params;
    const telemetry = sim.telemetry;

    const L = params.length_km;
    const x0 = 94;
    const x1 = w - 176;
    const yCenter = 84;
    const fiberSpanPx = x1 - x0;

    const kmToPx = km => x0 + (fiberSpanPx * (km / L));

    // Draw Optical Fiber with power-based color intensity
    const isCut = telemetry.is_cut;
    const faultKm = params.fault_km;

    for (let px = x0; px < x1; px += 3) {
      const km = ((px - x0) / fiberSpanPx) * L;
      if (isCut && km > faultKm) {
        // Fiber cut: darkness beyond break point
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.25)';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 5]);
      } else {
        ctx.setLineDash([]);
        // Local power along fiber: P(z) = Ptx - 0.5 - alpha * z
        const localPowerDbm = params.launch_power_dbm - 0.5 - (telemetry.attenuation_alpha * km);
        const normIntensity = Math.max(0.12, Math.min(1.0, (localPowerDbm + 45.0) / 48.0));
        ctx.strokeStyle = `rgba(56, 189, 248, ${normIntensity})`;
        ctx.lineWidth = 4;
      }
      ctx.beginPath();
      ctx.moveTo(px, yCenter);
      ctx.lineTo(px + 3.5, yCenter);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw Forward Optical Pulses (width broadens with dispersion)
    const numPulses = 15;
    const speed = 0.00015;
    for (let i = 0; i < numPulses; i++) {
      const phase = ((time * speed) + (i / numPulses)) % 1.0;
      const km = phase * L;

      if (isCut && km > faultKm) continue;

      const px = kmToPx(km);
      // Pulse broadening: dispersion ratio widens the ellipse
      const dispRatioAtKm = (telemetry.pulse_spread_ps / telemetry.bit_period_ps) * (km / L);
      const radiusX = Math.max(3.0, 3.2 + Math.min(dispRatioAtKm, 2.5) * 6.5);
      const powerAtKm = params.launch_power_dbm - (telemetry.attenuation_alpha * km);
      const alphaP = Math.max(0.15, Math.min(1.0, (powerAtKm + 45.0) / 48.0));

      ctx.fillStyle = `rgba(56, 189, 248, ${alphaP})`;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8 * alphaP;
      ctx.beginPath();
      ctx.ellipse(px, yCenter, radiusX, 3.0, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.shadowBlur = 0;

    // Draw Backward Reflectometry Packets (if break or dirty connector present)
    if (isCut || params.fault_type === 3) {
      const bSpeed = 0.00035;
      const bPhase = (time * bSpeed) % 1.0;
      for (let j = 0; j < 3; j++) {
        const q = (bPhase + j / 3.0) % 1.0;
        const km = faultKm * (1.0 - q);
        const px = kmToPx(km);
        ctx.fillStyle = `rgba(245, 158, 11, ${0.85 * (1.0 - q)})`;
        ctx.beginPath();
        ctx.arc(px, yCenter, 2.8, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Transmitter Node Box (Tx)
    this.drawNodeBox(ctx, 16, yCenter, 'Tx', '1550 nm', '#38bdf8', 'Launch');

    // Receiver Node Box (Rx)
    const rxMargin = telemetry.margin_db;
    const rxColor = isCut ? '#ef4444' : (rxMargin >= 3.0 ? '#10b981' : (rxMargin >= 0.0 ? '#f59e0b' : '#ef4444'));
    const rxLabel = isCut ? 'No Light' : `${telemetry.prx_dbm.toFixed(1)} dBm`;
    this.drawNodeBox(ctx, w - 82, yCenter, 'Rx', rxLabel, rxColor, 'Receiver');

    // Fiber connecting EDFA to Rx
    ctx.strokeStyle = isCut ? 'rgba(100, 116, 139, 0.25)' : `rgba(56, 189, 248, ${Math.max(0.2, (telemetry.prx_dbm + 30) / 40)})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x1 + 6, yCenter);
    ctx.lineTo(x1 + 48, yCenter);
    ctx.moveTo(x1 + 48, yCenter);
    ctx.lineTo(w - 82, yCenter);
    ctx.stroke();

    // EDFA Optical Amplifier Triangle
    const isEdfaFault = params.fault_type === 5;
    const edfaColor = isEdfaFault ? '#ef4444' : '#10b981';
    ctx.fillStyle = isEdfaFault ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    ctx.strokeStyle = edfaColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1 + 14, yCenter - 22);
    ctx.lineTo(x1 + 14, yCenter + 22);
    ctx.lineTo(x1 + 46, yCenter);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.textAlign = 'center';
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11.5px "IBM Plex Sans", sans-serif';
    const gainText = isEdfaFault ? 'EDFA Degraded' : 'EDFA +15 dB';
    ctx.fillText(gainText, x1 + 30, yCenter + 38);

    // Distance Kilometer Ticks
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "IBM Plex Sans", sans-serif';
    for (let k = 0; k <= 4; k++) {
      const km = (L * k) / 4.0;
      const px = kmToPx(km);
      ctx.fillRect(px, yCenter + 12, 1, 5);
      ctx.fillText(`${km.toFixed(0)} km`, px, yCenter + 28);
    }

    // Injected Fault Marker (Ground Truth)
    if (params.fault_type >= 1 && params.fault_type <= 3) {
      const fxPx = kmToPx(faultKm);
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(fxPx, yCenter - 44);
      ctx.lineTo(fxPx, yCenter + 10);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ef4444';
      ctx.font = '600 11.5px "Sora", sans-serif';
      const faultName = ['Normal', 'Fiber Cut', 'Macro-Bend', 'Dirty Connector'][params.fault_type];
      ctx.fillText(`Injected: ${faultName}`, Math.max(65, Math.min(w - 95, fxPx)), yCenter - 48);
    } else if (params.fault_type === 4) {
      ctx.fillStyle = '#b69cff';
      ctx.font = '600 12px "Sora", sans-serif';
      ctx.fillText('Distributed Aging across whole span (Injected)', (x0 + x1) / 2, yCenter - 42);
    } else if (params.fault_type === 5) {
      ctx.fillStyle = '#5ee3ff';
      ctx.font = '600 12px "Sora", sans-serif';
      ctx.fillText('EDFA Pump Degradation (Injected)', x1 + 30, yCenter - 30);
    } else if (params.fault_type === 6) {
      ctx.fillStyle = '#ff7ccf';
      ctx.font = '600 12px "Sora", sans-serif';
      ctx.fillText('High Chromatic Dispersion ISI (Injected)', (x0 + x1) / 2, yCenter - 42);
    }

    // AI Localization Estimate Marker
    if (diag && diag.localized_km != null && diag.predicted_class >= 1 && diag.predicted_class <= 3) {
      const ax = kmToPx(Math.max(0, Math.min(L, diag.localized_km)));
      ctx.fillStyle = '#10b981';
      ctx.beginPath();
      ctx.moveTo(ax, yCenter + 16);
      ctx.lineTo(ax - 6, yCenter + 26);
      ctx.lineTo(ax + 6, yCenter + 26);
      ctx.closePath();
      ctx.fill();

      ctx.font = '600 11.5px "Sora", sans-serif';
      ctx.fillText(`AI: ${diag.localized_km.toFixed(1)} km`, Math.max(50, Math.min(w - 110, ax)), yCenter + 46);
    }
  }

  drawNodeBox(ctx, bx, y, title, sub, borderColor, role) {
    ctx.fillStyle = '#0e1836';
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(bx, y - 28, 64, 56, 10);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '700 13.5px "Sora", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(title, bx + 32, y - 4);

    ctx.font = '11px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(sub, bx + 32, y + 14);
  }
}
