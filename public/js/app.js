/**
 * FiberGuard Frontend - Main Application Controller
 * Compact, intuitive, and responsive management of Link Simulator,
 * Scopes, AI Model Benchmarks, and Field Summary.
 */

// Application State
const state = {
  activeTab: 'overview',
  params: {
    length_km: 80.0,
    launch_power_dbm: 0.0,
    bit_rate_gbps: 10.0,
    fault_type: 0,
    fault_pos_ratio: 0.45,
    severity: 0.5,
    seed: 42
  },
  simData: null,
  diagData: null,
  modelsBenchmarkData: null,
  selectedBenchmarkModel: 'Random Forest',
  linkVis: null,
  otdrScope: null,
  eyeScope: null
};

// Formatting Helpers
function fmtDbm(val) {
  if (val == null) return '–';
  return (val >= 0 ? '+' : '') + val.toFixed(1);
}

function fmtBER(b) {
  if (b == null) return '–';
  if (b < 1e-15) return '&lt; 10<sup>−15</sup>';
  if (b > 0.1) return b.toFixed(2);
  const exp = Math.floor(Math.log10(b));
  const mant = b / Math.pow(10, exp);
  return `${mant.toFixed(1)} × 10<sup>${String(exp).replace('-', '−')}</sup>`;
}

// DOM Elements shortcut
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
  initTabs();
  initControls();
  initTheme();
  initIncidentPresets();

  // Instantiate Canvas Scopes
  state.linkVis = new LinkVisualizer('cLink');
  state.otdrScope = new OTDRScope('cOtdr', 'otdrNote');
  state.eyeScope = new EyeScope('cEye', 'eyeNote');

  // Check backend connectivity
  await checkSystemStatus();

  // Initial Run
  await updateSimulation();

  // Load benchmarks
  loadBenchmarks();
});

// System Status Check
async function checkSystemStatus() {
  const status = await API.checkHealth();
  const indicator = $('statusIndicator');
  const txt = $('statusText');
  const modelSelect = $('modelSelect');

  if (API.isBackendOnline) {
    indicator.classList.add('online');
    txt.innerHTML = `Backend Online · <b>${status.active_model}</b>`;
    if (status.available_models) {
      modelSelect.innerHTML = status.available_models
        .map(m => `<option value="${m}" ${m === status.active_model ? 'selected' : ''}>${m}</option>`)
        .join('');
    }
  } else {
    indicator.classList.remove('online');
    txt.innerHTML = `Standalone · <b>k-NN Fallback</b>`;
  }
}

// Tab Switching
function initTabs() {
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      if (targetTab) switchTab(targetTab);
    });
  });
}
window.switchTab = switchTab;

function switchTab(tabId) {
  state.activeTab = tabId;
  $$('.tab-btn').forEach(b => b.classList.toggle('active', b.getAttribute('data-tab') === tabId));
  $$('.tab-panel').forEach(v => v.classList.toggle('active', v.id === `tab-${tabId}`));

  // Trigger canvas redraws on tab activation
  if (tabId === 'overview') {
    setTimeout(() => {
      state.linkVis?.resize();
    }, 40);
  } else if (tabId === 'scopes') {
    setTimeout(() => {
      state.otdrScope?.resize();
      state.eyeScope?.resize();
      if (state.simData) {
        state.otdrScope?.render();
        state.eyeScope?.render();
      }
    }, 40);
  } else if (tabId === 'benchmarks') {
    loadBenchmarks();
  }
}

// Controls & Sliders
function initControls() {
  $('sL').addEventListener('input', e => {
    state.params.length_km = parseFloat(e.target.value);
    syncControlsUI();
    updateSimulation();
  });

  $('sP').addEventListener('input', e => {
    state.params.launch_power_dbm = parseFloat(e.target.value);
    syncControlsUI();
    updateSimulation();
  });

  $('sR').addEventListener('change', e => {
    state.params.bit_rate_gbps = parseFloat(e.target.value);
    updateSimulation();
  });

  $('sPos').addEventListener('input', e => {
    state.params.fault_pos_ratio = parseFloat(e.target.value) / 100.0;
    syncControlsUI();
    updateSimulation();
  });

  $('sSev').addEventListener('input', e => {
    state.params.severity = parseFloat(e.target.value) / 100.0;
    syncControlsUI();
    updateSimulation();
  });

  $$('.fault-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const f = parseInt(chip.getAttribute('data-fault'), 10);
      state.params.fault_type = f;
      syncControlsUI();
      updateSimulation();
    });
  });

  $('bMeasure').addEventListener('click', () => {
    state.params.seed = (state.params.seed || 42) + 1;
    updateSimulation();
  });

  $('bRandom').addEventListener('click', () => {
    let f;
    do { f = 1 + Math.floor(Math.random() * 6); } while (f === state.params.fault_type);
    state.params.fault_type = f;
    state.params.fault_pos_ratio = 0.15 + 0.70 * Math.random();
    state.params.severity = 0.2 + 0.75 * Math.random();
    state.params.seed = Math.floor(Math.random() * 10000);
    $('sPos').value = Math.round(state.params.fault_pos_ratio * 100);
    $('sSev').value = Math.round(state.params.severity * 100);
    syncControlsUI();
    updateSimulation();
  });

  $('modelSelect').addEventListener('change', async e => {
    const selected = e.target.value;
    await API.selectModel(selected);
    await checkSystemStatus();
    updateSimulation();
  });
}

function syncControlsUI() {
  const p = state.params;
  $('vL').textContent = `${p.length_km.toFixed(0)} km`;
  $('vP').textContent = `${fmtDbm(p.launch_power_dbm)} dBm`;
  $('vPos').textContent = `${(p.fault_pos_ratio * p.length_km).toFixed(1)} km`;
  $('vSev').textContent = `${Math.round(p.severity * 100)}%`;

  const isDistanceApplicable = (p.fault_type >= 1 && p.fault_type <= 3);
  const isSeverityApplicable = (p.fault_type > 0);

  $('cPos').style.display = isDistanceApplicable ? 'block' : 'none';
  $('cSev').style.display = isSeverityApplicable ? 'block' : 'none';

  $$('.fault-chip').forEach(chip => {
    const cf = parseInt(chip.getAttribute('data-fault'), 10);
    chip.classList.toggle('active', cf === p.fault_type);
  });

  const sevLabels = ['Severity', 'Severity', 'Bend Tightness', 'Contamination Level', 'Cable Aging', 'Pump Degradation', 'Linewidth'];
  $('lSev').textContent = sevLabels[p.fault_type] || 'Severity';
}

// Main Simulation & Diagnostic Update Pipeline
async function updateSimulation() {
  try {
    const simRes = await API.simulate(state.params);
    state.simData = simRes;

    const diagRes = await API.predict({
      simulation_params: state.params,
      features: simRes.features,
      model_name: API.activeModel
    });
    state.diagData = diagRes;

    state.linkVis?.updateData(simRes, diagRes);
    state.otdrScope?.updateData(simRes, diagRes);
    state.eyeScope?.updateData(simRes);

    renderKPIs(simRes.telemetry);
    renderDiagnosis(simRes, diagRes);
    renderPowerBudget(simRes.power_budget);

  } catch (err) {
    console.error('[FiberGuard] Error updating simulation:', err);
  }
}

// Render Real-Time KPIs
function renderKPIs(tel) {
  const isCut = tel.is_cut;
  const margin = tel.margin_db;

  const pVal = isCut ? 'LOS' : `${fmtDbm(tel.measured_prx_dbm)}`;
  const pSub = isCut ? 'Loss of Signal' : `${margin >= 0 ? '+' : ''}${margin.toFixed(1)} dB margin`;
  const pClass = isCut ? 'bad' : (margin >= 3.0 ? 'good' : (margin >= 0.0 ? 'warn' : 'bad'));
  updateKpiCard('kpiRxPower', pVal, isCut ? '' : 'dBm', pSub, pClass);

  const oVal = isCut ? '–' : `${tel.measured_osnr_db.toFixed(1)}`;
  const oSub = isCut ? 'No carrier' : (tel.measured_osnr_db >= 22.0 ? 'Nominal (>22 dB)' : (tel.measured_osnr_db >= 16.0 ? 'Degraded ASE' : 'Severe noise'));
  const oClass = isCut ? 'bad' : (tel.measured_osnr_db >= 22.0 ? 'good' : (tel.measured_osnr_db >= 16.0 ? 'warn' : 'bad'));
  updateKpiCard('kpiOsnr', oVal, isCut ? '' : 'dB', oSub, oClass);

  const qVal = (tel.q_effective > 0.1) ? tel.q_effective.toFixed(1) : '–';
  const qDb = tel.q_db != null ? `${tel.q_db.toFixed(1)} dB` : '–';
  const qClass = (tel.ber < 1e-9) ? 'good' : ((tel.ber < 1e-3) ? 'warn' : 'bad');
  updateKpiCard('kpiQFactor', qVal, '', qDb, qClass);

  const bVal = fmtBER(tel.ber);
  const bSub = (tel.ber < 1e-9) ? 'Error-free' : ((tel.ber < 1e-3) ? 'Hard FEC active' : 'Link Outage');
  const bClass = (tel.ber < 1e-9) ? 'good' : ((tel.ber < 1e-3) ? 'warn' : 'bad');
  updateKpiCard('kpiBer', bVal, '', bSub, bClass, true);

  const sVal = tel.pulse_spread_ps.toFixed(0);
  const sRatio = (tel.pulse_spread_ps / tel.bit_period_ps);
  const sSub = `${(sRatio * 100).toFixed(0)}% of UI`;
  const sClass = (sRatio < 0.5) ? 'good' : ((sRatio < 1.0) ? 'warn' : 'bad');
  updateKpiCard('kpiDispersion', sVal, 'ps', sSub, sClass);
}

function updateKpiCard(id, val, unit, sub, statusClass, isHtml = false) {
  const el = $(id);
  if (!el) return;
  el.className = `kpi-box ${statusClass}`;
  const numEl = el.querySelector('.kpi-num');
  if (numEl) numEl.innerHTML = `${val} <small>${unit}</small>`;
  const descEl = el.querySelector('.kpi-desc');
  if (descEl) descEl.textContent = sub;
}

// Render Clean, Uncluttered AI Diagnosis Card
function renderDiagnosis(sim, diag) {
  const diagContainer = $('diagContent');
  if (!diagContainer) return;

  const pClass = diag.predicted_class;
  const isMatch = (pClass === sim.params.fault_type);
  const confPct = Math.round(diag.confidence * 100);

  let locBadge = '';
  if (diag.localized_km != null && pClass >= 1 && pClass <= 3) {
    locBadge = ` @ <strong>${diag.localized_km.toFixed(1)} km</strong>`;
  }

  const details = diag.details || {};
  const legacy = diag.legacy_threshold || {};

  diagContainer.innerHTML = `
    <div class="diag-card-clean">
      <div class="verdict-banner">
        <div class="verdict-left">
          <div class="verdict-pill-icon" style="background: ${diag.color}">
            ${diag.short_name[0]}
          </div>
          <div>
            <h3>${diag.class_name}${locBadge}</h3>
            <p>${isMatch ? '<span style="color:var(--emerald)">✓ Matches injected fault</span>' : `<span style="color:var(--rose)">True: ${sim.params.fault_name}</span>`}</p>
          </div>
        </div>
        <div class="conf-badge">${confPct}% Certainty</div>
      </div>

      <div class="prob-single-row">
        <span style="color:var(--text-muted);width:85px">${diag.class_name}</span>
        <div class="prob-meter">
          <div class="prob-meter-fill" style="width: ${confPct}%; background: ${diag.color}"></div>
        </div>
        <span style="font-family:var(--font-mono);font-size:11.5px">${confPct}%</span>
      </div>

      <div class="diag-simple-summary">
        <div>
          <b>Signature:</b> ${details.signature || 'Nominal readings.'}
        </div>
        <div>
          <b>Action:</b> ${details.action || 'No action needed.'}
        </div>
      </div>

      <div class="alarm-tag-row">
        <div class="alarm-tag">
          <b>Legacy Alarm:</b> ${legacy.alarm_triggered ? '<span style="color:var(--rose)">Triggered</span>' : 'Silent (Normal)'}
        </div>
        <div class="alarm-tag" style="border-color:var(--cyan)">
          <b>AI Detector:</b> <span style="color:var(--cyan)">${diag.class_name}</span>
        </div>
      </div>
    </div>
  `;
}

// Render Clean, Compact Power Budget Table
function renderPowerBudget(budgetItems) {
  const table = $('budgetTable');
  if (!table || !state.simData) return;
  const tel = state.simData.telemetry;
  const p = state.simData.params;

  const isCut = tel.is_cut;
  const fiberLoss = tel.attenuation_alpha * p.length_km;
  const pMargin = tel.margin_db;

  table.innerHTML = `
    <tr>
      <td>Tx Launch Power</td>
      <td>${fmtDbm(p.launch_power_dbm)} dBm</td>
      <td>1550 nm source</td>
    </tr>
    <tr class="${tel.attenuation_alpha > 0.22 ? 'bad' : ''}">
      <td>Fiber Attenuation</td>
      <td>-${fiberLoss.toFixed(1)} dB</td>
      <td>${tel.attenuation_alpha.toFixed(2)} dB/km × ${p.length_km.toFixed(0)} km</td>
    </tr>
    <tr class="${isCut || p.fault_type === 2 || p.fault_type === 3 ? 'bad' : ''}">
      <td>Fault Loss</td>
      <td>${isCut ? '−∞ dB' : (p.fault_type === 2 || p.fault_type === 3 ? '−3.2 dB' : '0.0 dB')}</td>
      <td>${isCut ? 'Fiber cut' : (p.fault_type === 2 || p.fault_type === 3 ? 'Bend / dirt' : 'None')}</td>
    </tr>
    <tr class="${p.fault_type === 5 ? 'bad' : ''}">
      <td>EDFA Booster Gain</td>
      <td>${p.fault_type === 5 ? '+3.0 dB' : '+15.0 dB'}</td>
      <td>${p.fault_type === 5 ? 'Degraded pump' : 'Nominal gain'}</td>
    </tr>
    <tr class="sum ${isCut ? 'bad' : (pMargin >= 3 ? 'good' : 'warn')}">
      <td>Rx Power & Margin</td>
      <td>${isCut ? 'LOS' : fmtDbm(tel.prx_dbm) + ' dBm'}</td>
      <td>${isCut ? 'Loss of Signal' : (pMargin >= 0 ? '+' : '') + pMargin.toFixed(1) + ' dB margin'}</td>
    </tr>
  `;
}

// Tab 3: Model Benchmarks
async function loadBenchmarks() {
  const data = await API.getModels();
  if (!data || !data.metrics) return;
  state.modelsBenchmarkData = data;

  const currentModelName = state.selectedBenchmarkModel || 'Random Forest';
  const modelMetrics = data.metrics[currentModelName] || data.metrics['Random Forest'];
  if (!modelMetrics) return;

  const selectorEl = $('modelPills');
  if (selectorEl) {
    selectorEl.innerHTML = data.available_models.map(m => `
      <button class="model-pill ${m === currentModelName ? 'active' : ''}" onclick="selectBenchmarkModel('${m}')">
        ${m}
      </button>
    `).join('');
  }

  $('bmAccuracy').textContent = `${(modelMetrics.accuracy * 100).toFixed(1)}%`;
  $('bmF1').textContent = `${(modelMetrics.macro_f1 * 100).toFixed(1)}%`;
  $('bmLegacyAcc').textContent = `${(modelMetrics.legacy_threshold_accuracy * 100).toFixed(1)}%`;
  $('bmTrainTime').textContent = `${modelMetrics.training_time_seconds}s`;

  renderConfusionMatrix(modelMetrics, data.fault_short);
  renderPerClassTable(modelMetrics.per_class_metrics);
  renderFeatureImportances(modelMetrics);
}

window.selectBenchmarkModel = function(modelName) {
  state.selectedBenchmarkModel = modelName;
  loadBenchmarks();
};

function renderConfusionMatrix(metrics, faultShort) {
  const cmEl = $('cmTable');
  if (!cmEl || !metrics.confusion_matrix) return;

  const cm = metrics.confusion_matrix;
  const norm = metrics.confusion_matrix_normalized;
  const n = cm.length;

  let html = '<thead><tr><th>True \\ Pred</th>';
  for (let j = 0; j < n; j++) html += `<th>${faultShort[j]}</th>`;
  html += '</tr></thead><tbody>';

  for (let i = 0; i < n; i++) {
    html += `<tr><th style="text-align:right;padding-right:6px">${faultShort[i]}</th>`;
    for (let j = 0; j < n; j++) {
      const count = cm[i][j];
      const pct = norm[i][j];
      const isDiagonal = (i === j);
      const bgOpacity = Math.min(0.9, (pct / 100.0) * 0.85);
      const bgStyle = isDiagonal
        ? `background: rgba(16, 185, 129, ${bgOpacity}); color: ${pct > 40 ? '#022014' : '#f8fafc'}`
        : (count > 0 ? `background: rgba(239, 68, 68, ${bgOpacity}); color: #f8fafc` : 'background: rgba(255,255,255,0.02); color: #64748b');

      html += `<td style="${bgStyle}" title="Actual: ${faultShort[i]}, Predicted: ${faultShort[j]} (${pct}%)">${count || '·'}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody>';
  cmEl.innerHTML = html;
}

function renderPerClassTable(classes) {
  const table = $('perClassTable');
  if (!table || !classes) return;

  const rows = classes.map(c => `
    <tr>
      <td><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${c.color};margin-right:6px"></span>${c.class_name}</td>
      <td style="font-family:var(--font-mono)">${(c.precision * 100).toFixed(0)}%</td>
      <td style="font-family:var(--font-mono)">${(c.recall * 100).toFixed(0)}%</td>
      <td style="font-family:var(--font-mono);font-weight:600;color:var(--cyan)">${(c.f1_score * 100).toFixed(0)}%</td>
      <td style="font-family:var(--font-mono);color:${c.legacy_recall < 0.6 ? 'var(--rose)' : 'var(--text-muted)'}">${(c.legacy_recall * 100).toFixed(0)}%</td>
    </tr>
  `).join('');

  table.innerHTML = `
    <thead>
      <tr>
        <th>Class</th>
        <th>Prec</th>
        <th>Recall</th>
        <th>F1</th>
        <th>Legacy Alarm</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  `;
}

function renderFeatureImportances(metrics) {
  const container = $('featImportanceBox');
  if (!container || !metrics.feature_importances) return;

  const names = ['Rx Power', 'OSNR', 'Slope α', 'Step Loss', 'Fresnel Spike', 'Cut Ratio', 'Dispersion'];
  const imps = metrics.feature_importances;

  const html = imps.slice(0, 5).map((val, idx) => {
    const pct = Math.round(val * 100);
    return `
      <div style="margin-bottom: 6px">
        <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
          <span>${names[idx]}</span>
          <span style="font-family:var(--font-mono);color:var(--cyan)">${(val * 100).toFixed(0)}%</span>
        </div>
        <div style="height:4px;background:var(--bg-input);border-radius:2px;overflow:hidden">
          <div style="height:100%;width:${pct}%;background:var(--cyan);border-radius:2px"></div>
        </div>
      </div>
    `;
  }).join('');

  container.innerHTML = html;
}

// Tab 4: Compact Incident Presets
function initIncidentPresets() {
  const presets = [
    { title: "Trench Cut (42 km)", fault: 1, pos: 0.52, sev: 0.9, len: 80, icon: "🚜" },
    { title: "Manhole Bend", fault: 2, pos: 0.38, sev: 0.75, len: 65, icon: "🤏" },
    { title: "Dirty Connector", fault: 3, pos: 0.25, sev: 0.8, len: 70, icon: "🔍" },
    { title: "Water Ingress Aging", fault: 4, pos: 0.5, sev: 0.85, len: 90, icon: "💧" },
    { title: "EDFA Pump Degraded", fault: 5, pos: 0.5, sev: 0.9, len: 85, icon: "⚡" },
    { title: "Dispersion Overload", fault: 6, pos: 0.5, sev: 0.95, len: 95, icon: "〰️" }
  ];

  const grid = $('incidentGrid');
  if (!grid) return;

  grid.innerHTML = presets.map((p, idx) => `
    <div class="preset-card" onclick="loadIncidentPreset(${idx})">
      <span>${p.icon}</span>
      <span>${p.title}</span>
    </div>
  `).join('');
}

window.loadIncidentPreset = async function(idx) {
  const presets = [
    { fault: 1, pos: 0.52, sev: 0.9, len: 80 },
    { fault: 2, pos: 0.38, sev: 0.75, len: 65 },
    { fault: 3, pos: 0.25, sev: 0.8, len: 70 },
    { fault: 4, pos: 0.5, sev: 0.85, len: 90 },
    { fault: 5, pos: 0.5, sev: 0.9, len: 85 },
    { fault: 6, pos: 0.5, sev: 0.95, len: 95 }
  ];

  const p = presets[idx];
  state.params.fault_type = p.fault;
  state.params.fault_pos_ratio = p.pos;
  state.params.severity = p.sev;
  state.params.length_km = p.len;
  $('sL').value = p.len;
  $('sPos').value = Math.round(p.pos * 100);
  $('sSev').value = Math.round(p.sev * 100);
  syncControlsUI();

  switchTab('overview');
  await updateSimulation();
};

// Theme Switching
function initTheme() {
  const btn = $('themeToggle');
  if (!btn) return;
  btn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    btn.innerHTML = next === 'dark' ? '☀️' : '🌙';
  });
}
