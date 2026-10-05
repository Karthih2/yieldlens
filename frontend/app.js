/**
 * YieldLens — Industrial Semiconductor Yield Intelligence & Stability Explorer
 * Frontend Application Engine
 */

const API_BASE = 'http://127.0.0.1:8000';

// Shortlist sensors required by the reduced Random Forest model
const SHORTLIST_SENSORS = [
  'sensor_21', 'sensor_0', 'sensor_285', 'sensor_225', 'sensor_290',
  'sensor_71', 'sensor_59', 'sensor_571', 'sensor_103', 'sensor_406',
  'sensor_70', 'sensor_138', 'sensor_473', 'sensor_287', 'sensor_117',
  'sensor_100', 'sensor_95'
];

// App State
const state = {
  currentView: 'dashboard',
  apiOnline: false,
  healthData: null,
  dashboardData: null,
  lastPrediction: null,
  activeBatchUpload: null,
  selectedBatchId: null,
  
  // Charts
  charts: {
    dashImportance: null,
    dashScatter: null,
    shapBar: null
  },

  // Stability Explorer Pagination State
  stability: {
    page: 1,
    pageSize: 50,
    search: '',
    tier: '',
    shortlistedOnly: false,
    total: 0,
    data: []
  },

  // Sample data cache
  sampleRows: []
};

// ================= DOM HELPER FUNCTIONS =================

const $ = id => document.getElementById(id);

function showToast(message, type = 'info', duration = 3500) {
  const container = $('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span> ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
  $('topbarTime').textContent = timeStr;
}

// ================= API SERVICE =================

async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' },
      ...options
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errJson.detail || `HTTP Error ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    console.error(`API Fetch Error [${endpoint}]:`, err);
    throw err;
  }
}

async function checkSystemHealth() {
  try {
    const data = await apiFetch('/health');
    state.apiOnline = true;
    state.healthData = data;

    $('sidebarStatusDot').className = 'status-indicator-dot online';
    $('sidebarStatusTitle').textContent = 'API Online (Operational)';
    $('sidebarStatusMeta').textContent = `Model: ${data.model_version || 'v1.0.0'}`;
    $('topbarModelBadge').textContent = `Model ${data.model_version || 'v1.0.0'}`;
  } catch (err) {
    state.apiOnline = false;
    $('sidebarStatusDot').className = 'status-indicator-dot offline';
    $('sidebarStatusTitle').textContent = 'API Offline';
    $('sidebarStatusMeta').textContent = 'Backend unreachable';
  }
}

// ================= NAVIGATION =================

function navigateTo(viewName) {
  const targetView = $(`view-${viewName}`);
  if (!targetView) return;

  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  targetView.classList.add('active');
  const navItem = $(`nav-${viewName}`);
  if (navItem) navItem.classList.add('active');

  const labels = {
    dashboard: 'Dashboard',
    predict: 'Wafer Yield Predictor',
    uploads: 'Batch Uploads & Inspection',
    stability: 'Stability Explorer'
  };
  $('breadcrumbActive').textContent = labels[viewName] || viewName;
  state.currentView = viewName;

  // View specific load triggers
  if (viewName === 'dashboard') loadDashboard();
  if (viewName === 'uploads') loadUploadsHistory();
  if (viewName === 'stability') loadStabilityExplorer();
}

// ================= DASHBOARD VIEW =================

async function loadDashboard() {
  try {
    const [dashData, stabilityData] = await Promise.all([
      apiFetch('/dashboard'),
      apiFetch('/stability-scores?page=1&page_size=600')
    ]);

    state.dashboardData = dashData;

    // Update Metric Cards
    $('dashTotalPreds').textContent = dashData.total_predictions.toLocaleString();
    $('dashTotalUploads').textContent = dashData.total_uploads.toLocaleString();
    $('dashHighRiskCount').textContent = dashData.high_risk_count.toLocaleString();
    $('dashHighRiskRatio').textContent = `${(dashData.high_risk_ratio * 100).toFixed(1)}% of total scored`;
    $('dashTopKCap').textContent = `K = ${dashData.top_k_capacity}`;

    // Render Recent Uploads
    renderRecentUploadsTable(dashData.recent_uploads);

    // Render Charts
    renderDashboardCharts(stabilityData.scores);

  } catch (err) {
    showToast(`Failed to load dashboard: ${err.message}`, 'error');
  }
}

function renderRecentUploadsTable(uploads) {
  const tbody = $('dashRecentUploadsBody');
  if (!uploads || uploads.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No batch CSV uploads found.</td></tr>`;
    return;
  }

  tbody.innerHTML = uploads.map(u => `
    <tr>
      <td class="mono">#${u.id}</td>
      <td class="mono">${u.filename}</td>
      <td class="mono">${u.row_count}</td>
      <td><span class="badge ${u.flagged_count > 0 ? 'badge-high' : 'badge-low'}">${u.flagged_count} wafers</span></td>
      <td class="mono">Top-${u.top_k}</td>
      <td class="mono">${new Date(u.created_at).toLocaleString()}</td>
      <td>
        <button class="btn btn-sm btn-outline" onclick="inspectBatchFromDash(${u.id})">Inspect Batch</button>
      </td>
    </tr>
  `).join('');
}

function inspectBatchFromDash(uploadId) {
  navigateTo('uploads');
  loadBatchPredictions(uploadId);
}

function getChartColors() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    text: isDark ? '#94a3b8' : '#475569',
    grid: isDark ? '#1e293b' : '#e2e8f0',
    primary: isDark ? '#3b82f6' : '#2563eb',
    shortlist: isDark ? '#10b981' : '#059669',
    other: isDark ? '#64748b' : '#94a3b8',
    bg: isDark ? '#131c2e' : '#ffffff'
  };
}

function renderDashboardCharts(scores) {
  if (!scores || !scores.length) return;
  const colors = getChartColors();

  // 1. Top 10 Shortlisted Sensors by Importance
  const shortlisted = scores.filter(s => s.in_shortlist);
  const top10 = [...shortlisted].sort((a, b) => b.avg_importance - a.avg_importance).slice(0, 10);

  const impCtx = $('dashImportanceChart').getContext('2d');
  if (state.charts.dashImportance) state.charts.dashImportance.destroy();

  state.charts.dashImportance = new Chart(impCtx, {
    type: 'bar',
    data: {
      labels: top10.map(s => s.sensor),
      datasets: [{
        label: 'Avg SHAP Importance',
        data: top10.map(s => s.avg_importance),
        backgroundColor: colors.shortlist,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: colors.text, font: { family: 'JetBrains Mono', size: 10 } }, grid: { display: false } },
        y: { ticks: { color: colors.text, font: { size: 10 } }, grid: { color: colors.grid } }
      }
    }
  });

  // 2. Stability vs Importance Scatter Chart
  const scatterCtx = $('dashScatterChart').getContext('2d');
  if (state.charts.dashScatter) state.charts.dashScatter.destroy();

  const shortlistedData = scores.filter(s => s.in_shortlist).map(s => ({ x: s.stability_score, y: s.avg_importance, sensor: s.sensor }));
  const nonShortlistedData = scores.filter(s => !s.in_shortlist && s.avg_importance > 0).map(s => ({ x: s.stability_score, y: s.avg_importance, sensor: s.sensor }));

  state.charts.dashScatter = new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Shortlisted (17)',
          data: shortlistedData,
          backgroundColor: colors.shortlist,
          pointRadius: 5
        },
        {
          label: 'Other Active Sensors',
          data: nonShortlistedData,
          backgroundColor: colors.other,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: colors.text, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.sensor} — Stability: ${ctx.raw.x.toFixed(2)}, Importance: ${ctx.raw.y.toFixed(5)}`
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Stability Score', color: colors.text }, ticks: { color: colors.text }, grid: { color: colors.grid } },
        y: { title: { display: true, text: 'Avg Importance', color: colors.text }, ticks: { color: colors.text }, grid: { color: colors.grid } }
      }
    }
  });
}

// ================= SINGLE WAFER PREDICTOR VIEW =================

function buildSensorInputsForm() {
  const grid = $('sensorInputsGrid');
  grid.innerHTML = SHORTLIST_SENSORS.map(sensor => `
    <div class="form-group">
      <label class="form-label" for="inp-${sensor}">
        ${sensor}
        <span class="star-tag">★</span>
      </label>
      <input
        type="number"
        step="any"
        id="inp-${sensor}"
        name="${sensor}"
        class="input-text"
        placeholder="Auto-impute median"
      />
    </div>
  `).join('');
}

async function loadSampleRowPreset(targetLabel) {
  try {
    if (!state.sampleRows.length) {
      const sampleRes = await apiFetch('/sample?limit=100');
      state.sampleRows = sampleRes.data;
    }

    const matching = state.sampleRows.filter(r => r.labels === targetLabel);
    if (!matching.length) {
      showToast('No matching sample row found', 'error');
      return;
    }

    const row = matching[Math.floor(Math.random() * matching.length)];
    SHORTLIST_SENSORS.forEach(s => {
      const inp = $(`inp-${s}`);
      if (inp && row[s] !== undefined) {
        inp.value = row[s];
      }
    });

    showToast(`Loaded ${targetLabel === 1 ? 'Fail (High Risk)' : 'Pass (Normal)'} sample wafer values`, 'success');
  } catch (err) {
    showToast(`Failed to load sample row: ${err.message}`, 'error');
  }
}

function clearPredictorForm() {
  SHORTLIST_SENSORS.forEach(s => {
    const inp = $(`inp-${s}`);
    if (inp) inp.value = '';
  });

  $('predictPlaceholder').classList.remove('hidden');
  $('predictResultBox').classList.add('hidden');
  $('shapExplanationPanel').classList.add('hidden');
  state.lastPrediction = null;
  showToast('Predictor form cleared', 'info');
}

async function submitSinglePrediction(e) {
  e.preventDefault();

  const sensorValues = {};
  SHORTLIST_SENSORS.forEach(s => {
    const val = $(`inp-${s}`).value;
    if (val !== '' && !isNaN(val)) {
      sensorValues[s] = parseFloat(val);
    }
  });

  const btn = $('predictSubmitBtn');
  btn.disabled = true;
  btn.textContent = 'Running Model Inference…';

  try {
    const result = await apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ sensor_values: sensorValues })
    });

    state.lastPrediction = result;
    displayPredictionResult(result);
    showToast(`Wafer prediction complete (ID #${result.prediction_id})`, 'success');
  } catch (err) {
    showToast(`Prediction failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Single Wafer Prediction`;
  }
}

function displayPredictionResult(result) {
  $('predictPlaceholder').classList.add('hidden');
  $('predictResultBox').classList.remove('hidden');
  $('shapExplanationPanel').classList.add('hidden');

  const { prediction_id, fail_probability, percentile_rank, risk_level, is_flagged_top_k, imputed_sensors } = result;

  $('resPredId').textContent = `ID: #${prediction_id}`;

  const pct = (fail_probability * 100).toFixed(1);
  $('resProbPct').textContent = `${pct}%`;
  $('resProbBar').style.width = `${pct}%`;

  const verdictCard = $('verdictCard');
  verdictCard.className = 'panel verdict-panel';

  let riskBadge = '';
  if (risk_level === 'HIGH') {
    riskBadge = '<span class="badge badge-high">HIGH RISK</span>';
    verdictCard.classList.add('fail-border');
    $('resProbBar').style.backgroundColor = 'var(--color-danger)';
  } else if (risk_level === 'MEDIUM') {
    riskBadge = '<span class="badge badge-medium">MEDIUM RISK</span>';
    $('resProbBar').style.backgroundColor = 'var(--color-warning)';
  } else {
    riskBadge = '<span class="badge badge-low">LOW RISK (PASS)</span>';
    verdictCard.classList.add('pass-border');
    $('resProbBar').style.backgroundColor = 'var(--color-success)';
  }

  $('resRiskBadge').innerHTML = riskBadge;
  $('resPercentile').textContent = `${percentile_rank.toFixed(1)}th Percentile (rel. to test batch)`;
  $('resTopKFlag').innerHTML = is_flagged_top_k
    ? '<span class="badge badge-high">⚠️ FLAGGED FOR DEFECT INSPECTION</span>'
    : '<span class="badge badge-low">✓ Clear (Not Flagged)</span>';

  $('resImputedCount').textContent = imputed_sensors && imputed_sensors.length
    ? `${imputed_sensors.length} auto-imputed (${imputed_sensors.join(', ')})`
    : '0 (All 17 inputs provided)';
}

async function loadShapForLastPrediction() {
  if (!state.lastPrediction) return;

  const btn = $('loadShapBtn');
  btn.disabled = true;
  btn.textContent = 'Computing SHAP Attributions…';

  try {
    const shapRes = await apiFetch(`/predictions/${state.lastPrediction.prediction_id}/explain`);
    renderShapExplanation(shapRes);
    showToast('SHAP feature attribution loaded', 'success');
  } catch (err) {
    showToast(`SHAP computation failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Compute & Load SHAP Attribution`;
  }
}

function renderShapExplanation(shapRes) {
  const panel = $('shapExplanationPanel');
  panel.classList.remove('hidden');

  const { top_contributors } = shapRes;
  const colors = getChartColors();

  // Render Horizontal Bar Chart
  const ctx = $('shapBarChart').getContext('2d');
  if (state.charts.shapBar) state.charts.shapBar.destroy();

  const labels = top_contributors.map(c => c.sensor);
  const values = top_contributors.map(c => c.shap_value);
  const barColors = values.map(v => v >= 0 ? '#ef4444' : '#3b82f6');

  state.charts.shapBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP Value',
        data: values,
        backgroundColor: barColors,
        borderRadius: 3
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: colors.text }, grid: { color: colors.grid } },
        y: { ticks: { color: colors.text, font: { family: 'JetBrains Mono' } }, grid: { display: false } }
      }
    }
  });

  // Render Table
  const tbody = $('shapTableBody');
  tbody.innerHTML = top_contributors.map((c, i) => `
    <tr>
      <td class="mono">${i + 1}</td>
      <td class="mono">${c.sensor}</td>
      <td class="mono" style="color: ${c.shap_value >= 0 ? 'var(--color-danger)' : 'var(--color-primary)'}">
        ${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(5)}
      </td>
      <td>
        <span class="badge ${c.shap_value >= 0 ? 'badge-high' : 'badge-accent'}">
          ${c.shap_value >= 0 ? '↑ Increases Fail Risk' : '↓ Decreases Fail Risk'}
        </span>
      </td>
    </tr>
  `).join('');
}

// ================= BATCH UPLOADS & INSPECTION VIEW =================

function initUploadZone() {
  const dropZone = $('dropZone');
  const fileInput = $('csvFileInput');
  const browseBtn = $('browseFileBtn');

  browseBtn.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  $('processUploadBtn').addEventListener('click', processBatchUpload);
}

function handleFileSelected(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please select a valid .csv file', 'error');
    return;
  }

  state.activeBatchFile = file;
  $('uploadFilenameText').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  $('uploadStatusBar').classList.remove('hidden');
}

async function processBatchUpload() {
  if (!state.activeBatchFile) return;

  const btn = $('processUploadBtn');
  btn.disabled = true;
  btn.textContent = 'Processing & Scoring Batch…';

  const formData = new FormData();
  formData.append('file', state.activeBatchFile);

  try {
    const res = await apiFetch('/uploads', {
      method: 'POST',
      body: formData
    });

    showToast(`Successfully processed batch upload "${res.upload.filename}"`, 'success');
    $('uploadStatusBar').classList.add('hidden');
    state.activeBatchFile = null;

    loadUploadsHistory();
    displayBatchPredictions(res);

  } catch (err) {
    showToast(`Batch upload failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Process & Screen Batch';
  }
}

async function loadUploadsHistory() {
  try {
    const uploads = await apiFetch('/uploads');
    $('uploadsHistoryCount').textContent = `${uploads.length} uploads`;

    const tbody = $('uploadsHistoryTableBody');
    if (!uploads.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="table-empty">No CSV batches uploaded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = uploads.map(u => `
      <tr>
        <td class="mono">#${u.id}</td>
        <td class="mono">${u.filename}</td>
        <td class="mono">${u.row_count}</td>
        <td><span class="badge ${u.flagged_count > 0 ? 'badge-high' : 'badge-low'}">${u.flagged_count} wafers</span></td>
        <td class="mono">${new Date(u.created_at).toLocaleString()}</td>
        <td>
          <button class="btn btn-sm btn-outline" onclick="loadBatchPredictions(${u.id})">Inspect Batch</button>
        </td>
      </tr>
    `).join('');

  } catch (err) {
    showToast(`Failed to load upload history: ${err.message}`, 'error');
  }
}

async function loadBatchPredictions(uploadId) {
  try {
    const batchDetail = await apiFetch(`/uploads/${uploadId}`);
    state.selectedBatchDetail = batchDetail;
    displayBatchPredictions(batchDetail);
  } catch (err) {
    showToast(`Failed to load batch details: ${err.message}`, 'error');
  }
}

function displayBatchPredictions(batchDetail) {
  const { upload, predictions } = batchDetail;
  state.currentBatchPredictions = predictions;

  $('batchDetailTitle').textContent = `Batch #${upload.id} — ${upload.filename}`;
  $('batchDetailSub').textContent = `Total Wafers: ${upload.row_count} | Flagged Top-K Wafers: ${upload.flagged_count} (Capacity K=${upload.top_k})`;

  renderFilteredBatchTable();
}

function renderFilteredBatchTable() {
  const predictions = state.currentBatchPredictions;
  if (!predictions) return;

  const filterVal = $('batchRiskFilter').value;
  let filtered = predictions;

  if (filterVal === 'HIGH') filtered = predictions.filter(p => p.risk_level === 'HIGH');
  else if (filterVal === 'MEDIUM') filtered = predictions.filter(p => p.risk_level === 'MEDIUM');
  else if (filterVal === 'LOW') filtered = predictions.filter(p => p.risk_level === 'LOW');
  else if (filterVal === 'TOP_K') filtered = predictions.filter(p => p.is_flagged_top_k);

  const tbody = $('batchPredictionsTableBody');
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="table-empty">No wafer predictions match the filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(p => `
    <tr>
      <td class="mono">#${p.id}</td>
      <td class="mono">Row ${p.row_index !== null ? p.row_index : '--'}</td>
      <td class="mono">${(p.fail_probability * 100).toFixed(1)}%</td>
      <td class="mono">${p.percentile_rank.toFixed(1)}th</td>
      <td>
        <span class="badge ${p.risk_level === 'HIGH' ? 'badge-high' : p.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}">
          ${p.risk_level}
        </span>
      </td>
      <td>
        ${p.is_flagged_top_k ? '<span class="badge badge-high">⚠️ Top-K Flagged</span>' : '<span class="badge badge-low">Clear</span>'}
      </td>
      <td class="mono">${p.imputed_sensors && p.imputed_sensors.length ? p.imputed_sensors.length : '0'}</td>
      <td>
        <button class="btn btn-sm btn-outline" onclick="openShapModal(${p.id})">SHAP Explanation</button>
      </td>
    </tr>
  `).join('');
}

async function downloadTestBatchCsv() {
  try {
    const sampleRes = await apiFetch('/sample?limit=30');
    const rows = sampleRes.data;
    if (!rows || !rows.length) return;

    const headers = Object.keys(rows[0]);
    const csvLines = [
      headers.join(','),
      ...rows.map(r => headers.map(h => r[h]).join(','))
    ];

    const csvBlob = new Blob([csvLines.join('\n')], { type: 'text/csv' });
    const downloadUrl = URL.createObjectURL(csvBlob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = 'test_wafer_batch_sample.csv';
    link.click();
    URL.revokeObjectURL(downloadUrl);

    showToast('Downloaded test wafer batch CSV file', 'success');
  } catch (err) {
    showToast(`Failed to generate sample CSV: ${err.message}`, 'error');
  }
}

// ================= SHAP MODAL =================

async function openShapModal(predictionId) {
  const backdrop = $('shapModalBackdrop');
  const title = $('modalTitle');
  const body = $('modalBody');

  backdrop.classList.remove('hidden');
  title.textContent = `Wafer SHAP Explanation — Prediction #${predictionId}`;
  body.innerHTML = `<div class="loading-spinner"></div>`;

  try {
    const shapRes = await apiFetch(`/predictions/${predictionId}/explain`);
    const { top_contributors } = shapRes;

    body.innerHTML = `
      <div class="table-container">
        <table class="data-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Sensor Column</th>
              <th>SHAP Value</th>
              <th>Directional Impact</th>
            </tr>
          </thead>
          <tbody>
            ${top_contributors.map((c, i) => `
              <tr>
                <td class="mono">${i + 1}</td>
                <td class="mono">${c.sensor}</td>
                <td class="mono" style="color: ${c.shap_value >= 0 ? 'var(--color-danger)' : 'var(--color-primary)'}">
                  ${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(5)}
                </td>
                <td>
                  <span class="badge ${c.shap_value >= 0 ? 'badge-high' : 'badge-accent'}">
                    ${c.shap_value >= 0 ? '↑ Increases Fail Risk' : '↓ Decreases Fail Risk'}
                  </span>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div style="color:var(--color-danger)">Failed to load SHAP: ${err.message}</div>`;
  }
}

function closeModal() {
  $('shapModalBackdrop').classList.add('hidden');
}

// ================= STABILITY EXPLORER VIEW =================

async function loadStabilityExplorer() {
  const { page, pageSize, search, tier, shortlistedOnly } = state.stability;

  let query = `/stability-scores?page=${page}&page_size=${pageSize}`;
  if (search) query += `&search=${encodeURIComponent(search)}`;
  if (tier) query += `&tier=${tier}`;
  if (shortlistedOnly) query += `&shortlisted_only=true`;

  try {
    const res = await apiFetch(query);
    state.stability.data = res.scores;
    state.stability.total = res.total;

    $('stabilityTotalBadge').textContent = `${res.total} Sensors`;
    renderStabilityTable(res.scores);
    renderPaginationControls(res.total, page, pageSize);

  } catch (err) {
    showToast(`Failed to load stability scores: ${err.message}`, 'error');
  }
}

function renderStabilityTable(scores) {
  const tbody = $('stabilityTableBody');
  if (!scores || !scores.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="table-empty">No sensors match your filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = scores.map(s => {
    let tierBadge = '';
    if (s.tier === 'high') tierBadge = '<span class="badge badge-high">High Stability</span>';
    else if (s.tier === 'medium') tierBadge = '<span class="badge badge-medium">Medium</span>';
    else if (s.tier === 'low') tierBadge = '<span class="badge badge-low">Low</span>';
    else tierBadge = '<span class="badge">Zero</span>';

    return `
      <tr>
        <td class="mono" style="font-weight: ${s.in_shortlist ? '700' : '400'}">${s.sensor}</td>
        <td class="mono">${s.stability_score.toFixed(2)}</td>
        <td class="mono">${s.avg_importance.toFixed(5)}</td>
        <td>${tierBadge}</td>
        <td>
          ${s.in_shortlist ? '<span class="badge badge-accent">★ Shortlisted (17)</span>' : '<span class="badge">Excluded</span>'}
        </td>
      </tr>
    `;
  }).join('');
}

function renderPaginationControls(total, page, pageSize) {
  const totalPages = Math.ceil(total / pageSize) || 1;
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);

  $('paginationInfo').textContent = `Showing ${start} to ${end} of ${total} sensors (Page ${page} of ${totalPages})`;

  const container = $('paginationBtns');
  let btnsHtml = `
    <button class="page-btn" ${page <= 1 ? 'disabled' : ''} onclick="changeStabilityPage(${page - 1})">Prev</button>
  `;

  for (let p = Math.max(1, page - 2); p <= Math.min(totalPages, page + 2); p++) {
    btnsHtml += `<button class="page-btn ${p === page ? 'active' : ''}" onclick="changeStabilityPage(${p})">${p}</button>`;
  }

  btnsHtml += `
    <button class="page-btn" ${page >= totalPages ? 'disabled' : ''} onclick="changeStabilityPage(${page + 1})">Next</button>
  `;

  container.innerHTML = btnsHtml;
}

function changeStabilityPage(newPage) {
  state.stability.page = newPage;
  loadStabilityExplorer();
}

// ================= THEME & EVENT BINDINGS =================

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);

  if (state.currentView === 'dashboard' && state.dashboardData) {
    loadDashboard();
  }
}

function bindEvents() {
  // Navigation Links
  document.querySelectorAll('.nav-item[data-view]').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(item.dataset.view);
    });
  });

  $('mobileMenuBtn')?.addEventListener('click', () => {
    $('sidebar').classList.toggle('mobile-open');
  });

  $('themeToggleBtn')?.addEventListener('click', toggleTheme);
  $('dashboardRefreshBtn')?.addEventListener('click', loadDashboard);

  // Predict View Buttons
  $('loadFailSampleBtn')?.addEventListener('click', () => loadSampleRowPreset(1));
  $('loadPassSampleBtn')?.addEventListener('click', () => loadSampleRowPreset(0));
  $('clearPredictFormBtn')?.addEventListener('click', clearPredictorForm);
  $('predictForm')?.addEventListener('submit', submitSinglePrediction);
  $('loadShapBtn')?.addEventListener('click', loadShapForLastPrediction);

  // Uploads View
  $('downloadTestCsvBtn')?.addEventListener('click', downloadTestBatchCsv);
  $('batchRiskFilter')?.addEventListener('change', renderFilteredBatchTable);

  // Modal
  $('modalCloseBtn')?.addEventListener('click', closeModal);
  $('shapModalBackdrop')?.addEventListener('click', (e) => {
    if (e.target === $('shapModalBackdrop')) closeModal();
  });

  // Stability Explorer Filters
  let searchTimeout = null;
  $('stabilitySearchInput')?.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      state.stability.search = e.target.value;
      state.stability.page = 1;
      loadStabilityExplorer();
    }, 300);
  });

  document.querySelectorAll('.tier-pill[data-tier]').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.tier-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.stability.tier = pill.dataset.tier;
      state.stability.page = 1;
      loadStabilityExplorer();
    });
  });

  $('shortlistOnlyCheckbox')?.addEventListener('change', (e) => {
    state.stability.shortlistedOnly = e.target.checked;
    state.stability.page = 1;
    loadStabilityExplorer();
  });
}

// ================= INIT =================

async function init() {
  buildSensorInputsForm();
  initUploadZone();
  bindEvents();
  updateClock();
  setInterval(updateClock, 1000);

  // Check health and initial dashboard load
  await checkSystemHealth();
  setInterval(checkSystemHealth, 30000);

  loadDashboard();
}

document.addEventListener('DOMContentLoaded', init);
