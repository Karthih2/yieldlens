/**
 * YieldLens — Frontend Application
 * Pure Vanilla JS SPA connecting to the FastAPI backend
 *
 * Routes:
 *   POST /predict            → Yield prediction
 *   GET  /explain/{row_id}   → SHAP explanation
 *   GET  /stability-scores   → All sensor stability data
 */

// ─── Configuration ───────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';

// The 17 shortlisted sensors the model requires (order matters for display)
const SHORTLIST_SENSORS = [
  'sensor_21', 'sensor_0',   'sensor_285', 'sensor_225', 'sensor_290',
  'sensor_71', 'sensor_59',  'sensor_571',  'sensor_103', 'sensor_406',
  'sensor_70', 'sensor_138', 'sensor_473',  'sensor_287', 'sensor_117',
  'sensor_100','sensor_95'
];

// Representative example values (realistic range ~0–1 for normalised semiconductor data)
const EXAMPLE_VALUES = {
  sensor_21:  0.4821, sensor_0:   0.3102, sensor_285: 0.5543, sensor_225: 0.2710,
  sensor_290: 0.6894, sensor_71:  0.5012, sensor_59:  0.7823, sensor_571: 0.4400,
  sensor_103: 0.3267, sensor_406: 0.4900, sensor_70:  0.5188, sensor_138: 0.4001,
  sensor_473: 0.3750, sensor_287: 0.5900, sensor_117: 0.4215, sensor_100: 0.2800,
  sensor_95:  0.1500
};

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  stabilityData:      [],      // All sensors from /stability-scores
  currentView:        'dashboard',
  lastPrediction:     null,    // { fail_probability, predicted_label, row_id }
  gaugeChart:         null,
  importanceChart:    null,
  scatterChart:       null,
  shapBarChart:       null,
  apiOnline:          false,

  // Stability Explorer state
  filteredSensors:    [],
  currentFilter:      'all',
  currentSearch:      '',
  currentSort:        'stability_desc',
  currentPage:        1,
  rowsPerPage:        20,
};

// ─── Utilities ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const qs = (sel, ctx = document) => ctx.querySelector(sel);

function showToast(message, type = 'info', duration = 3500) {
  const container = $('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const icons = {
    success: '✓', error: '✗', info: 'ℹ', warning: '⚠'
  };

  toast.innerHTML = `<span style="font-weight:700;font-size:15px">${icons[type] || 'ℹ'}</span> ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideInToast 0.3s ease reverse';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function showLoading(text = 'Loading…') {
  $('loadingOverlay').classList.remove('hidden');
  $('loadingText').textContent = text;
}

function hideLoading() {
  $('loadingOverlay').classList.add('hidden');
}

function formatPct(val) {
  return (val * 100).toFixed(1) + '%';
}

function formatNum(val, decimals = 4) {
  return val.toFixed(decimals);
}

function updateClock() {
  const now = new Date();
  $('topbarTime').textContent = now.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  });
}

// ─── API ──────────────────────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

async function checkApiStatus() {
  try {
    await apiFetch('/');
    state.apiOnline = true;
    setApiStatus('online', 'API Online');
  } catch {
    state.apiOnline = false;
    setApiStatus('offline', 'API Offline');
  }
}

function setApiStatus(status, text) {
  const dots = document.querySelectorAll('.status-dot');
  dots.forEach(d => {
    d.className = `status-dot ${status}`;
  });
  $('topbarStatusText').textContent = text;
}

// ─── Router ───────────────────────────────────────────────────────────────────
function navigateTo(viewName) {
  // Hide all views
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Show target view
  const view = $(`view-${viewName}`);
  if (!view) return;
  view.classList.add('active');

  // Activate nav
  const navItem = $(`nav-${viewName}`);
  if (navItem) navItem.classList.add('active');

  // Update breadcrumb
  const labels = {
    dashboard: 'Dashboard',
    predict:   'Yield Predictor',
    stability: 'Stability Explorer'
  };
  $('breadcrumbCurrent').textContent = labels[viewName] || viewName;

  state.currentView = viewName;

  // Close mobile sidebar
  closeMobileSidebar();

  // Load view data
  if (viewName === 'dashboard' && state.stabilityData.length === 0) {
    loadDashboard();
  } else if (viewName === 'dashboard') {
    // Re-render charts (they may have been destroyed)
    renderDashboardCharts();
  } else if (viewName === 'stability' && state.stabilityData.length === 0) {
    loadStabilityExplorer();
  } else if (viewName === 'stability') {
    applyFiltersAndRender();
  }
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function initSidebar() {
  // Desktop collapse
  $('sidebarToggle').addEventListener('click', () => {
    $('sidebar').classList.toggle('collapsed');
  });

  // Mobile open
  $('mobileMenuBtn').addEventListener('click', openMobileSidebar);

  // Nav links
  document.querySelectorAll('.nav-item[data-view]').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(item.dataset.view);
    });
  });
}

function openMobileSidebar() {
  $('sidebar').classList.add('mobile-open');
  let overlay = qs('.sidebar-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    overlay.addEventListener('click', closeMobileSidebar);
    document.body.appendChild(overlay);
  }
  overlay.classList.add('visible');
}

function closeMobileSidebar() {
  $('sidebar').classList.remove('mobile-open');
  const overlay = qs('.sidebar-overlay');
  if (overlay) overlay.classList.remove('visible');
}

// ─── Dashboard ────────────────────────────────────────────────────────────────
async function loadDashboard() {
  showLoading('Loading sensor data…');
  try {
    const data = await apiFetch('/stability-scores');
    state.stabilityData = data.scores;
    renderDashboardStats();
    renderDashboardCharts();
    renderShortlistTable();
  } catch (err) {
    showToast(`Failed to load stability data: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

function renderDashboardStats() {
  const scores = state.stabilityData;
  const shortlisted = scores.filter(s => s.in_shortlist);
  const highStability = scores.filter(s => s.stability_score >= 0.2);

  // Top sensor by avg_importance (from shortlisted only)
  const sorted = [...shortlisted].sort((a, b) => b.avg_importance - a.avg_importance);
  const top = sorted[0];

  $('statTotalVal').textContent = scores.length;
  $('statShortlistVal').textContent = shortlisted.length;
  $('statShortlistPct').textContent = `${((shortlisted.length / scores.length) * 100).toFixed(1)}% of all sensors`;
  $('statTopSensorVal').textContent = top ? top.sensor : '—';
  $('statTopSensorScore').textContent = top ? `Importance: ${formatNum(top.avg_importance)}` : '—';
  $('statHighStabVal').textContent = highStability.length;

  $('shortlistBadge').textContent = `${shortlisted.length} sensors`;
}

function getThemeColors() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    gridColor: isDark ? '#21262d' : '#e2e8f0',
    textColor: isDark ? '#8b949e' : '#64748b',
    tooltipBg: isDark ? '#161b22' : '#ffffff',
    tooltipBorder: isDark ? '#30363d' : '#cbd5e1',
    tooltipTitle: isDark ? '#f0f6fc' : '#0f172a',
    tooltipBody: isDark ? '#8b949e' : '#475569',
    shortlistColor: isDark ? '#3fb950' : '#059669',
    otherColor: isDark ? '#6e7681' : '#94a3b8'
  };
}

function renderDashboardCharts() {
  const scores = state.stabilityData;
  if (!scores.length) return;

  const tc = getThemeColors();

  // Destroy old charts
  if (state.importanceChart) { state.importanceChart.destroy(); state.importanceChart = null; }
  if (state.scatterChart)    { state.scatterChart.destroy();    state.scatterChart = null; }

  const shortlisted = scores.filter(s => s.in_shortlist);
  const top10 = [...shortlisted].sort((a, b) => b.avg_importance - a.avg_importance).slice(0, 10);

  // ── Bar Chart: Top 10 by importance ────────────────────────
  const barCtx = $('importanceChart').getContext('2d');
  state.importanceChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: top10.map(s => s.sensor),
      datasets: [{
        label: 'Avg Importance',
        data: top10.map(s => s.avg_importance),
        backgroundColor: top10.map(s =>
          s.in_shortlist ? tc.shortlistColor : tc.otherColor
        ),
        borderWidth: 0,
        borderRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody,
          padding: 10,
          displayColors: false,
          callbacks: {
            label: ctx => `Importance: ${formatNum(ctx.raw, 5)}`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: tc.textColor, font: { size: 10, family: 'JetBrains Mono' } },
          grid: { display: false }
        },
        y: {
          ticks: { color: tc.textColor, font: { size: 10 } },
          grid: { color: tc.gridColor },
          beginAtZero: true
        }
      }
    }
  });

  // ── Scatter Chart: Stability vs Importance ──────────────────
  const activeScores = scores.filter(s => s.avg_importance > 0);
  const shortlistedScatter = activeScores.filter(s => s.in_shortlist);
  const otherScatter = activeScores.filter(s => !s.in_shortlist);

  const scatterCtx = $('scatterChart').getContext('2d');
  state.scatterChart = new Chart(scatterCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Shortlisted',
          data: shortlistedScatter.map(s => ({ x: s.stability_score, y: s.avg_importance, sensor: s.sensor })),
          backgroundColor: tc.shortlistColor,
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: 'Not Shortlisted',
          data: otherScatter.map(s => ({ x: s.stability_score, y: s.avg_importance, sensor: s.sensor })),
          backgroundColor: tc.otherColor,
          pointRadius: 4,
          pointHoverRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: tc.textColor, font: { size: 11, family: 'Plus Jakarta Sans' }, boxWidth: 8, usePointStyle: true }
        },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody,
          padding: 10,
          callbacks: {
            label: ctx => [
              `${ctx.raw.sensor}`,
              `Stability: ${ctx.raw.x.toFixed(2)}`,
              `Importance: ${formatNum(ctx.raw.y, 5)}`
            ]
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Stability Score', color: tc.textColor, font: { size: 11 } },
          ticks: { color: tc.textColor, font: { size: 10 } },
          grid: { color: tc.gridColor },
        },
        y: {
          title: { display: true, text: 'Avg Importance', color: tc.textColor, font: { size: 11 } },
          ticks: { color: tc.textColor, font: { size: 10 } },
          grid: { color: tc.gridColor },
          beginAtZero: true
        }
      }
    }
  });
}

function renderShortlistTable() {
  const shortlisted = state.stabilityData.filter(s => s.in_shortlist);
  const sorted = [...shortlisted].sort((a, b) => b.avg_importance - a.avg_importance);
  const maxImportance = Math.max(...sorted.map(s => s.avg_importance));
  const tbody = $('shortlistTableBody');

  tbody.innerHTML = sorted.map((s, i) => {
    const tier = getStabilityTier(s.stability_score);
    const pct = maxImportance > 0 ? (s.avg_importance / maxImportance * 100).toFixed(1) : 0;
    return `
      <tr class="shortlisted">
        <td><span class="mono" style="color:#38bdf8;font-weight:600">${s.sensor}</span></td>
        <td><span class="mono">${s.stability_score.toFixed(2)}</span></td>
        <td>
          <div class="importance-bar-wrap">
            <div class="importance-bar-bg">
              <div class="importance-bar-fg" style="width:${pct}%"></div>
            </div>
            <span class="importance-val">${formatNum(s.avg_importance)}</span>
          </div>
        </td>
        <td>${renderTierBadge(tier)}</td>
        <td><span class="badge badge-green">Shortlisted</span></td>
      </tr>
    `;
  }).join('');
}

// ─── Dashboard Refresh ────────────────────────────────────────────────────────
$('dashboardRefresh').addEventListener('click', async () => {
  state.stabilityData = [];
  await loadDashboard();
  showToast('Dashboard data refreshed', 'success');
});

// ─── Predict Page ─────────────────────────────────────────────────────────────
function buildSensorForm() {
  const grid = $('sensorFormGrid');
  grid.innerHTML = SHORTLIST_SENSORS.map(sensor => `
    <div class="sensor-input-group">
      <label class="sensor-input-label" for="input-${sensor}">
        ${sensor}
        <span class="shortlist-tag">★</span>
      </label>
      <input
        id="input-${sensor}"
        class="sensor-input"
        type="number"
        step="any"
        placeholder="0.000"
        name="${sensor}"
        aria-label="Value for ${sensor}"
      />
    </div>
  `).join('');

  // Track value-filled state
  grid.querySelectorAll('.sensor-input').forEach(inp => {
    inp.addEventListener('input', () => {
      inp.classList.toggle('has-value', inp.value !== '');
    });
  });
}

function loadExampleValues() {
  SHORTLIST_SENSORS.forEach(sensor => {
    const input = $(`input-${sensor}`);
    if (input) {
      input.value = EXAMPLE_VALUES[sensor];
      input.classList.add('has-value');
    }
  });
  showToast('Example sensor values loaded', 'info');
}

function clearForm() {
  SHORTLIST_SENSORS.forEach(sensor => {
    const input = $(`input-${sensor}`);
    if (input) {
      input.value = '';
      input.classList.remove('has-value');
    }
  });
  $('resultPlaceholder').classList.remove('hidden');
  $('resultSection').classList.add('hidden');
  $('shapCard').classList.add('hidden');
  state.lastPrediction = null;
  showToast('Form cleared', 'info');
}

async function submitPrediction(e) {
  e.preventDefault();

  // Collect values
  const sensorValues = {};
  let missingFields = [];

  SHORTLIST_SENSORS.forEach(sensor => {
    const val = $(`input-${sensor}`)?.value;
    if (val === '' || val === null || val === undefined) {
      missingFields.push(sensor);
    } else {
      sensorValues[sensor] = parseFloat(val);
    }
  });

  if (missingFields.length > 0) {
    showToast(`Missing values for: ${missingFields.slice(0, 3).join(', ')}${missingFields.length > 3 ? '…' : ''}`, 'warning');
    return;
  }

  const btn = $('predictBtn');
  btn.disabled = true;
  btn.innerHTML = `
    <span style="display:inline-block;width:14px;height:14px;border:2px solid #000;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite"></span>
    Running…
  `;

  try {
    const result = await apiFetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ sensor_values: sensorValues })
    });

    state.lastPrediction = result;
    renderPredictionResult(result);
    showToast(`Prediction complete — Row ID: ${result.row_id}`, 'success');

  } catch (err) {
    showToast(`Prediction failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 1l6 4-6 4V1zM2 5h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Run Prediction
    `;
  }
}

function renderPredictionResult(result) {
  const { fail_probability, predicted_label, row_id } = result;
  const isFail = predicted_label === 1;

  // Show result section
  $('resultPlaceholder').classList.add('hidden');
  $('resultSection').classList.remove('hidden');
  $('shapCard').classList.add('hidden');

  // Badge
  $('resultRowBadge').textContent = `Row #${row_id}`;

  // Gauge
  renderGauge(fail_probability);
  $('gaugePct').textContent = formatPct(fail_probability);

  // Verdict
  const verdictEl = $('outcomeVerdict');
  const verdictIcon = $('verdictIcon');
  const verdictText = $('verdictText');

  verdictEl.className = `outcome-verdict ${isFail ? 'fail-verdict' : 'pass-verdict'}`;
  verdictIcon.textContent = isFail ? '⚠' : '✓';
  verdictText.textContent = isFail ? 'WAFER FAIL PREDICTED' : 'WAFER PASS PREDICTED';

  // Meta
  $('metaFailProb').textContent = formatPct(fail_probability);
  $('metaLabel').innerHTML = isFail
    ? `<span style="color:var(--fail);font-weight:600">Fail (1)</span>`
    : `<span style="color:var(--pass);font-weight:600">Pass (0)</span>`;
  $('metaRowId').textContent = `#${row_id}`;
}

function renderGauge(failProb) {
  if (state.gaugeChart) { state.gaugeChart.destroy(); state.gaugeChart = null; }

  const tc = getThemeColors();
  const pct = failProb;
  const passColor = tc.shortlistColor;
  const failColor = document.documentElement.getAttribute('data-theme') === 'dark' ? '#f85149' : '#e11d48';
  const color = pct >= 0.5 ? failColor : passColor;
  const bgColor = document.documentElement.getAttribute('data-theme') === 'dark' ? '#21262d' : '#f1f5f9';

  const ctx = $('gaugeChart').getContext('2d');
  state.gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [pct, 1 - pct],
        backgroundColor: [color, bgColor],
        borderWidth: 0,
        borderRadius: 2,
      }]
    },
    options: {
      cutout: '80%',
      rotation: -90,
      circumference: 180,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 600, easing: 'easeOutCubic' }
    }
  });
}

async function loadShapExplanation() {
  if (!state.lastPrediction) return;

  const btn = $('explainBtn');
  btn.disabled = true;
  btn.textContent = 'Loading…';

  try {
    const result = await apiFetch(`/explain/${state.lastPrediction.row_id}`);
    renderShapCard(result);
    showToast('SHAP explanation loaded', 'success');
  } catch (err) {
    showToast(`Explanation failed: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.3"/><path d="M7 6v4M7 4.5v.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
      Reload SHAP
    `;
  }
}

function renderShapCard(result) {
  $('shapCard').classList.remove('hidden');
  const { top_contributors } = result;
  const tc = getThemeColors();

  // ── SHAP Bar Chart ──────────────────────────────────────────
  if (state.shapBarChart) { state.shapBarChart.destroy(); state.shapBarChart = null; }

  const labels = top_contributors.map(c => c.sensor);
  const values = top_contributors.map(c => c.shap_value);
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const colors = values.map(v => v >= 0 ? (isDark ? '#f85149' : '#e11d48') : (isDark ? '#2f81f7' : '#2563eb'));

  const ctx = $('shapBarChart').getContext('2d');
  state.shapBarChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'SHAP Value',
        data: values,
        backgroundColor: colors,
        borderWidth: 0,
        borderRadius: 2,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: tc.tooltipBg,
          borderColor: tc.tooltipBorder,
          borderWidth: 1,
          titleColor: tc.tooltipTitle,
          bodyColor: tc.tooltipBody,
          padding: 8,
          displayColors: false,
          callbacks: {
            label: ctx => `SHAP: ${ctx.raw >= 0 ? '+' : ''}${ctx.raw.toFixed(5)}`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: tc.textColor, font: { size: 10 } },
          grid: { color: tc.gridColor },
          title: { display: true, text: 'SHAP Value', color: tc.textColor, font: { size: 11 } }
        },
        y: {
          ticks: { color: tc.textAccent || tc.textColor, font: { size: 11, family: 'JetBrains Mono' } },
          grid: { display: false }
        }
      }
    }
  });

  // ── SHAP Table ──────────────────────────────────────────────
  const maxAbsShap = Math.max(...values.map(Math.abs));
  const wrapper = $('shapTableWrapper');
  wrapper.innerHTML = `
    <table class="shap-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Sensor</th>
          <th>SHAP Value</th>
          <th>Direction</th>
          <th style="min-width:120px">Magnitude</th>
        </tr>
      </thead>
      <tbody>
        ${top_contributors.map((c, i) => {
          const isPos = c.shap_value >= 0;
          const barPct = maxAbsShap > 0 ? (Math.abs(c.shap_value) / maxAbsShap * 100).toFixed(1) : 0;
          return `
            <tr>
              <td style="color:#4a5568">${i + 1}</td>
              <td class="mono" style="color:#00d4ff">${c.sensor}</td>
              <td class="mono" style="color:${isPos ? '#ff4d6a' : '#818cf8'}">${isPos ? '+' : ''}${c.shap_value.toFixed(5)}</td>
              <td><span style="color:${isPos ? '#ff4d6a' : '#818cf8'};font-size:12px">${isPos ? '↑ Increases risk' : '↓ Decreases risk'}</span></td>
              <td>
                <div class="shap-bar-container">
                  <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}" style="width:${barPct}%"></div>
                </div>
              </td>
            </tr>
          `;
        }).join('')}
      </tbody>
    </table>
  `;
}

// ─── Stability Explorer ───────────────────────────────────────────────────────
async function loadStabilityExplorer() {
  showLoading('Loading all sensors…');
  try {
    if (state.stabilityData.length === 0) {
      const data = await apiFetch('/stability-scores');
      state.stabilityData = data.scores;
    }
    state.filteredSensors = [...state.stabilityData];
    applyFiltersAndRender();
  } catch (err) {
    showToast(`Failed to load sensors: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

function applyFiltersAndRender() {
  let data = [...state.stabilityData];
  const search = state.currentSearch.toLowerCase();
  const filter = state.currentFilter;
  const sort = state.currentSort;

  // Search
  if (search) {
    data = data.filter(s => s.sensor.toLowerCase().includes(search));
  }

  // Filter
  switch (filter) {
    case 'shortlisted': data = data.filter(s => s.in_shortlist); break;
    case 'high':        data = data.filter(s => s.stability_score >= 0.4); break;
    case 'medium':      data = data.filter(s => s.stability_score > 0 && s.stability_score < 0.4); break;
    case 'zero':        data = data.filter(s => s.stability_score === 0); break;
  }

  // Sort
  switch (sort) {
    case 'stability_desc': data.sort((a, b) => b.stability_score - a.stability_score); break;
    case 'stability_asc':  data.sort((a, b) => a.stability_score - b.stability_score); break;
    case 'importance_desc': data.sort((a, b) => b.avg_importance - a.avg_importance); break;
    case 'importance_asc':  data.sort((a, b) => a.avg_importance - b.avg_importance); break;
    case 'sensor_asc':
      data.sort((a, b) => {
        const numA = parseInt(a.sensor.replace('sensor_', ''));
        const numB = parseInt(b.sensor.replace('sensor_', ''));
        return numA - numB;
      });
      break;
  }

  state.filteredSensors = data;
  state.currentPage = 1;

  updateFilterStats();
  renderStabilityTable();
  renderPagination();
}

function updateFilterStats() {
  const total = state.stabilityData.length;
  const showing = state.filteredSensors.length;
  const shortlisted = state.filteredSensors.filter(s => s.in_shortlist).length;

  $('filterStats').textContent = `Showing ${showing} of ${total} sensors${shortlisted > 0 ? ` — ${shortlisted} shortlisted` : ''}`;
  $('stabilityCount').textContent = `${showing} sensors`;
}

function renderStabilityTable() {
  const start = (state.currentPage - 1) * state.rowsPerPage;
  const end   = start + state.rowsPerPage;
  const page  = state.filteredSensors.slice(start, end);
  const maxImp = Math.max(...state.filteredSensors.map(s => s.avg_importance), 0.0001);
  const tbody = $('stabilityTableBody');

  if (page.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="table-loading">No sensors match your filters</td></tr>`;
    $('tableRowCount').textContent = '0 sensors';
    return;
  }

  tbody.innerHTML = page.map((s, i) => {
    const tier = getStabilityTier(s.stability_score);
    const pct  = (s.avg_importance / maxImp * 100).toFixed(1);
    const rowClass = s.in_shortlist ? 'shortlisted' : (s.stability_score > 0 ? 'high-stability' : '');
    return `
      <tr class="${rowClass}">
        <td><span class="mono" style="color:${s.in_shortlist ? '#38bdf8' : '#f1f5f9'};font-weight:${s.in_shortlist ? '600' : '400'}">${s.sensor}</span></td>
        <td><span class="mono" style="color:${s.stability_score >= 0.4 ? '#f59e0b' : s.stability_score > 0 ? '#06b6d4' : '#64748b'}">${s.stability_score.toFixed(2)}</span></td>
        <td>
          <div class="importance-bar-wrap">
            <div class="importance-bar-bg">
              <div class="importance-bar-fg" style="width:${pct}%;background:${s.in_shortlist ? '#10b981' : '#64748b'}"></div>
            </div>
            <span class="importance-val">${s.avg_importance > 0 ? formatNum(s.avg_importance) : '—'}</span>
          </div>
        </td>
        <td>${renderTierBadge(tier)}</td>
        <td>${s.in_shortlist
            ? `<span class="badge badge-green">Shortlisted</span>`
            : `<span class="badge badge-gray">Not selected</span>`
          }
        </td>
      </tr>
    `;
  }).join('');

  const totalPages = Math.ceil(state.filteredSensors.length / state.rowsPerPage);
  $('tableRowCount').textContent = `Showing ${start + 1}–${Math.min(end, state.filteredSensors.length)} of ${state.filteredSensors.length}`;
}

function renderPagination() {
  const total = Math.ceil(state.filteredSensors.length / state.rowsPerPage);
  const current = state.currentPage;
  const pagination = $('pagination');

  if (total <= 1) { pagination.innerHTML = ''; return; }

  const pages = [];

  // Always show first, last, current ±1
  const visible = new Set([1, total, current, current - 1, current + 1].filter(p => p >= 1 && p <= total));
  const sorted = [...visible].sort((a, b) => a - b);

  const prevBtn = `<button class="page-btn" ${current === 1 ? 'disabled' : ''} id="prevPage">‹</button>`;
  const nextBtn = `<button class="page-btn" ${current === total ? 'disabled' : ''} id="nextPage">›</button>`;

  let html = prevBtn;
  let prev = 0;
  for (const p of sorted) {
    if (prev && p - prev > 1) html += `<span class="page-btn" style="cursor:default;border:none;color:#4a5568">…</span>`;
    html += `<button class="page-btn ${p === current ? 'active' : ''}" data-page="${p}">${p}</button>`;
    prev = p;
  }
  html += nextBtn;

  pagination.innerHTML = html;

  // Events
  pagination.querySelectorAll('[data-page]').forEach(btn => {
    btn.addEventListener('click', () => {
      state.currentPage = parseInt(btn.dataset.page);
      renderStabilityTable();
      renderPagination();
    });
  });

  $('prevPage')?.addEventListener('click', () => {
    if (state.currentPage > 1) { state.currentPage--; renderStabilityTable(); renderPagination(); }
  });

  $('nextPage')?.addEventListener('click', () => {
    if (state.currentPage < total) { state.currentPage++; renderStabilityTable(); renderPagination(); }
  });
}

// ─── Stability Tier Helpers ───────────────────────────────────────────────────
function getStabilityTier(score) {
  if (score >= 0.4) return 'high';
  if (score >= 0.2) return 'medium';
  if (score > 0)   return 'low';
  return 'zero';
}

function renderTierBadge(tier) {
  const config = {
    high:   { cls: 'tier-high',   label: 'High' },
    medium: { cls: 'tier-medium', label: 'Medium' },
    low:    { cls: 'tier-low',    label: 'Low' },
    zero:   { cls: 'tier-zero',   label: 'Zero' },
  };
  const { cls, label } = config[tier] || config.zero;
  return `<span class="tier-badge ${cls}">${label}</span>`;
}

// ─── Event Bindings ───────────────────────────────────────────────────────────
function bindEvents() {
  // Predict form
  $('predictForm')?.addEventListener('submit', submitPrediction);
  $('loadExampleBtn')?.addEventListener('click', loadExampleValues);
  $('clearFormBtn')?.addEventListener('click', clearForm);
  $('explainBtn')?.addEventListener('click', loadShapExplanation);

  // Stability filter pills
  document.querySelectorAll('.filter-pill[data-filter]').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.currentFilter = pill.dataset.filter;
      state.currentPage = 1;
      applyFiltersAndRender();
    });
  });

  // Search
  $('sensorSearch')?.addEventListener('input', e => {
    state.currentSearch = e.target.value;
    applyFiltersAndRender();
  });

  // Sort select
  $('sortSelect')?.addEventListener('change', e => {
    state.currentSort = e.target.value;
    applyFiltersAndRender();
  });

  // Stability table header sort
  document.querySelectorAll('#stabilityTable th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      const isDesc = state.currentSort === `${col}_desc`;
      state.currentSort = isDesc ? `${col}_asc` : `${col}_desc`;
      $('sortSelect').value = state.currentSort;

      // Update arrow indicators
      document.querySelectorAll('#stabilityTable th.sortable .sort-arrow').forEach(a => a.textContent = '↕');
      th.querySelector('.sort-arrow').textContent = isDesc ? '↑' : '↓';
      applyFiltersAndRender();
    });
  });

  // Theme toggle
  $('themeToggleBtn')?.addEventListener('click', toggleTheme);
}

// ─── Theme Management ────────────────────────────────────────────────────────
function initTheme() {
  const saved = localStorage.getItem('yieldlens_theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const next = current === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('yieldlens_theme', next);
  showToast(`Switched to ${next} theme`, 'info', 2000);

  // Re-render current view charts with updated theme colors
  if (state.currentView === 'dashboard') {
    renderDashboardCharts();
  } else if (state.currentView === 'predict' && state.lastPrediction) {
    renderGauge(state.lastPrediction.fail_probability);
  }
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function startClock() {
  updateClock();
  setInterval(updateClock, 1000);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  initTheme();
  initSidebar();
  buildSensorForm();
  bindEvents();
  startClock();

  // Check API status
  setApiStatus('checking', 'Checking API…');
  await checkApiStatus();

  // Periodically re-check
  setInterval(checkApiStatus, 30_000);

  // Load initial view
  await loadDashboard();
}

// Kick off when DOM is ready
document.addEventListener('DOMContentLoaded', init);
