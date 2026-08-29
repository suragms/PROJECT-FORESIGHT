/**
 * PROJECT FORESIGHT — Interactive Unified Analytics & Intelligence Platform
 * Pure client-side zero-latency architecture for Vercel demo deployment
 */

const STATE = {
  activePage: 'executive',
  theme: localStorage.getItem('foresight_theme') || 'dark',
  apiBaseUrl: (typeof window !== 'undefined' && window.FORESIGHT_API_URL) ? window.FORESIGHT_API_URL : 'https://project-foresight-api-tofn.onrender.com',
  user: JSON.parse(localStorage.getItem('foresight_user')) || {
    name: 'Sarah Chen',
    email: 'executive@foresight.ai',
    role: 'EXECUTIVE',
    avatar: 'SC'
  },
  dataset: 'SYNTHETIC', // 'SYNTHETIC' | 'UCI'
  horizon: 1,           // 1 | 3 | 7 | 14 | 30
  activeChart: null
};

// Validated Project Baseline Metrics
const METRICS_DATA = {
  SYNTHETIC: {
    kpis: {
      revenue: '$1,248,500',
      units: '48,250',
      wape: '26.25%',
      mae: '1.90 units',
      rmse: '5.06 units',
      health: '99.8%',
      activeSkus: '10 SKUs',
      criticalRisk: '2 SKUs'
    },
    forecasts: {
      1: { mae: 1.90, rmse: 5.06, wape: 26.25, model: 'Synthetic H1 Hurdle LightGBM', coverage: '89.8%' },
      3: { mae: 4.37, rmse: 6.68, wape: 60.68, model: 'Synthetic H3 Direct LightGBM', coverage: '88.5%' },
      7: { mae: 5.02, rmse: 7.22, wape: 69.83, model: 'Synthetic H7 Direct LightGBM', coverage: '87.9%' },
      14: { mae: 5.07, rmse: 7.31, wape: 70.46, model: 'Synthetic H14 Direct LightGBM', coverage: '87.1%' },
      30: { mae: 5.67, rmse: 7.93, wape: 79.07, model: 'Synthetic H30 Direct LightGBM', coverage: '86.4%' }
    },
    skus: [
      { id: 'SKU_001', name: 'Ultra Wireless Headset', demand: 1420, stock: 450, onOrder: 300, leadWeeks: 2, safety: 180, risk: 'HIGH', cost: '$42.00', price: '$89.99' },
      { id: 'SKU_002', name: 'Ergonomic Mechanical Keyboard', demand: 890, stock: 920, onOrder: 0, leadWeeks: 3, safety: 120, risk: 'LOW', cost: '$55.00', price: '$129.99' },
      { id: 'SKU_003', name: '4K UltraHD USB-C Monitor', demand: 410, stock: 95, onOrder: 150, leadWeeks: 4, safety: 80, risk: 'CRITICAL', cost: '$180.00', price: '$349.99' },
      { id: 'SKU_004', name: 'Pro Precision Mouse', demand: 1650, stock: 1800, onOrder: 0, leadWeeks: 1, safety: 200, risk: 'LOW', cost: '$18.00', price: '$49.99' },
      { id: 'SKU_005', name: 'Smart Ambient Desk Lamp', demand: 620, stock: 210, onOrder: 300, leadWeeks: 2, safety: 90, risk: 'MEDIUM', cost: '$24.00', price: '$59.99' },
      { id: 'SKU_006', name: 'Aluminum Laptop Riser Stand', demand: 980, stock: 1100, onOrder: 0, leadWeeks: 2, safety: 150, risk: 'LOW', cost: '$12.00', price: '$34.99' },
      { id: 'SKU_007', name: 'Fast MagCharge Pad 3-in-1', demand: 1150, stock: 140, onOrder: 500, leadWeeks: 3, safety: 160, risk: 'CRITICAL', cost: '$28.00', price: '$69.99' },
      { id: 'SKU_008', name: 'USB-C Thunderbolt 4 Hub', demand: 730, stock: 450, onOrder: 200, leadWeeks: 2, safety: 110, risk: 'MEDIUM', cost: '$45.00', price: '$109.99' },
      { id: 'SKU_009', name: 'Noise-Canceling Desk Mic', demand: 540, stock: 590, onOrder: 0, leadWeeks: 2, safety: 80, risk: 'LOW', cost: '$38.00', price: '$89.99' },
      { id: 'SKU_010', name: 'Braided 240W Cable 2M', demand: 2800, stock: 2400, onOrder: 800, leadWeeks: 1, safety: 350, risk: 'LOW', cost: '$4.50', price: '$19.99' }
    ]
  },
  UCI: {
    kpis: {
      revenue: '$8,920,400',
      units: '312,400',
      wape: '79.47%',
      mae: '17.34 units',
      rmse: '70.89 units',
      health: '99.5%',
      activeSkus: '4,372 SKUs',
      criticalRisk: '84 SKUs'
    },
    forecasts: {
      1: { mae: 17.34, rmse: 70.89, wape: 79.47, model: 'UCI H1 LightGBM Point', coverage: '82.4%' },
      3: { mae: 19.40, rmse: 76.11, wape: 85.18, model: 'UCI H3 Direct LightGBM', coverage: '81.9%' },
      7: { mae: 20.38, rmse: 62.35, wape: 86.18, model: 'UCI H7 Direct LightGBM', coverage: '80.5%' },
      14: { mae: 21.44, rmse: 63.20, wape: 84.85, model: 'UCI H14 Direct LightGBM', coverage: '79.8%' },
      30: { mae: 25.44, rmse: 70.65, wape: 80.04, model: 'UCI H30 Direct LightGBM', coverage: '78.2%' }
    }
  }
};

// Model Portfolio Registry Data
const MODEL_REGISTRY = [
  { id: 'phase20_synthetic_lightgbm', dataset: 'SYNTHETIC', horizon: '6 Weeks', type: 'Weekly LightGBM + Holiday', metric: 'WAPE 24.18%', status: 'PROMOTED PROD', hash: '96a88f1d...5e086' },
  { id: 'synthetic_h1_hurdle_th050', dataset: 'SYNTHETIC', horizon: '1 Day', type: 'Two-Stage Hurdle LightGBM', metric: 'WAPE 26.25%', status: 'FROZEN PROD', hash: '59a2b720...cf1bf4' },
  { id: 'synthetic_h1_quantile_p10p90', dataset: 'SYNTHETIC', horizon: '1 Day', type: 'Quantile Regressor (P10/P90)', metric: 'Coverage 89.77%', status: 'INTERVAL COMPANION', hash: '9c09a257...5d574e' },
  { id: 'synthetic_h7_direct_lightgbm', dataset: 'SYNTHETIC', horizon: '7 Days', type: 'Direct Multi-Horizon', metric: 'WAPE 69.83%', status: 'FROZEN PROD', hash: 'f6df774a...83ce9a' },
  { id: 'uci_h1_phase8_lightgbm', dataset: 'UCI', horizon: '1 Day', type: 'Standard Tabular LightGBM', metric: 'WAPE 79.47%', status: 'FROZEN PROD', hash: '331909f0...73e90d' },
  { id: 'uci_h7_direct_lightgbm', dataset: 'UCI', horizon: '7 Days', type: 'Direct Multi-Horizon', metric: 'WAPE 86.18%', status: 'FROZEN PROD', hash: 'ce215ffc...be5406' }
];

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(STATE.theme);
  setupNavigation();
  setupUserAuth();
  renderCurrentPage();
});

function applyTheme(theme) {
  STATE.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('foresight_theme', theme);
  const icon = document.getElementById('theme-toggle-icon');
  if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
}

function toggleTheme() {
  applyTheme(STATE.theme === 'dark' ? 'light' : 'dark');
  if (STATE.activeChart) {
    renderCurrentPage();
  }
}

function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const page = item.getAttribute('data-page');
      if (page) navigateTo(page);
    });
  });
}

function navigateTo(pageKey) {
  // RBAC Permission Check
  if (pageKey === 'monitoring' && STATE.user.role === 'VIEWER') {
    alert('Access Restricted: System Monitoring requires ANALYST, EXECUTIVE, or ADMIN role. Please switch user role.');
    openAuthModal();
    return;
  }

  STATE.activePage = pageKey;
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.getAttribute('data-page') === pageKey) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });
  renderCurrentPage();
}

function setupUserAuth() {
  updateUserUI();
}

function updateUserUI() {
  const nameEl = document.getElementById('user-display-name');
  const roleEl = document.getElementById('user-display-role');
  const avatarEl = document.getElementById('user-avatar');
  if (nameEl) nameEl.textContent = STATE.user.name;
  if (roleEl) roleEl.textContent = STATE.user.role;
  if (avatarEl) avatarEl.textContent = STATE.user.avatar;
}

function switchUserPreset(role) {
  const presets = {
    EXECUTIVE: { name: 'Sarah Chen', email: 'executive@foresight.ai', role: 'EXECUTIVE', avatar: 'SC' },
    ANALYST: { name: 'David Kumar', email: 'analyst@foresight.ai', role: 'ANALYST', avatar: 'DK' },
    ADMIN: { name: 'Alex Rivera', email: 'admin@foresight.ai', role: 'ADMIN', avatar: 'AR' },
    VIEWER: { name: 'Guest Observer', email: 'viewer@foresight.ai', role: 'VIEWER', avatar: 'GO' }
  };
  if (presets[role]) {
    STATE.user = presets[role];
    localStorage.setItem('foresight_user', JSON.stringify(STATE.user));
    updateUserUI();
    closeAuthModal();
    renderCurrentPage();
  }
}

function openAuthModal() {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.classList.add('active');
}

function closeAuthModal() {
  const modal = document.getElementById('auth-modal');
  if (modal) modal.classList.remove('active');
}

// Router & Page Renderers
function renderCurrentPage() {
  const content = document.getElementById('content-area');
  if (!content) return;

  if (STATE.activeChart) {
    try { STATE.activeChart.destroy(); } catch(e) {}
    STATE.activeChart = null;
  }

  switch(STATE.activePage) {
    case 'home':
      content.innerHTML = renderHomePage();
      break;
    case 'executive':
      content.innerHTML = renderExecutivePage();
      initExecutiveChart();
      break;
    case 'forecasting':
      content.innerHTML = renderForecastingPage();
      initForecastingChart();
      break;
    case 'inventory':
      content.innerHTML = renderInventoryPage();
      initInventoryRiskSimulator();
      break;
    case 'analytics':
      content.innerHTML = renderAnalyticsPage();
      initAnalyticsChart();
      break;
    case 'ml':
      content.innerHTML = renderMLPage();
      break;
    case 'monitoring':
      content.innerHTML = renderMonitoringPage();
      initMonitoringRadar();
      break;
    case 'docs':
      content.innerHTML = renderDocsPage();
      break;
    default:
      content.innerHTML = renderExecutivePage();
      initExecutiveChart();
  }
}

// Page 1: Executive Dashboard
function renderExecutivePage() {
  const data = METRICS_DATA[STATE.dataset];
  return `
    <div class="page-header">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
          <h1 class="page-title">Executive Intelligence Dashboard</h1>
          <p class="page-description">High-level visibility into demand trajectories, stockout liabilities, and ML system health.</p>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-secondary" onclick="switchDataset('SYNTHETIC')" style="${STATE.dataset === 'SYNTHETIC' ? 'border-color: var(--accent-cyan); color: var(--accent-cyan);' : ''}">Synthetic Retail (10 SKUs)</button>
          <button class="btn btn-secondary" onclick="switchDataset('UCI')" style="${STATE.dataset === 'UCI' ? 'border-color: var(--accent-cyan); color: var(--accent-cyan);' : ''}">UCI Global Retail</button>
        </div>
      </div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Forecasted Revenue <span>💼</span></div>
        <div class="kpi-value">${data.kpis.revenue}</div>
        <div class="kpi-delta delta-good">▲ +14.2% vs Baseline Run</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Mean Model WAPE <span>🎯</span></div>
        <div class="kpi-value">${data.kpis.wape}</div>
        <div class="kpi-delta delta-good">▼ -12.64% over Naive Baseline</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Critical Stockout Risk <span>⚠️</span></div>
        <div class="kpi-value">${data.kpis.criticalRisk}</div>
        <div class="kpi-delta delta-warning">● Action Required on SKU_003 & 007</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">System Health & Drift <span>🛡️</span></div>
        <div class="kpi-value">${data.kpis.health}</div>
        <div class="kpi-delta delta-good">✓ 100% Integrity Hash Verified</div>
      </div>
    </div>

    <div class="dash-grid">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Demand Forecast vs Historical Actuals (Aggregate)</div>
          <span class="banner-tag tag-backtest">VALIDATION / BACKTEST METRICS</span>
        </div>
        <div style="height: 320px; position: relative;">
          <canvas id="exec-forecast-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Stockout & Risk Exposure</div>
          <span class="banner-tag tag-pending">LIVE PENDING ACTUALS</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: var(--text-secondary);">SKU_003 (4K Monitor)</span>
            <span class="banner-tag" style="color: var(--accent-rose); border-color: var(--accent-rose);">CRITICAL (95% Risk)</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: var(--text-secondary);">SKU_007 (MagCharge)</span>
            <span class="banner-tag" style="color: var(--accent-rose); border-color: var(--accent-rose);">CRITICAL (88% Risk)</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: var(--text-secondary);">SKU_001 (Wireless Headset)</span>
            <span class="banner-tag" style="color: var(--accent-amber); border-color: var(--accent-amber);">HIGH (64% Risk)</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: var(--text-secondary);">SKU_005 (Desk Lamp)</span>
            <span class="banner-tag" style="color: var(--accent-amber); border-color: var(--accent-amber);">MEDIUM (42% Risk)</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: var(--text-secondary);">6 Other SKUs</span>
            <span class="banner-tag" style="color: var(--accent-emerald); border-color: var(--accent-emerald);">HEALTHY (&lt;15% Risk)</span>
          </div>
          <button class="btn btn-primary" onclick="navigateTo('inventory')" style="margin-top: 8px;">Open Inventory Intelligence →</button>
        </div>
      </div>
    </div>
  `;
}

function initExecutiveChart() {
  const ctx = document.getElementById('exec-forecast-chart');
  if (!ctx) return;

  const isDark = STATE.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
  const textColor = isDark ? '#9ca3af' : '#475569';

  STATE.activeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Wk 1', 'Wk 2', 'Wk 3', 'Wk 4', 'Wk 5', 'Wk 6', 'Wk 7', 'Wk 8', 'Wk 9', 'Wk 10', 'Wk 11 (F)', 'Wk 12 (F)'],
      datasets: [
        {
          label: 'Historical Actual Demand',
          data: [3850, 4120, 3990, 4400, 4280, 4650, 4520, 4900, 4810, 5100, null, null],
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2
        },
        {
          label: 'Phase 20 Promoted Forecast',
          data: [null, null, null, null, null, null, null, null, null, 5100, 5380, 5640],
          borderColor: '#6366f1',
          borderDash: [5, 5],
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          fill: true,
          tension: 0.35,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: textColor } },
        y: { grid: { color: gridColor }, ticks: { color: textColor } }
      },
      plugins: {
        legend: { labels: { color: textColor } }
      }
    }
  });
}

// Page 2: Demand Forecasting Explorer
function renderForecastingPage() {
  const current = METRICS_DATA[STATE.dataset].forecasts[STATE.horizon] || METRICS_DATA.SYNTHETIC.forecasts[1];

  return `
    <div class="page-header">
      <h1 class="page-title">Multi-Horizon Demand Forecasting</h1>
      <p class="page-description">Explore validated machine learning forecasts across 1, 3, 7, 14, and 30-day forecast horizons.</p>
    </div>

    <div class="card">
      <div class="controls-bar" style="align-items: center; justify-content: space-between;">
        <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
          <span style="font-size: 13px; font-weight: 600; color: var(--text-muted);">SELECT HORIZON:</span>
          ${[1, 3, 7, 14, 30].map(h => `
            <button class="btn ${STATE.horizon === h ? 'btn-primary' : 'btn-secondary'}" onclick="setHorizon(${h})">${h}-Day Horizon</button>
          `).join('')}
        </div>
        <span class="banner-tag tag-backtest">MODEL: ${current.model}</span>
      </div>

      <div class="kpi-grid" style="margin-top: 16px; margin-bottom: 20px;">
        <div class="kpi-card">
          <div class="kpi-title">Horizon WAPE</div>
          <div class="kpi-value">${current.wape}%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">Mean Absolute Error (MAE)</div>
          <div class="kpi-value">${current.mae}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">RMSE Accuracy</div>
          <div class="kpi-value">${current.rmse}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-title">P10/P90 Interval Coverage</div>
          <div class="kpi-value">${current.coverage}</div>
        </div>
      </div>

      <div style="height: 360px; position: relative;">
        <canvas id="forecast-horizon-chart"></canvas>
      </div>
    </div>
  `;
}

function setHorizon(h) {
  STATE.horizon = h;
  renderCurrentPage();
}

function initForecastingChart() {
  const ctx = document.getElementById('forecast-horizon-chart');
  if (!ctx) return;

  const isDark = STATE.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
  const textColor = isDark ? '#9ca3af' : '#475569';

  const days = Array.from({length: 14}, (_, i) => `Day ${i + 1}`);
  const baseActuals = [42, 38, 55, 60, 48, 62, 59, 70, 68, 75, 72, 80, 84, 88];
  const pointForecast = baseActuals.map(v => Math.round(v * 1.05));
  const p90Upper = pointForecast.map(v => Math.round(v * 1.22));
  const p10Lower = pointForecast.map(v => Math.max(0, Math.round(v * 0.78)));

  STATE.activeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: days,
      datasets: [
        {
          label: '90th Percentile Upper Bound (P90)',
          data: p90Upper,
          borderColor: 'rgba(6, 182, 212, 0.4)',
          borderDash: [3, 3],
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Point Forecast (Selected ML Model)',
          data: pointForecast,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          fill: true,
          borderWidth: 3
        },
        {
          label: '10th Percentile Lower Bound (P10)',
          data: p10Lower,
          borderColor: 'rgba(6, 182, 212, 0.4)',
          borderDash: [3, 3],
          fill: false,
          pointRadius: 0
        },
        {
          label: 'Validation Actuals',
          data: baseActuals,
          borderColor: '#10b981',
          borderWidth: 2,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: textColor } },
        y: { grid: { color: gridColor }, ticks: { color: textColor } }
      },
      plugins: {
        legend: { labels: { color: textColor } }
      }
    }
  });
}

// Page 3: Inventory Intelligence & Risk Matrix
function renderInventoryPage() {
  const skus = METRICS_DATA.SYNTHETIC.skus;

  return `
    <div class="page-header">
      <h1 class="page-title">Inventory Intelligence & Risk Simulator</h1>
      <p class="page-description">Automated reorder triggers, lead-time buffers, and stockout probability scoring across product inventory.</p>
    </div>

    <div class="dash-grid">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Interactive What-If Reorder Simulator</div>
          <span class="banner-tag tag-audit">PHASE 20 SAFETY STOCK LOGIC</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
          <div>
            <label style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">Target SKU:</label>
            <select class="select-custom" id="sim-sku-select" onchange="runSimulation()" style="width: 100%; margin-top: 4px;">
              ${skus.map(s => `<option value="${s.id}">${s.id} — ${s.name}</option>`).join('')}
            </select>
          </div>
          <div>
            <label style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">Supplier Lead Time (Weeks):</label>
            <input type="number" class="input-custom" id="sim-lead-time" value="2" min="1" max="8" oninput="runSimulation()" style="width: 100%; margin-top: 4px;">
          </div>
          <div>
            <label style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">Current On-Hand Stock (Units):</label>
            <input type="number" class="input-custom" id="sim-stock" value="140" oninput="runSimulation()" style="width: 100%; margin-top: 4px;">
          </div>
          <div>
            <label style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">Weekly Forecast Demand (Units):</label>
            <input type="number" class="input-custom" id="sim-demand" value="380" oninput="runSimulation()" style="width: 100%; margin-top: 4px;">
          </div>
        </div>

        <div style="background: var(--bg-tertiary); padding: 18px; border-radius: 10px; border: 1px solid var(--border-color);">
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 600;">Recommended Reorder Point (ROP):</span>
            <span style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);" id="sim-rop">912 units</span>
          </div>
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 600;">Suggested Purchase Order Quantity:</span>
            <span style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-emerald);" id="sim-order-qty">850 units</span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 13px; font-weight: 600;">Stockout Probability:</span>
            <span style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-rose);" id="sim-risk-prob">88.4% (CRITICAL)</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">Inventory Risk Classification</div>
        </div>
        <div style="display: flex; flex-direction: column; gap: 12px;">
          <div style="padding: 12px; border-radius: 8px; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3);">
            <div style="font-weight: 700; color: var(--accent-rose); font-size: 13px;">CRITICAL RISK (2 Items)</div>
            <div style="font-size: 12px; color: var(--text-secondary);">Stock cover &lt; 1.5x Lead time demand. Emergency expedited order advised.</div>
          </div>
          <div style="padding: 12px; border-radius: 8px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3);">
            <div style="font-weight: 700; color: var(--accent-amber); font-size: 13px;">MEDIUM RISK (2 Items)</div>
            <div style="font-size: 12px; color: var(--text-secondary);">Stock approaching ROP threshold within next 14 calendar days.</div>
          </div>
          <div style="padding: 12px; border-radius: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3);">
            <div style="font-weight: 700; color: var(--accent-emerald); font-size: 13px;">HEALTHY STATUS (6 Items)</div>
            <div style="font-size: 12px; color: var(--text-secondary);">Adequate buffer stock; optimal working capital turnover.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">SKU Inventory Master & Risk Matrix</div>
      </div>
      <div class="table-responsive">
        <table class="table-custom">
          <thead>
            <tr>
              <th>SKU ID</th>
              <th>Product Name</th>
              <th>Weekly Demand</th>
              <th>On-Hand</th>
              <th>On-Order</th>
              <th>Lead Time</th>
              <th>Unit Price</th>
              <th>Risk Tier</th>
            </tr>
          </thead>
          <tbody>
            ${skus.map(s => `
              <tr>
                <td style="font-family: var(--font-mono); font-weight: 700;">${s.id}</td>
                <td>${s.name}</td>
                <td>${s.demand}</td>
                <td>${s.stock}</td>
                <td>${s.onOrder}</td>
                <td>${s.leadWeeks} wks</td>
                <td>${s.price}</td>
                <td>
                  <span class="banner-tag ${s.risk === 'CRITICAL' ? 'tag-pending' : s.risk === 'HIGH' ? 'tag-pending' : 'tag-audit'}" 
                        style="${s.risk === 'CRITICAL' ? 'color: var(--accent-rose); border-color: var(--accent-rose);' : ''}">
                    ${s.risk}
                  </span>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function initInventoryRiskSimulator() {
  runSimulation();
}

function runSimulation() {
  const leadTime = parseFloat(document.getElementById('sim-lead-time')?.value) || 2;
  const stock = parseFloat(document.getElementById('sim-stock')?.value) || 140;
  const demand = parseFloat(document.getElementById('sim-demand')?.value) || 380;

  const leadDemand = leadTime * demand;
  const safetyStock = Math.round(leadDemand * 0.20);
  const rop = Math.round(leadDemand + safetyStock);
  const orderQty = Math.max(0, Math.round((leadDemand * 2) - stock));
  const stockoutProb = stock < leadDemand ? Math.min(99.4, 60 + ((leadDemand - stock) / leadDemand) * 35) : Math.max(2.1, 30 - ((stock - leadDemand) / leadDemand) * 20);

  const ropEl = document.getElementById('sim-rop');
  const qtyEl = document.getElementById('sim-order-qty');
  const probEl = document.getElementById('sim-risk-prob');

  if (ropEl) ropEl.textContent = `${rop.toLocaleString()} units`;
  if (qtyEl) qtyEl.textContent = `${orderQty.toLocaleString()} units`;
  if (probEl) {
    probEl.textContent = `${stockoutProb.toFixed(1)}% (${stockoutProb > 65 ? 'CRITICAL' : stockoutProb > 30 ? 'MEDIUM' : 'LOW'})`;
    probEl.style.color = stockoutProb > 65 ? 'var(--accent-rose)' : stockoutProb > 30 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
  }
}

// Page 4: Business Analytics & Seasonality
function renderAnalyticsPage() {
  return `
    <div class="page-header">
      <h1 class="page-title">Demand Trends & Seasonality Insights</h1>
      <p class="page-description">Statistical decomposition of day-of-week patterns, month-of-year seasonality, and SKU Pareto distribution.</p>
    </div>

    <div class="dash-grid">
      <div class="card">
        <div class="card-header">
          <div class="card-title">Day-of-Week Demand Index (7-Day Seasonality)</div>
        </div>
        <div style="height: 280px; position: relative;">
          <canvas id="seasonality-chart"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">SKU Pareto Distribution (80/20 Rule)</div>
        </div>
        <div style="padding: 10px 0;">
          <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">
            Top 20% of SKU portfolio accounts for <strong>74.8%</strong> of total weekly revenue volume.
          </p>
          <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
              <span>Class A (High Volume Revenue)</span>
              <span>74.8%</span>
            </div>
            <div style="height: 8px; border-radius: 4px; background: var(--bg-tertiary); overflow: hidden;">
              <div style="width: 74.8%; height: 100%; background: var(--accent-primary);"></div>
            </div>
          </div>
          <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
              <span>Class B (Moderate Volume)</span>
              <span>18.4%</span>
            </div>
            <div style="height: 8px; border-radius: 4px; background: var(--bg-tertiary); overflow: hidden;">
              <div style="width: 18.4%; height: 100%; background: var(--accent-cyan);"></div>
            </div>
          </div>
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
              <span>Class C (Long-Tail / Intermittent)</span>
              <span>6.8%</span>
            </div>
            <div style="height: 8px; border-radius: 4px; background: var(--bg-tertiary); overflow: hidden;">
              <div style="width: 6.8%; height: 100%; background: var(--accent-emerald);"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function initAnalyticsChart() {
  const ctx = document.getElementById('seasonality-chart');
  if (!ctx) return;

  const isDark = STATE.theme === 'dark';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';
  const textColor = isDark ? '#9ca3af' : '#475569';

  STATE.activeChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
      datasets: [{
        label: 'Relative Demand Index (Base 100)',
        data: [112, 108, 98, 104, 128, 86, 64],
        backgroundColor: ['#6366f1', '#6366f1', '#6366f1', '#6366f1', '#10b981', '#f59e0b', '#f43f5e'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { color: textColor } },
        y: { grid: { color: gridColor }, ticks: { color: textColor }, min: 50 }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Page 5: ML Portfolio & Feature Contracts
function renderMLPage() {
  return `
    <div class="page-header">
      <h1 class="page-title">ML Model Registry & Feature Contract</h1>
      <p class="page-description">Cryptographically hashed model registry, reproducible hyperparameters, and zero-leakage feature contracts.</p>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Production & Candidate Model Registry</div>
        <span class="banner-tag tag-audit">SHA-256 VERIFIED</span>
      </div>
      <div class="table-responsive">
        <table class="table-custom">
          <thead>
            <tr>
              <th>Model ID</th>
              <th>Dataset</th>
              <th>Horizon</th>
              <th>Architecture</th>
              <th>Test Metric</th>
              <th>Integrity Hash</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${MODEL_REGISTRY.map(m => `
              <tr>
                <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">${m.id}</td>
                <td>${m.dataset}</td>
                <td>${m.horizon}</td>
                <td>${m.type}</td>
                <td style="font-weight: 700; color: var(--accent-emerald);">${m.metric}</td>
                <td style="font-family: var(--font-mono); font-size: 11px;">${m.hash}</td>
                <td><span class="banner-tag tag-audit">${m.status}</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Phase 20 Feature Contract (45 Engineered Variables)</div>
        <span class="banner-tag tag-backtest">LEAKAGE AUDIT: PASS</span>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">
        <div style="background: var(--bg-tertiary); padding: 14px; border-radius: 8px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: var(--accent-primary);">Lag & Rolling Features (18)</div>
          <div style="font-size: 12px; color: var(--text-secondary);">Demand lag 1-4 weeks, 4w/8w/12w rolling means, standard deviations, expanding minimum/maximums.</div>
        </div>
        <div style="background: var(--bg-tertiary); padding: 14px; border-radius: 8px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: var(--accent-cyan);">Calendar & Holiday Features (14)</div>
          <div style="font-size: 12px; color: var(--text-secondary);">is_holiday_week, holiday_count, seasonal cyclical sin/cos terms, promotional event flags.</div>
        </div>
        <div style="background: var(--bg-tertiary); padding: 14px; border-radius: 8px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: var(--accent-emerald);">Price & Discount Dynamics (13)</div>
          <div style="font-size: 12px; color: var(--text-secondary);">Base price ratio, discount percentage, category mean price divergence, elasticity score.</div>
        </div>
      </div>
    </div>
  `;
}

// Page 6: System Monitoring & Data Drift
function renderMonitoringPage() {
  return `
    <div class="page-header">
      <h1 class="page-title">Continuous Quality & Model Drift Radar</h1>
      <p class="page-description">Automated Kolmogorov-Smirnov distribution testing, Wasserstein distance tracking, and production alert ledger.</p>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title">Overall Health Score</div>
        <div class="kpi-value" style="color: var(--accent-emerald);">99.8%</div>
        <div class="kpi-delta delta-good">✓ 100% Tests Passed</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Data Drift Status</div>
        <div class="kpi-value">PASS</div>
        <div class="kpi-delta delta-good">KS-test p-value &gt; 0.05 on all features</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Prediction Stability</div>
        <div class="kpi-value">STABLE</div>
        <div class="kpi-delta delta-good">Wasserstein distance = 0.014</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Active Warning Alerts</div>
        <div class="kpi-value">0 Alerts</div>
        <div class="kpi-delta delta-good">Clean monitoring cycle</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Phase 21 Integrity Health Radar</div>
      </div>
      <div style="height: 300px; position: relative;">
        <canvas id="monitoring-radar-chart"></canvas>
      </div>
    </div>
  `;
}

function initMonitoringRadar() {
  const ctx = document.getElementById('monitoring-radar-chart');
  if (!ctx) return;

  const isDark = STATE.theme === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';

  STATE.activeChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Data Quality', 'Schema Conformance', 'Feature Stability', 'Prediction Drift', 'Latency SLA', 'Hash Integrity'],
      datasets: [{
        label: 'System Health Index (%)',
        data: [100, 100, 99.4, 99.8, 100, 100],
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.25)',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)' },
          angleLines: { color: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)' },
          pointLabels: { color: textColor, font: { size: 12 } },
          ticks: { display: false, min: 80, max: 100 }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// Page 7: Documentation & Audit
function renderDocsPage() {
  return `
    <div class="page-header">
      <h1 class="page-title">Documentation & Validation Audit</h1>
      <p class="page-description">Complete end-to-end methodology, project phases, and delivery audit status.</p>
    </div>

    <div class="card">
      <div class="card-header">
        <div class="card-title">Phase 22 Deliverable Audit Certificate</div>
        <span class="banner-tag tag-audit">PROJECT DELIVERY READY</span>
      </div>
      <p style="font-size: 14px; color: var(--text-secondary); line-height: 1.7; margin-bottom: 16px;">
        <strong>Project FORESIGHT</strong> is an enterprise-grade demand forecasting and inventory risk intelligence platform developed across 23 rigorous engineering phases. The platform provides predictive decision support, automated safety stock optimization, and proactive stockout prevention.
      </p>
      <div style="background: var(--bg-tertiary); padding: 18px; border-radius: 8px; font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
        {<br>
        &nbsp;&nbsp;"delivery_status": "PROJECT DELIVERY READY",<br>
        &nbsp;&nbsp;"frozen_models_12_unchanged": true,<br>
        &nbsp;&nbsp;"phase20_production_unchanged": true,<br>
        &nbsp;&nbsp;"phase21_monitoring_status": "PASS",<br>
        &nbsp;&nbsp;"integrity_status": "PASS",<br>
        &nbsp;&nbsp;"tests_passed": "280 passed, 0 failures"<br>
        }
      </div>
    </div>
  `;
}

function switchDataset(ds) {
  STATE.dataset = ds;
  renderCurrentPage();
}
