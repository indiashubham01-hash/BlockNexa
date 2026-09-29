/**
 * BlockNexa Frontend — Backend API Client Service
 * Connects React UI to FastAPI server on http://127.0.0.1:8000
 */

const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL)
  || (typeof window !== 'undefined'
    ? (window.location.port === '8000' || window.location.port === '5173' ? '' : 'http://127.0.0.1:8000')
    : 'http://127.0.0.1:8000');

/**
 * Check backend connection and model availability
 */
export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Backend health check failed:', err.message);
    return { status: 'offline', error: err.message };
  }
}

/**
 * Predict defect failure/deferral risk using RandomForestClassifier
 */
export async function predictDefectRisk(payload) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ml/predict-risk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Predict risk failed, falling back:', err.message);
    const score = Math.min(100, Math.round(
      0.35 * (payload.severity * 10) +
      0.30 * (payload.safety_criticality * 10) +
      0.20 * Math.min(100, (payload.overdue_days / 14) * 100) +
      0.15 * Math.min(100, (payload.rail_wear_mm / 5.0) * 100)
    ));
    return {
      predicted_risk_prob: score / 100,
      predicted_risk_percent: score,
      priority_class: score >= 75 ? 'CRITICAL' : score >= 55 ? 'HIGH' : score >= 35 ? 'MEDIUM' : 'ROUTINE',
      is_emergency_bypass: score >= 80,
      recommended_action: score >= 75 ? 'Emergency Corridor Intervention' : 'Scheduled Megablock',
      top_risk_factors: [{ factor: 'Fallback Calculation (Server Booting)', contribution: `${score}%` }],
      isFallback: true,
    };
  }
}

/**
 * Predict train delay minutes using GradientBoostingRegressor
 */
export async function predictTrainDelay(payload) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ml/predict-delay`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Predict delay failed, falling back:', err.message);
    const delay = Math.max(1.0, Math.round(
      (payload.block_duration_hours * 6.0) * (payload.traffic_intensity || 1.0)
      - ((payload.activities_bundled || 1) - 1) * 3.0
      - (payload.has_loop_reroute ? 8.0 : 0)
    ));
    return {
      predicted_delay_minutes: delay,
      mitigation_strategy: payload.has_loop_reroute ? 'Loop Line Bypass' : 'Standard Possession',
      mitigation_options: [
        { strategy: 'Standalone Possession', delay_minutes: delay * 2 },
        { strategy: 'Joint Megablock', delay_minutes: delay },
      ],
      isFallback: true,
    };
  }
}

/**
 * Fetch trained model evaluation metrics
 */
export async function fetchModelMetrics() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/ml/metrics`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Fetch metrics failed:', err.message);
    return null;
  }
}

/**
 * Run combinatorial block optimization
 */
export async function solveBlockPlan(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/planner/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Solve block plan failed:', err.message);
    return null;
  }
}

/**
 * Query Indian Railways train timetable by station code from Trains_Schedule_CLEANED.csv
 */
export async function fetchStationTrains(stationCode = 'BSL') {
  try {
    const res = await fetch(`${API_BASE_URL}/api/timetable/trains?station_code=${encodeURIComponent(stationCode)}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Fetch station trains failed:', err.message);
    return null;
  }
}

/**
 * Fetch historical maintenance block records
 */
export async function fetchBlockHistory() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/history/blocks`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Fetch block history failed:', err.message);
    return [];
  }
}

/**
 * Fetch real-time GPS telemetry and moving train statuses along Bhusawal Division
 */
export async function fetchLiveTrains() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/live-trains`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] Fetch live trains failed:', err.message);
    return null;
  }
}

// =====================================================================
// Complete Mathematical Formulation & Cutting-Edge Value-Adds API
// =====================================================================

/**
 * 1. Composite Priority Index (CPI) Formulation
 */
export async function calculateCPI(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/cpi-calculate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] calculateCPI failed, fallback:', err.message);
    const s = payload.severity ?? 0.85;
    const od = payload.overdue_days ?? 14;
    const ca = payload.asset_criticality ?? 1.0;
    const gmt = payload.gmt ?? 420;
    const w1 = 0.35, w2 = 0.25, w3 = 0.25, w4 = 0.15;
    const overdueTerm = 1 - Math.exp(-od / 7);
    const score = w1 * s + w2 * overdueTerm + w3 * ca + w4 * (gmt / 650);
    return {
      cpi_score: Math.round(score * 1000) / 1000,
      cpi_percent: Math.round(score * 1000) / 10,
      priority_class: score >= 0.75 ? 'CRITICAL' : score >= 0.55 ? 'HIGH' : 'MEDIUM',
      components: {
        severity_term: w1 * s,
        overdue_term: w2 * overdueTerm,
        asset_criticality_term: w3 * ca,
        tonnage_term: w4 * (gmt / 650)
      },
      weights: { w1_severity: w1, w2_overdue: w2, w3_criticality: w3, w4_gmt: w4 }
    };
  }
}

/**
 * 1A. Non-Linear Asset Degradation & Weibull / TQI Kinetics
 */
export async function fetchDegradationTrajectory(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/degradation-trajectory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] fetchDegradationTrajectory failed:', err.message);
    return null;
  }
}

/**
 * 2. Multi-Objective Mixed-Integer Linear Program (MILP) Solver
 */
export async function solveMILP(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/milp-solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] solveMILP failed:', err.message);
    return null;
  }
}

/**
 * Feature 1: Dynamic Single Line Working (SLW) Bi-Directional Simulator
 */
export async function simulateSLW(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/slw-simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] simulateSLW failed:', err.message);
    return null;
  }
}

/**
 * Feature 2: Counterfactual XAI Controller Decision Cards (SHAP & Counterfactuals)
 */
export async function fetchXAIExplanation(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/xai-explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] fetchXAIExplanation failed:', err.message);
    return null;
  }
}

/**
 * Feature 3: Green Traction Energy & Carbon Minimization (ESG Optimizer)
 */
export async function optimizeESG(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/esg-optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] optimizeESG failed:', err.message);
    return null;
  }
}

/**
 * Feature 4: Digital Twin Track Deformation Forecast (Auto-TSR Imposer)
 */
export async function forecastCTMC(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/ctmc-forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] forecastCTMC failed:', err.message);
    return null;
  }
}

/**
 * Feature 5: Offline-First Edge-Mesh Digital Token System (PWI / SI Mobile Sign-off)
 */
export async function generateOfflineToken(payload = {}) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/offline-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] generateOfflineToken failed:', err.message);
    return null;
  }
}

/**
 * Feature 6: Predictive Shadow Possession Opportunism
 */
export async function fetchShadowPossessions() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/math/shadow-possessions`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[BlockNexa API] fetchShadowPossessions failed:', err.message);
    return [];
  }
}
