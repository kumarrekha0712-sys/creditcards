// script.js
// Connects the threshold slider + dashboard cards to the FastAPI backend.

const API_BASE_URL = window.location.origin;

const slider = document.getElementById("threshold-slider");
const thresholdDisplay = document.getElementById("threshold-display");
const errorBanner = document.getElementById("error-banner");

let debounceTimer = null;

function showError(message) {
  errorBanner.textContent = `⚠️ ${message}`;
  errorBanner.classList.remove("hidden");
}

function clearError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function updateDashboard(data) {
  document.getElementById("threshold-display").textContent = Number(data.threshold).toFixed(2);

  document.getElementById("metric-accuracy").textContent = formatPercent(data.accuracy);
  document.getElementById("metric-recall").textContent = formatPercent(data.recall);
  document.getElementById("metric-precision").textContent = formatPercent(data.precision);
  document.getElementById("metric-f1").textContent = formatPercent(data.f1_score);

  document.getElementById("cell-tn").textContent = data.true_negative;
  document.getElementById("cell-fp").textContent = data.false_positive;
  document.getElementById("cell-fn").textContent = data.false_negative;
  document.getElementById("cell-tp").textContent = data.true_positive;

  document.getElementById("count-tn").textContent = data.true_negative;
  document.getElementById("count-fp").textContent = data.false_positive;
  document.getElementById("count-fn").textContent = data.false_negative;
  document.getElementById("count-tp").textContent = data.true_positive;

  document.getElementById("total-count").textContent = data.total_test_transactions ?? "--";
  document.getElementById("total-fraud").textContent = data.total_actual_fraud ?? "--";

  // Bars: scale relative to the largest of the four counts so the chart stays legible
  const counts = {
    tn: data.true_negative,
    fp: data.false_positive,
    fn: data.false_negative,
    tp: data.true_positive,
  };
  const maxCount = Math.max(1, ...Object.values(counts));
  for (const key of Object.keys(counts)) {
    const bar = document.getElementById(`bar-${key}`);
    const pct = (counts[key] / maxCount) * 100;
    bar.style.width = `${pct}%`;
  }
}

async function fetchEvaluation(threshold) {
  try {
    const response = await fetch(`${API_BASE_URL}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${response.status}`);
    }

    const data = await response.json();
    clearError();
    updateDashboard(data);
  } catch (err) {
    showError(
      `Could not reach the fraud detection API (${err.message}). ` +
      `Make sure the backend is running at ${API_BASE_URL} (python -m uvicorn main:app --reload --port 8000).`
    );
  }
}

async function loadInitialMetrics() {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${response.status}`);
    }
    const data = await response.json();
    clearError();
    updateDashboard(data);
  } catch (err) {
    showError(
      `Could not reach the fraud detection API (${err.message}). ` +
      `Make sure the backend is running at ${API_BASE_URL} (python -m uvicorn main:app --reload --port 8000).`
    );
  }
}

slider.addEventListener("input", (event) => {
  const threshold = parseFloat(event.target.value);
  thresholdDisplay.textContent = threshold.toFixed(2);

  // Debounce so we don't hammer the API on every pixel of drag
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    fetchEvaluation(threshold);
  }, 120);
});

// Initial load
loadInitialMetrics();
