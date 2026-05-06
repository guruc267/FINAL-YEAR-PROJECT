/* ═══════════════════════════════════════════════════════════
   FreshAgent – Frontend JavaScript (Enhanced)
   ═══════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;
let pollInterval = null;
let phonePollInterval = null;
let selectedFile = null;
let analysisCount = 0;
let currentMode = 'fruit';  // 'fruit' or 'veggie'
let phoneCamUrl = '';

// ── Startup ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  document.getElementById("esp32Endpoint").textContent =
    `${window.location.protocol}//${window.location.hostname}:9090/api/esp32-stream`;

  // Re-ping health every 30s
  setInterval(checkApiHealth, 30000);
});

async function checkApiHealth() {
  const dot = document.getElementById("apiStatus");
  const text = document.getElementById("apiStatusText");
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    dot.className = "status-dot online";
    text.textContent = data.model_loaded ? "Model Ready ✓" : "API Online – Fruit Model Not Loaded";
  } catch {
    dot.className = "status-dot offline";
    text.textContent = "API Offline";
  }
}

// ── Mode Switching (Fruit / Veggie) ─────────────────────────
function setMode(mode) {
  currentMode = mode;
  const isFruit = mode === 'fruit';

  // Update toggle button styles
  document.getElementById('modeBtn-fruit').classList.toggle('active', isFruit);
  document.getElementById('modeBtn-veggie').classList.toggle('active', !isFruit);

  // Update UI text
  document.getElementById('uploadCardTitle').textContent = isFruit ? '📂 Select Fruit Image' : '📂 Select Vegetable Image';
  document.getElementById('resItemLabel').textContent = isFruit ? 'Fruit Type' : 'Vegetable Type';
  document.getElementById('resConfLabel').textContent = isFruit ? 'Fruit Confidence' : 'Veggie Confidence';
  document.getElementById('loaderModel').textContent = isFruit
    ? 'FreshAgent · EfficientNetV2-B0 · ONNX Runtime'
    : 'VeggieAgent · EfficientNetV2-B0 · ONNX Runtime';

  // Show veggie-not-trained notice if needed (then hide after check)
  const notice = document.getElementById('veggieNotice');
  if (!isFruit) {
    fetch(`${API_BASE}/api/veggie-status`, { signal: AbortSignal.timeout(3000) })
      .then(r => r.json())
      .then(d => {
        notice.classList.toggle('hidden', d.veggie_model_loaded);
      })
      .catch(() => notice.classList.remove('hidden'));
  } else {
    notice.classList.add('hidden');
  }

  // Clear previous result on mode switch
  clearUpload();
}

// ── Tab Switching ────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll(".tab").forEach(t => {
    t.classList.remove("active");
    t.setAttribute("aria-selected", "false");
  });
  document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));

  document.getElementById(`tab-${tab}`).classList.add("active");
  document.getElementById(`tab-${tab}`).setAttribute("aria-selected", "true");
  document.getElementById(`panel-${tab}`).classList.remove("hidden");

  if (tab !== "esp32") stopPolling();
  if (tab !== "phone") stopPhonePolling();

  // Load QR code when switching to phone tab
  if (tab === "phone" && !phoneCamUrl) loadPhoneCamQR();
}

// ── Drag and Drop ────────────────────────────────────────────
function dragOver(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.add("drag-over");
}
function dragLeave() {
  document.getElementById("dropZone").classList.remove("drag-over");
}
function dropFile(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
}
function fileSelected(e) {
  const f = e.target.files[0];
  if (f) handleFile(f);
}

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const preview = document.getElementById("previewImg");
    preview.src = ev.target.result;
    preview.style.opacity = "0";
    setTimeout(() => { preview.style.transition = "opacity 0.4s"; preview.style.opacity = "1"; }, 10);

    document.getElementById("dropZone").classList.add("hidden");
    document.getElementById("previewWrap").classList.remove("hidden");
    document.getElementById("resultCard").style.display = "none";

    // Show file name
    const sizeKB = (file.size / 1024).toFixed(1);
    const nameEl = document.getElementById("fileName");
    if (nameEl) nameEl.textContent = `${file.name}  (${sizeKB} KB)`;
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  selectedFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("previewImg").src = "";
  document.getElementById("dropZone").classList.remove("hidden");
  document.getElementById("previewWrap").classList.add("hidden");
  document.getElementById("resultCard").style.display = "none";
}

// ── Analysis ─────────────────────────────────────────────────
async function analyzeManual() {
  if (!selectedFile) return;

  const loader = document.getElementById("uploadLoader");
  const btn = document.getElementById("analyzeBtn");
  loader.classList.remove("hidden");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner-ring" style="width:18px;height:18px;border-width:2px;display:inline-block;"></span> Analyzing…`;

  // Choose endpoint based on current mode
  const endpoint = currentMode === 'veggie' ? '/api/upload-veggie' : '/api/upload-manual';

  try {
    const form = new FormData();
    form.append("file", selectedFile);

    const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      showToast(`❌ ${data.detail || "Server error — make sure the model is loaded."}`, "danger");
      return;
    }
    analysisCount++;
    renderResult(data, "manual");

  } catch (err) {
    showToast(`⚠️ Network error: ${err.message}`, "warn");
  } finally {
    loader.classList.add("hidden");
    btn.disabled = false;
    btn.innerHTML = "🔍 Analyze Image";
  }
}

// ── Result Rendering ─────────────────────────────────────────
function renderResult(data, source) {
  if (source === "manual") {
    const card = document.getElementById("resultCard");
    card.style.display = "block";
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });

    const banner = document.getElementById("safetyBanner");
    banner.textContent = getSafetyEmoji(data.safety_class) + " " + data.safety;
    banner.className = `result-hero ${data.safety_class}`;

    // Veggie response uses 'veggie' field; fruit uses 'fruit'
    const itemName = data.veggie || data.fruit || '—';
    const itemConf = data.veggie_confidence ?? data.fruit_confidence ?? 0;

    animateValue("resFruit", itemName);
    animateValue("resCondition", data.condition_display || data.condition);
    animateValue("resFruitConf", `${itemConf}%`);
    animateValue("resCondConf", `${data.cond_confidence}%`);

    renderBars("condBars", data.cond_probs);

    // Grad-CAM
    const camWrap = document.getElementById("camWrap");
    if (data.gradcam) {
      const camImg = document.getElementById("camImg");
      camImg.src = data.gradcam;
      camWrap.classList.remove("hidden");
    } else {
      camWrap.classList.add("hidden");
    }

  } else if (source === "phone") {
    // Phone Camera
    const card = document.getElementById("phoneResultCard");
    card.style.display = "block";

    const banner = document.getElementById("phoneSafetyBanner");
    banner.textContent = getSafetyEmoji(data.safety_class) + " " + data.safety;
    banner.className = `result-hero ${data.safety_class}`;

    const itemName = data.veggie || data.fruit || '—';
    document.getElementById("phoneFruit").textContent = itemName;
    document.getElementById("phoneCondition").textContent = data.condition;
    document.getElementById("phoneConf").textContent = `${data.cond_confidence}%`;
    document.getElementById("phoneTime").textContent = new Date(data.timestamp).toLocaleTimeString();

    renderBars("phoneCondBars", data.cond_probs);
  } else {
    // ESP32
    const card = document.getElementById("esp32ResultCard");
    card.style.display = "block";

    const banner = document.getElementById("esp32SafetyBanner");
    banner.textContent = getSafetyEmoji(data.safety_class) + " " + data.safety;
    banner.className = `result-hero ${data.safety_class}`;

    document.getElementById("e32Fruit").textContent = data.fruit;
    document.getElementById("e32Condition").textContent = data.condition;
    document.getElementById("e32Conf").textContent = `${data.cond_confidence}%`;
    document.getElementById("e32Time").textContent = new Date(data.timestamp).toLocaleTimeString();

    renderBars("e32CondBars", data.cond_probs);
  }
}

function getSafetyEmoji(safetyClass) {
  const map = { "safe-class": "✅", "warn-class": "⚠️", "danger-class": "🚨" };
  return map[safetyClass] || "ℹ️";
}

function animateValue(id, value) {
  const el = document.getElementById(id);
  el.style.opacity = "0";
  el.style.transform = "translateY(4px)";
  el.style.transition = "opacity 0.3s, transform 0.3s";
  setTimeout(() => {
    el.textContent = value;
    el.style.opacity = "1";
    el.style.transform = "translateY(0)";
  }, 80);
}

function renderBars(containerId, probs) {
  if (!probs) return;
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  const colorMap = {
    "Fresh": "safe",
    "Rotten": "warn",
    "Formalin-mixed": "danger",
    "Formalin_Mixed": "danger",
    "Adulterated": "danger",
  };

  for (const [label, pct] of Object.entries(probs)) {
    const colorClass = colorMap[label] || "safe";
    const el = document.createElement("div");
    el.className = "bar-item";
    el.innerHTML = `
      <span class="bar-label">${label.replace("_", " ")}</span>
      <div class="bar-track">
        <div class="bar-fill ${colorClass}" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="bar-pct">${pct.toFixed(1)}%</span>
    `;
    container.appendChild(el);
  }

  // Animate bars after short delay for smooth entry
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll(".bar-fill").forEach(fill => {
        fill.style.width = fill.dataset.target + "%";
      });
    }, 100);
  });
}

// ── Toast Notification ───────────────────────────────────────
function showToast(message, type = "safe") {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 20px; font-size: 0.88rem;
    color: var(--text); box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    animation: slideUp 0.3s ease; max-width: 360px;
    backdrop-filter: blur(12px);
  `;
  if (type === "danger") toast.style.borderColor = "rgba(248,113,113,0.4)";
  if (type === "warn") toast.style.borderColor = "rgba(251,191,36,0.4)";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = "0"; toast.style.transition = "opacity 0.3s"; }, 3500);
  setTimeout(() => document.body.removeChild(toast), 3900);
}

// ── ESP32-CAM Polling ────────────────────────────────────────
function startPolling() {
  document.getElementById("startPollBtn").style.display = "none";
  document.getElementById("stopPollBtn").style.display = "inline-block";
  document.getElementById("pollStatus").innerHTML =
    `<span class="live-badge"><span class="live-dot"></span>LIVE – updating every 3s</span>`;

  pollInterval = setInterval(pollESP32, 3000);
  pollESP32();
}

function stopPolling() {
  clearInterval(pollInterval);
  pollInterval = null;
  const startBtn = document.getElementById("startPollBtn");
  const stopBtn = document.getElementById("stopPollBtn");
  const status = document.getElementById("pollStatus");
  if (!startBtn || !stopBtn || !status) return;
  startBtn.style.display = "inline-block";
  stopBtn.style.display = "none";
  status.textContent = "";
}

async function pollESP32() {
  try {
    const res = await fetch(`${API_BASE}/api/latest-esp32`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();

    if (data.status === "no_data") {
      document.getElementById("pollStatus").innerHTML =
        `<span style="color:var(--warn)">🟡 Waiting for ESP32-CAM…</span>`;
      return;
    }

    // Show received image if available
    if (data.image_b64) {
      const img = document.getElementById("esp32PreviewImg");
      img.src = "data:image/jpeg;base64," + data.image_b64;
      img.classList.remove("hidden");
      document.querySelector(".esp32-placeholder")?.classList.add("hidden");
    }

    document.getElementById("pollStatus").innerHTML =
      `<span class="live-badge"><span class="live-dot"></span>LIVE – updating every 3s</span>`;
    renderResult(data, "esp32");

  } catch {
    document.getElementById("pollStatus").innerHTML =
      `<span style="color:var(--danger)">⚠️ Connection error — retrying…</span>`;
  }
}

// ── Phone Camera QR & Polling ────────────────────────────────
async function loadPhoneCamQR() {
  try {
    const res = await fetch(`${API_BASE}/api/server-info`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    phoneCamUrl = data.phone_cam_url;

    document.getElementById("phoneCamUrl").textContent = phoneCamUrl;
    const container = document.getElementById("qrContainer");
    container.innerHTML = '';

    // Use backend-generated QR code (works offline, no external API needed)
    const img = document.createElement("img");
    img.src = `${API_BASE}/api/qr-code`;
    img.alt = "Scan this QR code with your phone";
    img.className = "qr-svg";
    img.style.background = "#fff";
    img.style.padding = "8px";
    img.style.borderRadius = "12px";
    img.onerror = () => {
      container.innerHTML = `<div style="text-align:center;padding:16px;">
        <p style="color:var(--text-muted);font-size:0.82rem;margin-bottom:8px;">QR could not load. Open this URL on your phone:</p>
        <code style="color:var(--accent);font-size:0.85rem;word-break:break-all;">${phoneCamUrl}</code>
      </div>`;
    };
    container.appendChild(img);
  } catch {
    document.getElementById("qrContainer").innerHTML =
      '<div class="qr-loading" style="color:var(--danger)">❌ Could not detect server IP</div>';
  }
}

function copyPhoneUrl() {
  if (!phoneCamUrl) return;
  navigator.clipboard.writeText(phoneCamUrl).then(() => {
    showToast("📋 URL copied to clipboard!", "safe");
    const btn = document.getElementById("copyUrlBtn");
    btn.textContent = "✅";
    setTimeout(() => { btn.textContent = "📋"; }, 1500);
  });
}

function startPhonePolling() {
  document.getElementById("startPhonePollBtn").style.display = "none";
  document.getElementById("stopPhonePollBtn").style.display = "inline-block";
  document.getElementById("phonePollStatus").innerHTML =
    `<span class="live-badge"><span class="live-dot"></span>LIVE – updating every 3s</span>`;

  phonePollInterval = setInterval(pollPhone, 3000);
  pollPhone();
}

function stopPhonePolling() {
  clearInterval(phonePollInterval);
  phonePollInterval = null;
  const startBtn = document.getElementById("startPhonePollBtn");
  const stopBtn = document.getElementById("stopPhonePollBtn");
  const status = document.getElementById("phonePollStatus");
  if (!startBtn || !stopBtn || !status) return;
  startBtn.style.display = "inline-block";
  stopBtn.style.display = "none";
  status.textContent = "";
}

async function pollPhone() {
  try {
    const res = await fetch(`${API_BASE}/api/latest-phone`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();

    if (data.status === "no_data") {
      document.getElementById("phonePollStatus").innerHTML =
        `<span style="color:var(--warn)">🟡 Waiting for phone camera…</span>`;
      return;
    }

    // Show received image
    if (data.image_b64) {
      const img = document.getElementById("phonePreviewImg");
      img.src = "data:image/jpeg;base64," + data.image_b64;
      img.classList.remove("hidden");
      document.getElementById("phonePlaceholder")?.classList.add("hidden");
    }

    document.getElementById("phonePollStatus").innerHTML =
      `<span class="live-badge"><span class="live-dot"></span>LIVE – updating every 3s</span>`;
    renderResult(data, "phone");

  } catch {
    document.getElementById("phonePollStatus").innerHTML =
      `<span style="color:var(--danger)">⚠️ Connection error — retrying…</span>`;
  }
}

