
(() => {
  // --------------- Utilities ---------------
  const $ = (sel) => document.querySelector(sel);
  const $all = (sel) => Array.from(document.querySelectorAll(sel));

  const notices = $("#notices");
  const year = $("#year");
  year.textContent = new Date().getFullYear();

  function notify(msg, type = "info", timeout = 3500) {
    const div = document.createElement("div");
    div.className = `notice ${type}`;
    div.textContent = msg;
    notices.appendChild(div);
    if (timeout) setTimeout(() => div.remove(), timeout);
  }

  function readJSON(text) {
    if (!text || text.trim() === "") return null;
    try { return JSON.parse(text); } catch (e) { return null; }
  }

  function pretty(obj) {
    try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
  }

  function getStored(k, fallback="") {
    const v = localStorage.getItem(k);
    return v !== null ? v : fallback;
  }

  function saveStored(k, v) {
    localStorage.setItem(k, v);
  }

  function buildBaseUrl(raw) {
    if (!raw) return ""; // same origin
    // strip trailing slash
    return raw.replace(/\/+$/, "");
  }

  // --------------- Settings ---------------
  const baseUrlInput = $("#baseUrl");
  const apiKeyInput = $("#apiKey");
  baseUrlInput.value = getStored("ode.baseUrl", "");
  apiKeyInput.value = getStored("ode.apiKey", "");

  let BASE_URL = buildBaseUrl(baseUrlInput.value);
  let API_KEY = apiKeyInput.value;

  function api(path, { method = "GET", body = null, headers = {} } = {}) {
    const url = `${BASE_URL}${path}`;
    const finalHeaders = {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY || "",
      ...headers
    };
    return fetch(url, {
      method,
      headers: finalHeaders,
      body: body ? JSON.stringify(body) : null
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text}`);
      }
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) return res.json();
      return res.text();
    });
  }

  $("#btnSaveSettings").addEventListener("click", () => {
    BASE_URL = buildBaseUrl(baseUrlInput.value);
    API_KEY = apiKeyInput.value;
    saveStored("ode.baseUrl", BASE_URL);
    saveStored("ode.apiKey", API_KEY);
    notify("Settings saved.");
  });

  $("#btnPing").addEventListener("click", async () => {
    try {
      const info = await api("/");
      notify("Server reachable. See console for details.");
      console.log(info);
    } catch (e) {
      notify(`Ping failed: ${e.message}`, "error", 6000);
    }
  });

  // --------------- WebSocket ---------------
  const btnConnectWs = $("#btnConnectWs");
  const btnDisconnectWs = $("#btnDisconnectWs");
  const wsStatus = $("#wsStatus");

  let ws = null;
  const clientId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
  let subscriptions = new Set();

  function wsUrl() {
    const origin = BASE_URL || window.location.origin;
    return origin.replace(/^http/, "ws") + `/ws/${clientId}`;
  }

  function wsConnect() {
    if (ws) try { ws.close(); } catch {}
    ws = new WebSocket(wsUrl());
    ws.onopen = () => { wsStatus.textContent = "Connected"; notify("WebSocket connected."); };
    ws.onclose = () => { wsStatus.textContent = "Disconnected"; notify("WebSocket disconnected.", "warn"); };
    ws.onerror = (ev) => { console.error("ws error", ev); notify("WebSocket error", "error"); };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "ode_generated") {
          appendGenRow(msg.ode);
        } else if (msg.type === "subscribed") {
          notify(`Subscribed to ${msg.topic}`);
        } else if (msg.type === "pong") {
          // ignore
        } else {
          console.log("WS message:", msg);
        }
      } catch (e) {
        console.log("WS text:", ev.data);
      }
    };
  }

  function wsSend(obj) {
    if (!ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify(obj));
  }

  function subscribe(topic) {
    if (!subscriptions.has(topic)) {
      wsSend({ type: "subscribe", topic });
      subscriptions.add(topic);
    }
  }

  function unsubscribe(topic) {
    if (subscriptions.has(topic)) {
      wsSend({ type: "unsubscribe", topic });
      subscriptions.delete(topic);
    }
  }

  btnConnectWs.addEventListener("click", wsConnect);
  btnDisconnectWs.addEventListener("click", () => { try { ws.close(); } catch{} });

  // --------------- Dashboard ---------------
  async function refreshDashboard() {
    try {
      const stats = await api("/api/stats");
      $("#statGenerated").textContent = (stats.statistics?.total_generated_24h ?? 0).toLocaleString();
      $("#statVerifyRate").textContent = ((stats.statistics?.verification_rate ?? 0) * 100).toFixed(1) + "%";
      $("#statActiveJobs").textContent = stats.statistics?.active_jobs ?? "0";
      $("#statTotalJobs").textContent = stats.statistics?.total_jobs ?? "0";

      const gens = await api("/api/generators");
      $("#availableGenerators").textContent = pretty(gens);

      const funcs = await api("/api/functions");
      $("#availableFunctions").textContent = pretty(funcs);

      const health = await api("/health");
      $("#healthStatus").textContent = pretty(health);
    } catch (e) {
      notify(`Dashboard refresh failed: ${e.message}`, "error", 6000);
    }
  }

  // Populate selects with generators+functions
  async function populateSelects() {
    try {
      const gens = await api("/api/generators");
      const allGens = gens.all || [];
      const genSel = $("#genSelect");
      genSel.innerHTML = "";
      allGens.forEach((g) => {
        const opt = document.createElement("option");
        opt.value = g; opt.textContent = g;
        genSel.appendChild(opt);
      });

      const funcs = await api("/api/functions");
      const names = funcs.functions || [];
      const funcSel = $("#funcSelect");
      funcSel.innerHTML = "";
      names.forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f; opt.textContent = f;
        funcSel.appendChild(opt);
      });
    } catch (e) {
      notify(`Failed to load generators/functions: ${e.message}`, "error", 6000);
    }
  }

  // --------------- Generate ODEs ---------------
  const genResultsTbody = $("#genResults");
  const genProgress = $("#genProgress");
  const genJobPanel = $("#genJob");
  const genJobIdSpan = $("#genJobId");

  function appendGenRow(ode) {
    const idx = genResultsTbody.children.length + 1;
    const tr = document.createElement("tr");
    const verification = ode.verification || {};
    tr.innerHTML = `
      <td>${idx}</td>
      <td>${ode.generator}</td>
      <td>${ode.function}</td>
      <td>${(verification.verified ?? ode.verified) ? "✓" : "✗"}</td>
      <td>${(verification.confidence ?? ode.verification_confidence ?? 0).toFixed ? (verification.confidence ?? ode.verification_confidence ?? 0).toFixed(3) : (verification.confidence ?? ode.verification_confidence ?? 0)}</td>
      <td><code>${escapeHtml(ode.ode)}</code></td>
      <td><code>${escapeHtml(ode.solution)}</code></td>
    `;
    genResultsTbody.appendChild(tr);
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  $("#btnGenerate").addEventListener("click", async () => {
    genResultsTbody.innerHTML = "";
    genProgress.value = 0;
    genJobPanel.classList.add("hidden");

    const req = {
      generator: $("#genSelect").value,
      function: $("#funcSelect").value,
      count: parseInt($("#genCount").value || "1", 10),
      verify: $("#verifyToggle").value === "true",
      stream: $("#streamToggle").checked,
      parameters: readJSON($("#paramJson").value)
    };
    try {
      const resp = await api("/api/generate", { method: "POST", body: req });
      genJobIdSpan.textContent = resp.job_id;
      genJobPanel.classList.remove("hidden");
      if ($("#streamToggle").checked) {
        subscribe(`job:${resp.job_id}`);
      }
      refreshJob(resp.job_id, genProgress);
    } catch (e) {
      notify(`Generation failed: ${e.message}`, "error", 7000);
    }
  });

  $("#btnRefreshGenJob").addEventListener("click", () => {
    const id = genJobIdSpan.textContent || "";
    if (id) refreshJob(id, genProgress);
  });

  // --------------- Batch Generation ---------------
  const batchJobPanel = $("#batchJob");
  const batchProgress = $("#batchProgress");
  const batchJobIdSpan = $("#batchJobId");
  const batchMeta = $("#batchMeta");

  $("#btnBatchGenerate").addEventListener("click", async () => {
    batchJobPanel.classList.add("hidden");
    batchProgress.value = 0;
    batchMeta.textContent = "";

    const gens = ($("#batchGenerators").value || "").split(",").map(s => s.trim()).filter(Boolean);
    const funcs = ($("#batchFunctions").value || "").split(",").map(s => s.trim()).filter(Boolean);
    const req = {
      generators: gens,
      functions: funcs,
      samples_per_combination: parseInt($("#batchSamples").value || "5", 10),
      parameter_ranges: readJSON($("#batchParamRanges").value),
      verify: $("#batchVerify").value === "true",
      save_dataset: $("#batchSave").value === "true",
      dataset_name: $("#batchDatasetName").value || null
    };

    try {
      const resp = await api("/api/batch_generate", { method: "POST", body: req });
      batchJobIdSpan.textContent = resp.job_id;
      batchJobPanel.classList.remove("hidden");
      refreshJob(resp.job_id, batchProgress, (job) => {
        batchMeta.textContent = pretty(job.metadata || {});
      });
    } catch (e) {
      notify(`Batch start failed: ${e.message}`, "error", 7000);
    }
  });

  // --------------- Verify ---------------
  $("#btnVerify").addEventListener("click", async () => {
    const req = {
      ode: $("#verifyOde").value,
      solution: $("#verifySol").value,
      method: $("#verifyMethod").value,
      timeout: parseInt($("#verifyTimeout").value || "30", 10)
    };
    try {
      const result = await api("/api/verify", { method: "POST", body: req });
      $("#verifyResult").textContent = pretty(result);
    } catch (e) {
      notify(`Verification failed: ${e.message}`, "error", 7000);
    }
  });

  // --------------- Datasets ---------------
  async function listDatasets() {
    try {
      const data = await api("/api/datasets");
      const tbody = $("#datasetRows");
      tbody.innerHTML = "";
      (data.datasets || []).forEach(ds => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${ds.name}</td>
          <td>${(ds.size_bytes ?? 0).toLocaleString()}</td>
          <td>${ds.created_at || ""}</td>
          <td>
            <a href="${(BASE_URL || "")}/api/datasets/${encodeURIComponent(ds.name)}/download?format=jsonl" target="_blank">JSONL</a>
            &nbsp;|&nbsp;
            <a href="${(BASE_URL || "")}/api/datasets/${encodeURIComponent(ds.name)}/download?format=csv" target="_blank">CSV</a>
          </td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      notify(`Failed to list datasets: ${e.message}`, "error", 7000);
    }
  }
  $("#btnListDatasets").addEventListener("click", listDatasets);

  $("#btnCreateDatasetFromJob").addEventListener("click", async () => {
    const jobId = $("#dsFromJobId").value.trim();
    const name = $("#dsName").value.trim() || `dataset_${Date.now()}`;
    if (!jobId) { notify("Provide a job id", "warn"); return; }
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (!job.results || !Array.isArray(job.results) || job.results.length === 0) {
        notify("Job has no results to persist.", "warn");
        return;
      }
      const meta = await api(`/api/datasets/create?name=${encodeURIComponent(name)}`, { method: "POST", body: job.results });
      notify(`Dataset created: ${meta.name}`);
      listDatasets();
    } catch (e) {
      notify(`Create dataset failed: ${e.message}`, "error", 7000);
    }
  });

  // --------------- Jobs ---------------
  async function listJobs() {
    const status = $("#jobFilter").value;
    const limit = parseInt($("#jobLimit").value || "100", 10);
    try {
      const data = await api(`/api/jobs?status=${encodeURIComponent(status)}&limit=${limit}`);
      const tbody = $("#jobRows");
      tbody.innerHTML = "";
      (data.jobs || []).forEach(j => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${j.job_id}</code></td>
          <td>${j.metadata?.type || j.metadata?.job_type || ""}</td>
          <td>${j.status}</td>
          <td>${(j.progress ?? 0).toFixed(1)}%</td>
          <td>${j.updated_at || ""}</td>
          <td>
            <button class="secondary" data-action="inspect" data-id="${j.job_id}">Inspect</button>
            <button class="danger" data-action="cancel" data-id="${j.job_id}">Cancel</button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      tbody.addEventListener("click", async (ev) => {
        const el = ev.target.closest("button[data-action]");
        if (!el) return;
        const id = el.getAttribute("data-id");
        const action = el.getAttribute("data-action");
        if (action === "inspect") {
          const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
          alert(pretty(job));
        } else if (action === "cancel") {
          if (!confirm("Cancel this job?")) return;
          await api(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
          listJobs();
        }
      }, { once: true });

    } catch (e) {
      notify(`Failed to list jobs: ${e.message}`, "error", 7000);
    }
  }
  $("#btnListJobs").addEventListener("click", listJobs);

  async function refreshJob(jobId, progressEl, onMeta) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      progressEl.value = job.progress || 0;
      if (typeof onMeta === "function") {
        onMeta(job);
      }
    } catch (e) {
      notify(`Failed to refresh job: ${e.message}`, "error", 6000);
    }
  }

  // --------------- ML ---------------
  $("#btnTrainModel").addEventListener("click", async () => {
    try {
      const req = {
        dataset: $("#mlDataset").value.trim(),
        model_type: $("#mlModelType").value,
        epochs: parseInt($("#mlEpochs").value || "10", 10),
        batch_size: parseInt($("#mlBatch").value || "32", 10),
        learning_rate: parseFloat($("#mlLR").value || "0.001"),
        early_stopping: $("#mlES").value === "true",
        validation_split: parseFloat($("#mlVal").value || "0.2")
      };
      const resp = await api("/api/ml/train", { method: "POST", body: req });
      notify(`Training job created: ${resp.job_id}`);
    } catch (e) {
      notify(`Training start failed: ${e.message}`, "error", 7000);
    }
  });

  $("#btnListModels").addEventListener("click", async () => {
    try {
      const data = await api("/api/ml/models");
      const tbody = $("#modelRows");
      tbody.innerHTML = "";
      (data.models || []).forEach(m => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${m.name}</td>
          <td>${(m.size_bytes ?? 0).toLocaleString()}</td>
          <td>${m.created_at || ""}</td>
          <td><pre class="pre">${escapeHtml(pretty(m.metadata || {}))}</pre></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      notify(`Model listing failed: ${e.message}`, "error", 7000);
    }
  });

  $("#btnMLGenerate").addEventListener("click", async () => {
    try {
      const req = {
        model_path: $("#mlModelPath").value.trim(),
        n_samples: parseInt($("#mlNSamples").value || "5", 10),
        temperature: parseFloat($("#mlTemp").value || "0.8"),
        generators: null,
        functions: null
      };
      const resp = await api("/api/ml/generate", { method: "POST", body: req });
      $("#mlGenResults").textContent = pretty(resp);
    } catch (e) {
      notify(`ML generation failed: ${e.message}`, "error", 7000);
    }
  });

  // --------------- Boot ---------------
  populateSelects();
  refreshDashboard();
  listJobs();
  listDatasets();

  // Auto-reconnect WS if user saved settings previously
  if (getStored("ode.apiKey") !== null) {
    // Optionally connect
  }

})();