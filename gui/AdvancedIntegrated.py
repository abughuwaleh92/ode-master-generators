# streamlit_app.py
"""
ODE Master Generator — Streamlit UI (Full Rewrite)
==================================================

Key features
------------
• API prefix auto-detection (works with /api/v1 or root-mounted APIs)
• Config via env: API_BASE_URL, API_KEY, API_PREFIX (optional), USE_DEMO
• Demo fallback mode to keep UI usable without a running backend
• End-to-end surface: Quick Generate, Batch, Verify, Datasets, ML Train/Generate, Analysis, Tools
• Robust client: normalized responses, clear error messages, metrics preview
"""

from __future__ import annotations

import os
import time
import json
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional for parsing/plotting analytic expressions
try:
    import sympy as sp
except Exception:
    sp = None  # pragma: no cover

# ---------- Config ----------

st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = (os.getenv("API_BASE_URL") or "http://localhost:8000").rstrip("/")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1").strip()
if API_PREFIX and not API_PREFIX.startswith("/"):
    API_PREFIX = "/" + API_PREFIX
API_KEY = os.getenv("API_KEY", "test-key")
USE_DEMO = str(os.getenv("USE_DEMO", "0")).lower() in {"1", "true", "yes", "on"}
REQUEST_TIMEOUT = 30

# ---------- Styling ----------

st.markdown(
    """
    <style>
      .app-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:18px 22px;border-radius:14px;color:#fff;margin-bottom:18px;box-shadow:0 8px 20px rgba(0,0,0,.08)}
      .metric-card{background:#fff;padding:14px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:8px}
      .info-box{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:12px}
      .warn-box{background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:12px}
      .ok-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:8px;box-shadow:0 0 8px #10b981}
      .bad-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:8px;box-shadow:0 0 8px #ef4444}
      .latex-box{font-size:1.05em;padding:10px;border-radius:8px;background:#f8fafc;border:1px solid #e5e7eb;overflow-x:auto}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session ----------

def _init_state():
    ss = st.session_state
    ss.setdefault("api_status", None)
    ss.setdefault("api_prefix", API_PREFIX)  # detected/active prefix
    ss.setdefault("available_generators", [])
    ss.setdefault("available_functions", [])
    ss.setdefault("generated_odes", [])
    ss.setdefault("batch_dataset", [])
    ss.setdefault("current_dataset", None)
    ss.setdefault("available_datasets", [])
    ss.setdefault("available_models", [])
    ss.setdefault("ml_generated_odes", [])

_init_state()

# ---------- API Client ----------

class API:
    def __init__(self, base: str, key: str, prefix: str = API_PREFIX):
        self.base = base.rstrip("/")
        self.prefix = prefix.rstrip("/") if prefix else ""
        self.headers = {"X-API-Key": key, "Content-Type": "application/json"}

    # ---- URL builder
    def _make_url(self, path: str, *, prefixed: bool = True) -> str:
        if not prefixed:  # health/metrics are commonly unprefixed
            return f"{self.base}{path}"
        return f"{self.base}{self.prefix}{path}"

    # ---- Low-level request
    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        timeout: int = REQUEST_TIMEOUT,
        prefixed: bool = True,
    ) -> Tuple[Optional[Any], Optional[str]]:
        url = self._make_url(path, prefixed=prefixed)
        try:
            r = requests.request(method, url, headers=self.headers, json=json_body, timeout=timeout)
            if r.status_code >= 400:
                return None, f"{method} {url} → HTTP {r.status_code}: {r.text[:300]}"
            # /metrics is text/plain
            if path.endswith("/metrics"):
                return r.text, None
            return (r.json() if r.text.strip() else None), None
        except Exception as e:  # pragma: no cover
            return None, f"{method} {url} → {e}"

    # ---- Normalized helpers
    def health(self) -> Dict[str, Any]:
        data, err = self._request("GET", "/health", timeout=8, prefixed=False)
        if err:
            return {"status": "error", "error": err}
        return data or {"status": "unknown"}

    def _to_list(self, data: Any, *candidate_keys: str) -> List[str]:
        if data is None:
            return []
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            for k in candidate_keys:
                v = data.get(k)
                if isinstance(v, list):
                    return [str(x) for x in v]
        return []

    def generators(self) -> List[str]:
        data, _ = self._request("GET", "/generators")
        if isinstance(data, dict):
            # prefer explicit keys
            linear = data.get("linear", [])
            nonlinear = data.get("nonlinear", [])
            allv = data.get("all", [])
            if linear or nonlinear:
                return [*map(str, linear), *map(str, nonlinear)]
            if allv:
                return [str(x) for x in allv]
        return self._to_list(data, "generators", "all", "items")

    def functions(self) -> List[str]:
        data, _ = self._request("GET", "/functions")
        if isinstance(data, dict):
            fx = data.get("functions")
            if isinstance(fx, list):
                return [str(x) for x in fx]
        return self._to_list(data, "functions", "items")

    def generate(self, *, generator: str, function: str, parameters: Dict[str, Any], count: int = 1, verify: bool = True) -> Dict[str, Any] | None:
        body = {"generator": generator, "function": function, "parameters": parameters, "count": count, "verify": verify}
        data, err = self._request("POST", "/generate", json_body=body)
        if err:
            st.error(f"Generate failed: {err}")
            return None
        return data

    def batch_generate(self, *, generators: List[str], functions: List[str], samples_per_combination: int, parameters: Optional[Dict[str, Any]] = None, verify: bool = True, dataset_name: Optional[str] = None) -> Dict[str, Any] | None:
        body: Dict[str, Any] = {"generators": generators, "functions": functions, "samples_per_combination": samples_per_combination, "verify": verify}
        if parameters:
            body["parameters"] = parameters
        if dataset_name:
            body["dataset_name"] = dataset_name
        data, err = self._request("POST", "/batch_generate", json_body=body)
        if err:
            st.error(f"Batch generate failed: {err}")
            return None
        return data

    def verify(self, *, ode: str, solution: str, method: str = "substitution") -> Dict[str, Any] | None:
        body = {"ode": ode, "solution": solution, "method": method}
        data, err = self._request("POST", "/verify", json_body=body)
        if err:
            st.error(f"Verify failed: {err}")
            return None
        return data

    def datasets(self) -> Dict[str, Any]:
        data, _ = self._request("GET", "/datasets")
        return data or {"datasets": [], "count": 0}

    def create_dataset(self, *, odes: List[Dict[str, Any]], dataset_name: Optional[str] = None) -> Dict[str, Any] | None:
        body = {"odes": odes, "dataset_name": dataset_name}
        data, err = self._request("POST", "/datasets/create", json_body=body)
        if err:
            st.error(f"Create dataset failed: {err}")
            return None
        return data

    def models(self) -> Dict[str, Any]:
        data, _ = self._request("GET", "/models")
        return data or {"models": [], "count": 0}

    def stats(self) -> Dict[str, Any]:
        data, _ = self._request("GET", "/stats")
        return data or {}

    def metrics(self) -> str:
        data, _ = self._request("GET", "/metrics", prefixed=False)
        return data or ""

    def job_status(self, job_id: str) -> Dict[str, Any] | None:
        data, err = self._request("GET", f"/jobs/{job_id}")
        if err:
            st.error(f"Job status failed: {err}")
            return None
        return data

api = API(API_BASE_URL, API_KEY, API_PREFIX)

# ---------- Demo fallback ----------

DEMO_GENERATORS = ["L1","L2","L3","L4","N1","N2","N3","N4","N5","N6","N7"]
DEMO_FUNCTIONS = ["sine","cosine","tangent_safe","exponential","exp_scaled","quadratic","cubic","sinh","cosh","tanh","log_safe"]

# ---------- Cache wrappers ----------

@st.cache_data(ttl=60)
def cached_generators() -> List[str]:
    return api.generators()

@st.cache_data(ttl=60)
def cached_functions() -> List[str]:
    return api.functions()

@st.cache_data(ttl=30)
def cached_stats() -> Dict[str, Any]:
    return api.stats()

# ---------- Helpers ----------

def header():
    st.markdown(
        """
        <div class="app-header">
          <h2 style="margin:0">🔬 ODE Master Generator</h2>
          <div style="opacity:.9">Generate, verify, and analyze ODEs — end to end.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_chip(ok: bool) -> str:
    return f"<span class='{ 'ok-dot' if ok else 'bad-dot' }'></span>{'API Online' if ok else 'API Offline'}"

def probe_and_load_resources():
    """Probe /health, load generators & functions (with demo fallback + auto prefix detection)."""
    if st.session_state.api_status is None:
        with st.status("Checking API status…", expanded=True) as s:
            h = api.health()
            st.session_state.api_status = h
            if h.get("status") in {"healthy","ok","operational"}:
                s.update(label="API healthy", state="complete")
            else:
                s.update(label="API not healthy", state="error")

    # Try with current prefix → if empty list, try toggling
    gens = cached_generators()
    funcs = cached_functions()

    if not gens or not funcs:
        old = api.prefix
        candidates = [old]
        if old:
            candidates.append("")
        else:
            candidates.append("/api/v1")
        for cand in candidates:
            api.prefix = cand
            st.session_state.api_prefix = cand
            gens = api.generators()
            funcs = api.functions()
            if gens and funcs:
                break

    if (not gens or not funcs) and USE_DEMO:
        gens = gens or DEMO_GENERATORS
        funcs = funcs or DEMO_FUNCTIONS

    st.session_state.available_generators = gens
    st.session_state.available_functions = funcs

def render_equation_block(title: str, expr: str):
    st.markdown(f"**{title}**")
    st.markdown(f"<div class='latex-box'>{expr}</div>", unsafe_allow_html=True)

def plot_solution(expression: str, x_range: Tuple[float, float] = (-5,5), params: Optional[Dict[str, Any]] = None):
    if not sp:
        st.info("Sympy not available; skipping plot.")
        return None
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(expression)
        if params:
            for k, v in params.items():
                try:
                    expr = expr.subs(sp.Symbol(str(k)), v)
                except Exception:
                    pass
        f = sp.lambdify(x, expr, "numpy")
        xs = np.linspace(x_range[0], x_range[1], 1000)
        ys = f(xs)
        ys = np.real(np.nan_to_num(ys, nan=np.nan))
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="y(x)"))
        fig.update_layout(template="plotly_white", height=420, title="Solution plot", xaxis_title="x", yaxis_title="y(x)")
        return fig
    except Exception as e:
        st.info(f"Plotting skipped: {e}")
        return None

def wait_for_job(job_id: str, *, max_secs: int = 900, poll: float = 1.5) -> Optional[Dict[str, Any]]:
    start = time.time()
    prog = st.progress(0)
    info = st.empty()
    while time.time() - start < max_secs:
        js = api.job_status(job_id)
        if not js:
            time.sleep(poll)
            continue
        status = js.get("status", "unknown")
        progress = int(js.get("progress", 0))
        prog.progress(min(progress, 100))
        meta = js.get("metadata", {})
        if meta:
            if "current_epoch" in meta:
                info.text(f"{status} — epoch {meta.get('current_epoch')}/{meta.get('total_epochs','?')}")
            elif "current" in meta and "total" in meta:
                info.text(f"{status} — {meta.get('current')}/{meta.get('total')}")
            else:
                info.text(status)
        if status == "completed":
            prog.progress(100)
            return js
        if status == "failed":
            st.error(js.get("error", "Job failed"))
            return None
        time.sleep(poll)
    st.error("Job timed out")
    return None

def download_jsonl(data: List[Dict[str, Any]], filename: str = "dataset.jsonl"):
    payload = "\n".join(json.dumps(o) for o in data)
    b64 = base64.b64encode(payload.encode()).decode()
    st.markdown(f"<a download='{filename}' href='data:application/json;base64,{b64}'>📥 Download {filename}</a>", unsafe_allow_html=True)

# ---------- UI Widgets ----------

def params_controls(generator: str, key_prefix: str = "") -> Dict[str, Any]:
    col1, col2 = st.columns(2)
    params: Dict[str, Any] = {}
    with col1:
        params["alpha"] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1, key=f"{key_prefix}_alpha")
        params["beta"]  = st.slider("β (beta)", 0.1,  3.0, 1.0, 0.1, key=f"{key_prefix}_beta")
    with col2:
        params["M"]     = st.slider("M", -1.0, 1.0, 0.0, 0.1, key=f"{key_prefix}_M")
        if generator.startswith("N"):
            params["q"] = st.slider("q (power)", 2, 5, 2, key=f"{key_prefix}_q")
            if generator in {"N2","N3","N6","N7"}:
                params["v"] = st.slider("v (power)", 2, 5, 3, key=f"{key_prefix}_v")
        if generator in {"L4","N6"}:
            params["a"] = st.slider("a (pantograph)", 2.0, 5.0, 2.0, 0.5, key=f"{key_prefix}_a")
    return params

# ---------- Pages ----------

def page_dashboard():
    st.title("Dashboard")
    probe_and_load_resources()

    online = (st.session_state.api_status or {}).get("status") in {"healthy","ok","operational"}
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown("<div class='metric-card'>" + status_chip(online) + "</div>", unsafe_allow_html=True)
    stats = cached_stats()
    with colB:
        st.metric("Generated (24h)", f"{stats.get('total_generated_24h', 0):,}")
    with colC:
        st.metric("Verification Rate", f"{stats.get('verification_success_rate', 0):.1%}")
    with colD:
        st.metric("Active Jobs", stats.get("active_jobs", 0))

    st.subheader("Recent ODEs")
    if not st.session_state.generated_odes:
        st.info("No recent ODEs yet. Try Quick Generate.")
    else:
        for i, ode in enumerate(st.session_state.generated_odes[-5:][::-1], 1):
            with st.expander(f"Recent {i}: {ode.get('generator','?')} × {ode.get('function','?')}"):
                render_equation_block("ODE", ode.get("ode", ode.get("ode_symbolic", "")))
                sol = ode.get("solution") or ode.get("solution_symbolic")
                if sol:
                    render_equation_block("Solution", f"y(x) = {sol}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity", ode.get("complexity", ode.get("complexity_score", "N/A")))
                c2.metric("Verified", "✅" if ode.get("verified") else "❌")
                gt = ode.get("properties", {}).get("generation_time_ms") or (ode.get("generation_time", 0) * 1000)
                c3.metric("Gen Time", f"{gt:.0f} ms")

    st.subheader("Resources")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generators", len(st.session_state.available_generators))
    c2.metric("Functions", len(st.session_state.available_functions))
    c3.metric("Datasets", len(st.session_state.available_datasets))
    c4.metric("Models", len(st.session_state.available_models))

def page_quick_generate():
    st.title("⚡ Quick Generate")
    probe_and_load_resources()

    gens = st.session_state.available_generators
    funcs = st.session_state.available_functions
    if not gens or not funcs:
        st.error("No generators/functions available. Check API or enable USE_DEMO=1.")
        return

    c1, c2 = st.columns(2)
    with c1:
        generator = st.selectbox("Generator", gens, index=0)
    with c2:
        function = st.selectbox("Function", funcs, index=0)

    st.markdown("### Parameters")
    params = params_controls(generator, key_prefix="quick")

    with st.expander("Advanced"):
        colA, colB = st.columns(2)
        with colA:
            verify = st.checkbox("Verify solution", True)
            count  = st.number_input("Number of ODEs", 1, 10, 1)
        with colB:
            show_plot = st.checkbox("Plot solution (if available)", True)
            x_range   = st.slider("Plot range", -10.0, 10.0, (-5.0, 5.0))

    if st.button("🚀 Generate", type="primary"):
        with st.status("Generating…", expanded=True) as s:
            resp = api.generate(generator=generator, function=function, parameters=params, count=count, verify=verify)
            if not resp or "job_id" not in resp:
                s.update(label="Failed to start job", state="error")
                return
            s.update(label=f"Job {resp['job_id']} started")
        js = wait_for_job(resp["job_id"], max_secs=600)
        if js and js.get("results"):
            st.success(f"Generated {len(js['results'])} ODE(s)")
            for i, ode in enumerate(js["results"], 1):
                st.session_state.generated_odes.append(ode)
                st.markdown("---")
                st.subheader(f"ODE {i}")
                render_equation_block("Generated ODE", ode.get("ode", ""))
                if ode.get("solution"):
                    render_equation_block("Solution", f"y(x) = {ode['solution']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Complexity", ode.get("complexity", "N/A"))
                c2.metric("Verified", "✅" if ode.get("verified") else "❌")
                conf = ode.get("properties", {}).get("verification_confidence", 0)
                c3.metric("Confidence", f"{conf:.2%}")
                tms = ode.get("properties", {}).get("generation_time_ms", 0)
                c4.metric("Time", f"{tms:.0f} ms")
                if show_plot and ode.get("solution"):
                    fig = plot_solution(ode["solution"], x_range, params)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No results returned.")

    if st.session_state.generated_odes:
        st.markdown("### Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download all"):
                download_jsonl(st.session_state.generated_odes, "quick_generated_odes.jsonl")
        with col2:
            if st.button("🗑️ Clear cache"):
                st.session_state.generated_odes = []
                st.success("Cleared.")

def page_batch_generation():
    st.title("📦 Batch Generation")
    probe_and_load_resources()
    gens = st.session_state.available_generators
    funcs = st.session_state.available_functions
    if not gens or not funcs:
        st.error("No generators/functions available. Check API or enable USE_DEMO=1.")
        return

    tab1, tab2, tab3 = st.tabs(["Configuration", "Parameter Ranges", "Advanced"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            linear = [g for g in gens if g.startswith("L")]
            nonlinear = [g for g in gens if g.startswith("N")]
            st.markdown("**Linear Generators**")
            sel_lin = st.multiselect("", linear, default=linear[:2] if len(linear) >= 2 else linear, key="batch_lin")
            st.markdown("**Nonlinear Generators**")
            sel_non = st.multiselect(" ", nonlinear, default=nonlinear[:2] if len(nonlinear) >= 2 else nonlinear, key="batch_non")
            selected_generators = sel_lin + sel_non
        with col2:
            st.markdown("**Functions**")
            sel_funcs = st.multiselect("Choose functions", funcs, default=funcs[:5])
        samples = st.slider("Samples per combination", 1, 20, 5)
        total = len(selected_generators) * len(sel_funcs) * samples
        st.info(f"This will generate **{total:,}** ODEs")

    param_ranges: Dict[str, Any] = {}
    with tab2:
        left, right = st.columns(2)
        with left:
            param_ranges["alpha"] = st.multiselect("α (alpha)", [-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0], default=[0.0,1.0])
            param_ranges["beta"]  = st.multiselect("β (beta)",  [0.5,1.0,1.5,2.0,2.5,3.0], default=[1.0,2.0])
            param_ranges["M"]     = st.multiselect("M",         [-1.0,-0.5,0.0,0.5,1.0],   default=[0.0])
        with right:
            if any(g.startswith("N") for g in selected_generators):
                param_ranges["q"] = st.multiselect("q (power)", [2,3,4,5], default=[2,3])
                param_ranges["v"] = st.multiselect("v (power)", [2,3,4,5], default=[2,3])
            if any(g in {"L4","N6"} for g in selected_generators):
                param_ranges["a"] = st.multiselect("a (pantograph)", [2.0,2.5,3.0,3.5,4.0], default=[2.0])

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            verify = st.checkbox("Verify all", True)
            save_ds = st.checkbox("Save as dataset", True)
            ds_name = None
            if save_ds:
                ds_name = st.text_input("Dataset name", value=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with col2:
            export_fmt = st.selectbox("Export format (inline results)", ["JSONL","CSV"], index=0)

    if st.button("🚀 Start Batch", type="primary"):
        if not selected_generators or not sel_funcs:
            st.error("Please pick at least one generator and one function.")
            return
        with st.status("Starting batch…", expanded=True) as s:
            out = api.batch_generate(
                generators=selected_generators,
                functions=sel_funcs,
                samples_per_combination=samples,
                parameters=param_ranges if any(param_ranges.values()) else None,
                verify=verify,
                dataset_name=ds_name if save_ds else None,
            )
            if not out or "job_id" not in out:
                s.update(label="Batch failed to start", state="error")
                return
            s.update(label=f"Job {out['job_id']} queued")
        js = wait_for_job(out["job_id"], max_secs=3600, poll=3)
        if not js or js.get("status") != "completed":
            st.error("Batch did not complete.")
            return
        results = js.get("results", {})
        st.success(f"Done! Generated {results.get('total_generated', 0):,} ODEs")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", f"{results.get('total_generated', 0):,}")
        c2.metric("Verified", f"{results.get('verified_count', 0):,}")
        try:
            rate = (results.get('summary',{}).get('verified',0) / max(results.get('summary',{}).get('total',1),1))*100
        except Exception:
            rate = 0.0
        c3.metric("Success Rate", f"{rate:.1f}%")
        c4.metric("Avg Complexity", f"{results.get('summary',{}).get('avg_complexity',0):.1f}")

        if save_ds and results.get("dataset_info"):
            ds = results["dataset_info"]
            st.session_state.current_dataset = ds.get("name")
            dlist = api.datasets()
            st.session_state.available_datasets = dlist.get("datasets", [])
            st.info(f"Saved dataset: {ds.get('name')} ({ds.get('size',0):,} ODEs)")
        elif results.get("odes"):
            st.session_state.batch_dataset = results["odes"]
            st.markdown("### Sample")
            for i, ode in enumerate(results["odes"][:5], 1):
                with st.expander(f"Sample {i}: {ode.get('generator_name','?')} × {ode.get('function_name','?')}"):
                    render_equation_block("ODE", ode.get("ode_symbolic", ""))
                    sol = ode.get("solution_symbolic")
                    if sol:
                        render_equation_block("Solution", f"y(x) = {sol}")
            st.markdown("### Export inline results")
            if st.button("📥 Download"):
                if export_fmt == "JSONL":
                    download_jsonl(results["odes"], f"batch_odes_{len(results['odes'])}.jsonl")
                else:
                    df = pd.DataFrame(results["odes"]).to_csv(index=False)
                    b64 = base64.b64encode(df.encode()).decode()
                    st.markdown(f"<a download='batch_odes.csv' href='data:text/csv;base64,{b64}'>📥 CSV</a>", unsafe_allow_html=True)

def page_verify():
    st.title("✅ Verification")
    col1, col2 = st.columns(2)
    with col1:
        ode_txt = st.text_area("ODE (SymPy Eq(...))", value="Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))", height=120)
    with col2:
        sol_txt = st.text_area("Proposed Solution", value="pi*sin(x)", height=120)
    method = st.selectbox("Method", ["substitution", "numerical", "checkodesol"])
    if st.button("🔍 Verify", type="primary"):
        with st.status("Verifying…", expanded=False):
            res = api.verify(ode=ode_txt, solution=sol_txt, method=method)
        if res:
            if res.get("verified"):
                st.success(f"Verified ✅ — Confidence {res.get('confidence',0):.2%}")
            else:
                st.error("Not verified ❌")
            st.json(res)

    st.divider()
    st.subheader("Batch Verification (JSONL)")
    up = st.file_uploader("Upload JSONL", type=["jsonl","json"])
    if up is not None:
        text = up.read().decode("utf-8")
        lines = [l for l in text.splitlines() if l.strip()]
        items: List[Dict[str, Any]] = []
        for line in lines:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
        st.info(f"Loaded {len(items)} records")
        if st.button("Run batch verify"):
            verified = 0
            out: List[Dict[str, Any]] = []
            pb = st.progress(0)
            for i, row in enumerate(items, 1):
                ode = row.get("ode") or row.get("ode_symbolic")
                sol = row.get("solution") or row.get("solution_symbolic")
                if ode and sol:
                    res = api.verify(ode=ode, solution=sol, method=method)
                    row["verification_result"] = res
                    if res and res.get("verified"):
                        verified += 1
                out.append(row)
                pb.progress(int(i/len(items)*100))
            st.success(f"{verified}/{len(items)} verified")
            if st.button("📥 Download verified set"):
                download_jsonl(out, "verified_odes.jsonl")

def page_datasets():
    st.title("📊 Datasets")
    if st.button("🔄 Refresh"):
        d = api.datasets()
        st.session_state.available_datasets = d.get("datasets", [])
        st.success("Refreshed.")
    if not st.session_state.available_datasets:
        st.info("No datasets yet. Use Batch Generation first.")
    else:
        for ds in st.session_state.available_datasets:
            with st.expander(f"📁 {ds.get('name')} ({ds.get('size',0):,} ODEs)"):
                st.write(f"Created: {ds.get('created_at','?')}")
                st.write(f"Path: {ds.get('path','?')}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Select", key=f"sel_{ds.get('name')}"):
                        st.session_state.current_dataset = ds.get("name")
                        st.success(f"Selected {ds.get('name')}")
                with c2:
                    st.caption("Download handled by your API or storage; add a link here if available.")

    st.subheader("Create from cached batch results")
    if st.session_state.batch_dataset:
        st.info(f"You have {len(st.session_state.batch_dataset)} ODEs cached")
        new_name = st.text_input("Dataset name", value=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if st.button("💾 Save as dataset"):
            res = api.create_dataset(odes=st.session_state.batch_dataset, dataset_name=new_name)
            if res:
                st.success("Saved.")
                d = api.datasets()
                st.session_state.available_datasets = d.get("datasets", [])
                st.session_state.batch_dataset = []
    else:
        st.caption("Run a batch first to create from cache.")

    if st.session_state.current_dataset:
        st.subheader(f"Current: {st.session_state.current_dataset}")
        c1, c2, c3 = st.columns(3)
        c1.button("📈 Analyze (see Analysis tab)")
        c2.button("🔧 Preprocess (placeholder)")
        with c3:
            if st.button("🗑️ Delete (placeholder)"):
                st.warning("Implement delete endpoint in API then wire here.")

def page_ml_training():
    st.title("🤖 ML Training")
    d = api.datasets()
    datasets = [ds.get("name") for ds in d.get("datasets", [])]
    if not datasets:
        st.warning("No datasets. Create one first.")
        return
    col1, col2 = st.columns(2)
    with col1:
        ds_name = st.selectbox("Dataset", datasets, index=0)
        epochs = st.slider("Epochs", 10, 200, 50, 10)
        model_type = st.selectbox("Model Type", ["pattern_net", "transformer", "vae"], index=0, format_func=lambda x: {
            "pattern_net": "PatternNet (fast)",
            "transformer": "Transformer (powerful)",
            "vae": "VAE (generative)",
        }[x])
    with col2:
        batch_size = st.selectbox("Batch size", [16,32,64,128], index=1)
        lr = st.select_slider("Learning rate", [1e-5,1e-4,1e-3,1e-2], value=1e-3, format_func=lambda x: f"{x:.0e}")
        early = st.checkbox("Early stopping", True)

    cfg: Dict[str, Any] = {"batch_size": batch_size, "learning_rate": lr, "early_stopping": early}
    if model_type == "pattern_net":
        c1, c2 = st.columns(2)
        with c1:
            cfg["hidden_dims"] = st.multiselect("Hidden dims", [64,128,256,512], default=[256,128,64])
        with c2:
            cfg["dropout_rate"] = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    elif model_type == "transformer":
        c1, c2 = st.columns(2)
        with c1:
            cfg["d_model"] = st.selectbox("d_model", [256,512,768], index=1)
            cfg["n_heads"] = st.selectbox("Heads", [4,8,12], index=1)
        with c2:
            cfg["n_layers"] = st.slider("Layers", 2, 12, 6)
            cfg["dim_feedforward"] = st.selectbox("FF dim", [1024,2048,4096], index=1)
    else:  # VAE
        c1, c2 = st.columns(2)
        with c1:
            cfg["latent_dim"] = st.slider("Latent dim", 16, 256, 64, 16)
            cfg["hidden_dim"] = st.slider("Hidden dim", 128, 512, 256, 32)
        with c2:
            cfg["beta"] = st.slider("KL β", 0.1, 10.0, 1.0, 0.1)

    if st.button("🚀 Start Training", type="primary"):
        body = {
            "dataset": ds_name,
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": cfg.get("batch_size", 32),
            "learning_rate": cfg.get("learning_rate", 0.001),
            "early_stopping": cfg.get("early_stopping", True),
            "config": cfg,
        }
        data, err = api._request("POST", "/ml/train", json_body=body)
        if err or not data or "job_id" not in data:
            st.error(f"Could not start training: {err or 'no response'}")
            return
        js = wait_for_job(data["job_id"], max_secs=epochs*15, poll=2)
        if js and js.get("status") == "completed":
            res = js.get("results", {})
            st.success("Training complete")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Loss", f"{res.get('final_metrics',{}).get('loss',0):.4f}")
            c2.metric("Acc", f"{res.get('final_metrics',{}).get('accuracy',0):.2%}")
            c3.metric("Val loss", f"{res.get('final_metrics',{}).get('validation_loss',0):.4f}")
            c4.metric("Time", f"{res.get('training_time',0):.1f}s")
            st.session_state.available_models = api.models().get("models", [])
        else:
            st.error("Training failed or timed out.")

def page_ml_generate():
    st.title("🧪 ML Generation")
    models = api.models().get("models", [])
    if not models:
        st.warning("No models available. Train one first.")
        return
    idx = st.selectbox("Model", list(range(len(models))), format_func=lambda i: f"{models[i].get('name','model')} — {models[i].get('metadata',{}).get('model_type','?')}")
    model = models[idx]
    st.markdown(
        f"""
        <div class='info-box'>
          <b>Type:</b> {model.get('metadata',{}).get('model_type','?')} &nbsp;·&nbsp;
          <b>Dataset:</b> {model.get('metadata',{}).get('dataset','?')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("# of ODEs", 5, 100, 20, 5)
        temp = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    with col2:
        target_gen = st.selectbox("Target generator (optional)", ["Auto"] + st.session_state.available_generators)
        target_fun = st.selectbox("Target function (optional)", ["Auto"] + st.session_state.available_functions)

    with st.expander("Advanced"):
        c1, c2 = st.columns(2)
        with c1:
            cmin = st.number_input("Min complexity", 10, 1000, 50, 10)
        with c2:
            cmax = st.number_input("Max complexity", cmin, 2000, 200, 10)

    if st.button("🎨 Generate", type="primary"):
        payload: Dict[str, Any] = {"model_path": model.get("path"), "n_samples": n_samples, "temperature": temp, "complexity_range": [cmin, cmax]}
        if target_gen != "Auto":
            payload["generator"] = target_gen
        if target_fun != "Auto":
            payload["function"] = target_fun
        data, err = api._request("POST", "/ml/generate", json_body=payload)
        if err or not data or "job_id" not in data:
            st.error(f"Failed to start ML generation: {err or 'no response'}")
            return
        js = wait_for_job(data["job_id"], max_secs=900, poll=2)
        if not js or js.get("status") != "completed":
            st.error("ML generation did not complete.")
            return
        results = js.get("results", {})
        odes = results.get("odes", [])
        st.success(f"Generated {len(odes)} ODEs")
        st.session_state.ml_generated_odes.extend(odes)
        show = st.slider("Preview count", 1, max(1,len(odes)), min(10, len(odes)))
        for i, ode in enumerate(odes[:show], 1):
            with st.expander(f"#{i} — {ode.get('generator','ML')} — {'✅' if ode.get('verified') else '❔'}"):
                render_equation_block("ODE", ode.get("ode",""))
                if ode.get("solution"):
                    render_equation_block("Solution", f"y(x) = {ode['solution']}")
        if odes and st.button("📥 Download"):
            download_jsonl(odes, f"ml_generated_{len(odes)}.jsonl")

def page_analysis():
    st.title("📈 Analysis")
    source = st.radio("Source", ["Generated", "Batch cache", "ML generated", "Dataset (placeholder)"] , horizontal=True)
    if source == "Generated":
        data = st.session_state.generated_odes
    elif source == "Batch cache":
        data = st.session_state.batch_dataset
    elif source == "ML generated":
        data = st.session_state.ml_generated_odes
    else:
        st.info("To analyze a stored dataset, add an endpoint to fetch rows by name and load it here.")
        data = []
    if not data:
        st.warning("No data to analyze yet.")
        return
    df = pd.DataFrame(data)
    st.write(df.head(3))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", len(df))
    if "verified" in df:
        good = int(df["verified"].sum())
        c2.metric("Verified", f"{good} ({good/len(df)*100:.1f}%)")
    if "generator" in df:
        c3.metric("Generators", df["generator"].nunique())
    if "function" in df:
        c4.metric("Functions", df["function"].nunique())

    if "generator" in df:
        st.subheader("By generator")
        counts = df["generator"].value_counts()
        fig = px.bar(x=counts.index, y=counts.values, labels={"x":"Generator","y":"Count"}, title="Distribution by generator")
        st.plotly_chart(fig, use_container_width=True)

    if "function" in df:
        st.subheader("Top functions")
        fcounts = df["function"].value_counts().head(15)
        fig = px.pie(values=fcounts.values, names=fcounts.index, title="Top 15 functions")
        st.plotly_chart(fig, use_container_width=True)

    if "complexity" in df:
        st.subheader("Complexity distribution")
        fig = px.histogram(df, x="complexity", nbins=40)
        st.plotly_chart(fig, use_container_width=True)

    if set(["generator","verified"]).issubset(df.columns):
        st.subheader("Verification rate by generator")
        ver = df.groupby("generator")["verified"].mean().mul(100).sort_values(ascending=False)
        fig = px.bar(x=ver.index, y=ver.values, labels={"x":"Generator","y":"Verification %"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Export analysis summary")
    summary = {
        "total": int(len(df)),
        "verified": int(df["verified"].sum()) if "verified" in df else 0,
        "verification_rate": float(df["verified"].mean()) if "verified" in df else 0.0,
        "unique_generators": int(df["generator"].nunique()) if "generator" in df else 0,
        "unique_functions": int(df["function"].nunique()) if "function" in df else 0,
        "avg_complexity": float(df["complexity"].mean()) if "complexity" in df else 0.0,
    }
    st.json(summary)
    if st.button("📥 Download report"):
        b64 = base64.b64encode(json.dumps({"generated_at": datetime.now().isoformat(), "summary": summary}).encode()).decode()
        st.markdown(f"<a download='analysis_report.json' href='data:application/json;base64,{b64}'>Download JSON</a>", unsafe_allow_html=True)

def page_tools():
    st.title("🛠️ Tools & Status")
    probe_and_load_resources()
    h = st.session_state.api_status or {}
    st.markdown(
        f"<div class='info-box'><b>Status:</b> {h.get('status','?')} · <b>ML:</b> {'✅' if h.get('ml_enabled') else '❌'} · <b>Timestamp:</b> {h.get('timestamp','?')}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Raw stats")
    st.json(cached_stats())

    st.subheader("Probe endpoints")
    col1, col2 = st.columns(2)
    with col1:
        d1, e1 = api._request("GET", "/generators")
        st.code((json.dumps(d1, indent=2) if isinstance(d1, dict) else str(d1))[:800])
    with col2:
        d2, e2 = api._request("GET", "/functions")
        st.code((json.dumps(d2, indent=2) if isinstance(d2, dict) else str(d2))[:800])

    st.subheader("Prometheus metrics (first 2k chars)")
    met = api.metrics()
    if met:
        st.text(met[:2000] + ("…" if len(met) > 2000 else ""))
    else:
        st.caption("No metrics or endpoint unavailable.")

    st.subheader("Cache / Session")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear cached ODEs"):
            st.session_state.generated_odes = []
            st.session_state.batch_dataset = []
            st.session_state.ml_generated_odes = []
            st.success("Cleared.")
    with c2:
        if st.button("🔄 Reset session (soft)"):
            for k in list(st.session_state.keys()):
                if k not in {"api_status","available_generators","available_functions","api_prefix"}:
                    del st.session_state[k]
            st.success("Session reset.")

def page_docs():
    st.title("📚 Documentation")
    st.markdown(
        f"""
        **Workflow**
        1) Quick/Batch Generate → 2) Verify → 3) Create Dataset → 4) Train ML → 5) ML Generate → 6) Analysis

        **Config**
        - `API_BASE_URL` = e.g. `https://your-api.example.com`
        - `API_KEY` = your key
        - `API_PREFIX` = `/api/v1` (default) or empty for root APIs. Auto-detected if incorrect.
        - `USE_DEMO` = `1` to enable demo fallback if API is unreachable.

        **Endpoints (expected)**
        - `/health` and `/metrics` (unprefixed)
        - API operations are available **both** at `/api/v1/*` and at unprefixed aliases (server provides both).
        """
    )

# ---------- Router ----------

def main():
    header()
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Dashboard","⚡ Quick Generate","📦 Batch Generation","✅ Verification","📊 Datasets","🤖 ML Training","🧪 ML Generation","📈 Analysis","🛠️ Tools","📚 Docs"],
        )
        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")
        st.caption(f"Prefix (active): {st.session_state.api_prefix or '—'}")
        st.caption(f"Demo mode: {'ON' if USE_DEMO else 'OFF'}")

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "⚡ Quick Generate":
        page_quick_generate()
    elif page == "📦 Batch Generation":
        page_batch_generation()
    elif page == "✅ Verification":
        page_verify()
    elif page == "📊 Datasets":
        page_datasets()
    elif page == "🤖 ML Training":
        page_ml_training()
    elif page == "🧪 ML Generation":
        page_ml_generate()
    elif page == "📈 Analysis":
        page_analysis()
    elif page == "🛠️ Tools":
        page_tools()
    else:
        page_docs()

if __name__ == "__main__":
    main()
