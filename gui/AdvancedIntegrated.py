"""
ODE Master Generator — Prefix-agnostic Streamlit GUI
----------------------------------------------------

Key features:
- No baked-in /api/v1. The client auto-discovers the correct prefix
  by probing the root (/) and trying both "" and "/api/v1".
- Robust normalization for endpoint responses with different shapes.
- End-to-end pages: Dashboard, Quick Generate, Batch, Verify, Datasets,
  ML Train, ML Generate, Analysis, Tools, Docs.
- Optional demo fallback (USE_DEMO=1) if the API is unreachable.

Environment:
  API_BASE_URL=https://your-api.example.com           # no trailing slash
  API_KEY=your-key
  API_PREFIX=                                        # "", "api/v1", or "/api/v1" (optional override)
  USE_DEMO=0|1
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

# Optional SymPy for plotting the solution
try:
    import sympy as sp
except Exception:
    sp = None

# ---------- Config ----------
st.set_page_config(page_title="ODE Master Generator", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

API_BASE_URL = (os.getenv("API_BASE_URL") or "https://ode-api-production.up.railway.app").rstrip("/")
API_KEY = os.getenv("API_KEY", "test-key")
RAW_PREFIX = os.getenv("API_PREFIX", "").strip()
USE_DEMO = str(os.getenv("USE_DEMO", "0")).lower() in {"1", "true", "yes"}
REQUEST_TIMEOUT = 30

# ---------- Styles ----------
st.markdown(
    """
    <style>
      .app-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px;border-radius:14px;color:#fff;margin-bottom:18px}
      .metric-card{background:#fff;padding:14px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
      .chip{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef2ff;border:1px solid #c7d2fe}
      .ok{color:#10b981}.bad{color:#ef4444}
      .latex-box{font-size:1.05em;padding:10px;border-radius:8px;background:#f8fafc;border:1px solid #e5e7eb;overflow-x:auto}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session ----------
def _init_state():
    ss = st.session_state
    ss.setdefault("resolved_prefix", None)  # "", "/api/v1", etc
    ss.setdefault("api_status", {"status": "unknown"})
    ss.setdefault("available_generators", [])
    ss.setdefault("available_functions", [])
    ss.setdefault("generated_odes", [])
    ss.setdefault("batch_cache", [])
    ss.setdefault("available_datasets", [])
    ss.setdefault("available_models", [])
_init_state()

# ---------- API Client (prefix-agnostic) ----------

class API:
    """
    A tiny client that:
      1) Tries to discover the right prefix from GET "/" (if provided).
      2) Falls back to probing both "" and "/api/v1" on first call.
      3) Caches the working prefix in session.
    """

    def __init__(self, base: str, key: str, initial_prefix: Optional[str] = RAW_PREFIX):
        self.base = base.rstrip("/")
        self.headers = {"X-API-Key": key, "Content-Type": "application/json"}
        p = (initial_prefix or "").strip()
        if p and not p.startswith("/"):
            p = "/" + p
        self.prefix = p or (st.session_state.get("resolved_prefix") or "")
        self._probed = False

    # ---- helpers
    def _join(self, prefix: str, path: str) -> str:
        return f"{self.base}{prefix}{path}"

    def _maybe_discover(self):
        if self._probed:
            return
        self._probed = True
        # 1) Try root discovery
        try:
            r = requests.get(self._join("", "/"), timeout=6)
            if r.ok:
                data = {}
                try:
                    data = r.json()
                except Exception:
                    data = {}
                # Heuristic: if the discovery advertises "/api/v1/generate", assume "/api/v1"
                hinted = ""
                try:
                    api_map = (data or {}).get("endpoints", {}).get("api", {})
                    # flatten all string values and look for "/api/v1/"
                    values = []
                    for v in api_map.values():
                        if isinstance(v, str):
                            values.append(v)
                        elif isinstance(v, dict):
                            values.extend([x for x in v.values() if isinstance(x, str)])
                    if any("/api/v1/" in s for s in values):
                        hinted = "/api/v1"
                except Exception:
                    pass
                if hinted and not self.prefix:
                    self.prefix = hinted
        except Exception:
            pass
        # 2) If user passed API_PREFIX, keep it. Otherwise defer to first call fallback.

    def _request(self, method: str, path: str, *, json_body: Any | None = None, timeout: int = REQUEST_TIMEOUT):
        self._maybe_discover()

        # Try with current prefix first
        prefixes_to_try = [self.prefix]
        # If not set or fails with 404, we’ll try the other one
        if self.prefix == "":
            prefixes_to_try.append("/api/v1")
        elif self.prefix == "/api/v1":
            prefixes_to_try.append("")
        else:
            # Odd custom prefix? also try without and with /api/v1 as last resorts
            prefixes_to_try.extend(["", "/api/v1"])

        last_err = None
        for pfx in dict.fromkeys(prefixes_to_try):  # de-dup while keeping order
            url = self._join(pfx, path)
            try:
                r = requests.request(method, url, headers=self.headers, json=json_body, timeout=timeout)
            except Exception as e:
                last_err = f"{method} {url} -> {e}"
                continue

            if r.status_code == 404:
                last_err = f"{method} {url} -> HTTP 404"
                continue
            if r.status_code >= 400:
                return None, f"{method} {url} -> HTTP {r.status_code}: {r.text[:300]}"

            # Success -> adopt this prefix permanently
            if pfx != self.prefix:
                self.prefix = pfx
                st.session_state.resolved_prefix = pfx

            if path.endswith("/metrics"):
                return r.text, None
            try:
                return (r.json() if r.text.strip() else None), None
            except Exception:
                return None, "Non-JSON response"

        return None, last_err or "All attempts failed"

    # ---- normalized endpoints (UNPREFIXED PATHS) ----
    def health(self) -> Dict[str, Any]:
        data, err = self._request("GET", "/health", timeout=8)
        return {"status": "error", "error": err} if err else (data or {"status": "unknown"})

    def generators(self) -> List[str]:
        data, err = self._request("GET", "/generators")
        if err or data is None:
            return []
        try:
            if isinstance(data, list): return [str(x) for x in data]
            if "all" in data: return [str(x) for x in data["all"]]
            if "generators" in data: return [str(x) for x in data["generators"]]
            lin = data.get("linear", []) if isinstance(data, dict) else []
            non = data.get("nonlinear", []) if isinstance(data, dict) else []
            return [*map(str, lin), *map(str, non)]
        except Exception:
            return []

    def functions(self) -> List[str]:
        data, err = self._request("GET", "/functions")
        if err or data is None:
            return []
        try:
            if isinstance(data, list): return [str(x) for x in data]
            if "functions" in data: return [str(x) for x in data["functions"]]
            if "data" in data and "functions" in data["data"]: return [str(x) for x in data["data"]["functions"]]
            if "items" in data: return [str(x) for x in data["items"]]
        except Exception:
            pass
        return []

    def generate(self, *, generator: str, function: str, parameters: Dict[str, Any], count: int = 1, verify: bool = True):
        body = {"generator": generator, "function": function, "parameters": parameters, "count": count, "verify": verify}
        data, err = self._request("POST", "/generate", json_body=body)
        if err:
            st.error(f"Generate failed: {err}")
            return None
        return data

    def batch_generate(self, *, generators: List[str], functions: List[str], samples_per_combination: int, parameters: Optional[Dict[str, Any]] = None, verify: bool = True, dataset_name: Optional[str] = None):
        body: Dict[str, Any] = {"generators": generators, "functions": functions, "samples_per_combination": samples_per_combination, "verify": verify}
        if parameters: body["parameters"] = parameters
        if dataset_name: body["dataset_name"] = dataset_name
        data, err = self._request("POST", "/batch_generate", json_body=body)
        if err:
            st.error(f"Batch generate failed: {err}")
            return None
        return data

    def verify(self, *, ode: str, solution: str, method: str = "substitution"):
        body = {"ode": ode, "solution": solution, "method": method}
        data, err = self._request("POST", "/verify", json_body=body)
        if err:
            st.error(f"Verify failed: {err}")
            return None
        return data

    def datasets(self):
        data, _ = self._request("GET", "/datasets")
        return data or {"datasets": [], "count": 0}

    def create_dataset(self, *, odes: List[Dict[str, Any]], dataset_name: Optional[str] = None):
        body = {"odes": odes, "dataset_name": dataset_name}
        data, err = self._request("POST", "/datasets/create", json_body=body)
        if err:
            st.error(f"Create dataset failed: {err}")
            return None
        return data

    def models(self):
        data, _ = self._request("GET", "/models")
        return data or {"models": [], "count": 0}

    def stats(self):
        data, _ = self._request("GET", "/stats")
        return data or {}

    def metrics(self) -> str:
        data, _ = self._request("GET", "/metrics")
        return data or ""

    def job_status(self, job_id: str):
        data, err = self._request("GET", f"/jobs/{job_id}")
        if err:
            st.error(f"Job status failed: {err}")
            return None
        return data


api = API(API_BASE_URL, API_KEY, initial_prefix=RAW_PREFIX)

# ---------- Demo fallback ----------
DEMO_GENERATORS = ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
DEMO_FUNCTIONS  = ["sine", "cosine", "tangent_safe", "exponential", "exp_scaled", "quadratic", "cubic", "sinh", "cosh", "tanh", "log_safe"]

# ---------- Cache wrappers ----------
@st.cache_data(ttl=60)
def cached_generators() -> List[str]: return api.generators()

@st.cache_data(ttl=60)
def cached_functions() -> List[str]: return api.functions()

@st.cache_data(ttl=30)
def cached_stats() -> Dict[str, Any]: return api.stats()

# ---------- Helpers ----------
def header():
    st.markdown("""<div class="app-header"><h2 style="margin:0">🔬 ODE Master Generator</h2>
    <div style="opacity:.9">Generate, verify, and analyze ODEs — end to end.</div></div>""", unsafe_allow_html=True)

def chip(ok: bool) -> str:
    return f"<span class='chip {'ok' if ok else 'bad'}'>{'API Online' if ok else 'API Offline'}</span>"

def probe_and_load():
    if st.session_state.api_status.get("status") == "unknown":
        st.session_state.api_status = api.health()
    gens = cached_generators()
    funcs = cached_functions()
    if (not gens or not funcs) and USE_DEMO:
        gens = gens or DEMO_GENERATORS
        funcs = funcs or DEMO_FUNCTIONS
    st.session_state.available_generators = gens
    st.session_state.available_functions = funcs

def render_block(title: str, expr: str):
    st.markdown(f"**{title}**")
    st.markdown(f"<div class='latex-box'>{expr}</div>", unsafe_allow_html=True)

def plot_solution(expr: str, x_range=(-5.0, 5.0), params: Optional[Dict[str, Any]] = None):
    if not sp:
        st.info("Sympy not available in this runtime.")
        return None
    try:
        x = sp.Symbol("x")
        e = sp.sympify(expr)
        if params:
            for k, v in params.items():
                try: e = e.subs(sp.Symbol(str(k)), v)
                except Exception: pass
        f = sp.lambdify(x, e, "numpy")
        xs = np.linspace(x_range[0], x_range[1], 600)
        ys = np.real(np.nan_to_num(f(xs), nan=np.nan))
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="y(x)"))
        fig.update_layout(template="plotly_white", height=420, title="Solution plot", xaxis_title="x", yaxis_title="y(x)")
        return fig
    except Exception as e:
        st.info(f"Plotting skipped: {e}")
        return None

def wait_for_job(job_id: str, *, max_secs: int = 900, poll: float = 1.5) -> Optional[Dict[str, Any]]:
    start = time.time(); bar = st.progress(0); info = st.empty()
    while time.time() - start < max_secs:
        js = api.job_status(job_id)
        if js:
            status = js.get("status", "unknown")
            bar.progress(int(js.get("progress", 0)))
            meta = js.get("metadata", {})
            if meta:
                if "current_epoch" in meta: info.text(f"{status} — epoch {meta['current_epoch']}/{meta.get('total_epochs','?')}")
                elif "current" in meta and "total" in meta: info.text(f"{status} — {meta['current']}/{meta['total']}")
                else: info.text(status)
            if status == "completed": bar.progress(100); return js
            if status == "failed": st.error(js.get("error", "Job failed")); return None
        time.sleep(poll)
    st.error("Job timed out")
    return None

def download_jsonl(rows: List[Dict[str, Any]], name="dataset.jsonl"):
    payload = "\n".join(json.dumps(o) for o in rows)
    b64 = base64.b64encode(payload.encode()).decode()
    st.markdown(f"<a download='{name}' href='data:application/json;base64,{b64}'>📥 Download {name}</a>", unsafe_allow_html=True)

# ---------- Pages ----------
def page_dashboard():
    st.title("Dashboard")
    probe_and_load()
    online = (st.session_state.api_status or {}).get("status") in {"healthy", "ok", "operational"}
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'>{chip(online)}</div>", unsafe_allow_html=True)
    stats = cached_stats()
    with c2: st.metric("Generated (24h)", f"{stats.get('total_generated_24h', 0):,}")
    with c3: st.metric("Verification Rate", f"{stats.get('verification_success_rate', 0):.1%}")
    with c4: st.metric("Active Jobs", stats.get("active_jobs", 0))
    st.subheader("Recent ODEs")
    if not st.session_state.generated_odes:
        st.info("No recent ODEs yet. Try Quick Generate.")
    else:
        for i, ode in enumerate(st.session_state.generated_odes[-6:][::-1], 1):
            with st.expander(f"{i}. {ode.get('generator','?')} × {ode.get('function','?')} — {'✅' if ode.get('verified') else '❔'}"):
                render_block("ODE", ode.get("ode", ode.get("ode_symbolic","")))
                sol = ode.get("solution") or ode.get("solution_symbolic")
                if sol: render_block("Solution", f"y(x) = {sol}")

def params_controls(generator: str, key_prefix: str = "") -> Dict[str, Any]:
    col1, col2 = st.columns(2); p: Dict[str, Any] = {}
    with col1:
        p["alpha"] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1, key=f"{key_prefix}_alpha")
        p["beta"]  = st.slider("β (beta)", 0.1, 3.0, 1.0, 0.1, key=f"{key_prefix}_beta")
    with col2:
        p["M"] = st.slider("M", -1.0, 1.0, 0.0, 0.1, key=f"{key_prefix}_M")
        if generator.startswith("N"):
            p["q"] = st.slider("q (power)", 2, 5, 2, key=f"{key_prefix}_q")
            if generator in {"N2", "N3", "N6", "N7"}:
                p["v"] = st.slider("v (power)", 2, 5, 3, key=f"{key_prefix}_v")
        if generator in {"L4", "N6"}:
            p["a"] = st.slider("a (pantograph)", 2.0, 5.0, 2.0, 0.5, key=f"{key_prefix}_a")
    return p

def page_quick_generate():
    st.title("⚡ Quick Generate")
    probe_and_load()
    gens, funcs = st.session_state.available_generators, st.session_state.available_functions
    if not gens or not funcs:
        st.error("No generators/functions available. Check API or enable USE_DEMO=1.")
        return
    c1, c2 = st.columns(2)
    with c1: generator = st.selectbox("Generator", gens, index=0)
    with c2: function  = st.selectbox("Function", funcs, index=min(0, len(funcs)-1))
    st.markdown("### Parameters")
    params = params_controls(generator, key_prefix="quick")
    with st.expander("Advanced"):
        colA, colB = st.columns(2)
        with colA:
            verify = st.checkbox("Verify solution", True)
            count  = st.number_input("Number of ODEs", 1, 10, 1)
        with colB:
            show_plot = st.checkbox("Plot solution (if available)", True)
            x_range  = st.slider("Plot range", -10.0, 10.0, (-5.0, 5.0))
    if st.button("🚀 Generate", type="primary"):
        start = api.generate(generator=generator, function=function, parameters=params, count=count, verify=verify)
        if not start or "job_id" not in start:
            st.error("Failed to start job"); return
        js = wait_for_job(start["job_id"], max_secs=600)
        if js and js.get("results"):
            st.success(f"Generated {len(js['results'])} ODE(s)")
            for ode in js["results"]:
                st.session_state.generated_odes.append(ode)
                st.markdown("---")
                render_block("Generated ODE", ode.get("ode",""))
                if ode.get("solution"): render_block("Solution", f"y(x) = {ode['solution']}")
                if show_plot and ode.get("solution"):
                    fig = plot_solution(ode["solution"], x_range, params)
                    if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No results returned.")

def page_batch_generation():
    st.title("📦 Batch Generation")
    probe_and_load()
    gens, funcs = st.session_state.available_generators, st.session_state.available_functions
    if not gens or not funcs:
        st.error("No generators/functions available. Check API or enable USE_DEMO=1.")
        return
    left, right = st.columns(2)
    with left:
        linear    = [g for g in gens if g.startswith("L")]
        nonlinear = [g for g in gens if g.startswith("N")]
        st.markdown("**Linear Generators**")
        sel_lin = st.multiselect("", linear, default=linear[:2] or linear)
        st.markdown("**Nonlinear Generators**")
        sel_non = st.multiselect(" ", nonlinear, default=nonlinear[:2] or nonlinear)
        selected_g = sel_lin + sel_non
    with right:
        sel_funcs = st.multiselect("Functions", funcs, default=funcs[:5])
    samples = st.slider("Samples per combination", 1, 20, 5)
    total = len(selected_g) * len(sel_funcs) * samples
    st.info(f"This will generate **{total:,}** ODEs")
    st.markdown("### Parameter Ranges")
    pr: Dict[str, Any] = {}
    c1, c2 = st.columns(2)
    with c1:
        pr["alpha"] = st.multiselect("α", [-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0], default=[0,1.0])
        pr["beta"]  = st.multiselect("β", [0.5,1.0,1.5,2.0,2.5,3.0], default=[1.0,2.0])
        pr["M"]     = st.multiselect("M", [-1.0,-0.5,0,0.5,1.0], default=[0])
    with c2:
        if any(g.startswith("N") for g in selected_g):
            pr["q"] = st.multiselect("q", [2,3,4,5], default=[2,3])
            pr["v"] = st.multiselect("v", [2,3,4,5], default=[2,3])
        if any(g in {"L4","N6"} for g in selected_g):
            pr["a"] = st.multiselect("a", [2.0,2.5,3.0,3.5,4.0], default=[2.0])
    st.markdown("### Advanced")
    colA, colB = st.columns(2)
    with colA:
        verify = st.checkbox("Verify all", True)
        save_ds = st.checkbox("Save as dataset", True)
        ds_name = st.text_input("Dataset name", value=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}") if save_ds else None
    with colB:
        export_fmt = st.selectbox("Export format (inline results)", ["JSONL","CSV"], index=0)
    if st.button("🚀 Start Batch", type="primary"):
        if not selected_g or not sel_funcs:
            st.error("Pick at least one generator and one function."); return
        out = api.batch_generate(generators=selected_g, functions=sel_funcs, samples_per_combination=samples, parameters=pr if any(pr.values()) else None, verify=verify, dataset_name=ds_name if save_ds else None)
        if not out or "job_id" not in out:
            st.error("Batch failed to start"); return
        js = wait_for_job(out["job_id"], max_secs=3600, poll=3)
        if not js or js.get("status") != "completed":
            st.error("Batch did not complete."); return
        results = js.get("results", {})
        st.success(f"Done! Generated {results.get('total_generated', 0):,} ODEs")
        if save_ds and results.get("dataset_info"):
            ds = results["dataset_info"]
            st.session_state.available_datasets.append(ds)
            st.info(f"Saved dataset: {ds.get('name')} ({ds.get('size',0):,} ODEs)")
        elif results.get("odes"):
            st.session_state.batch_cache = results["odes"]
            st.markdown("### Sample")
            for i, ode in enumerate(results["odes"][:5], 1):
                with st.expander(f"Sample {i}: {ode.get('generator_name','?')} × {ode.get('function_name','?')}"):
                    render_block("ODE", ode.get("ode_symbolic",""))
                    sol = ode.get("solution_symbolic")
                    if sol: render_block("Solution", f"y(x) = {sol}")
            if st.button("📥 Download"):
                if export_fmt == "JSONL":
                    download_jsonl(results["odes"], f"batch_odes_{len(results['odes'])}.jsonl")
                else:
                    df = pd.DataFrame(results["odes"]).to_csv(index=False)
                    b64 = base64.b64encode(df.encode()).decode()
                    st.markdown(f"<a download='batch_odes.csv' href='data:text/csv;base64,{b64}'>CSV</a>", unsafe_allow_html=True)

def page_verify():
    st.title("✅ Verification")
    col1, col2 = st.columns(2)
    with col1:
        ode_txt = st.text_area("ODE (SymPy Eq(...))", "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))", height=120)
    with col2:
        sol_txt = st.text_area("Proposed Solution", "pi*sin(x)", height=120)
    method = st.selectbox("Method", ["substitution","numerical","checkodesol"])
    if st.button("🔍 Verify", type="primary"):
        res = api.verify(ode=ode_txt, solution=sol_txt, method=method)
        if res:
            if res.get("verified"): st.success(f"Verified ✅ — Confidence {res.get('confidence',0):.2%}")
            else: st.error("Not verified ❌")
            st.json(res)

def page_datasets():
    st.title("📊 Datasets")
    if st.button("🔄 Refresh"):
        d = api.datasets()
        st.session_state.available_datasets = d.get("datasets", [])
        st.success("Refreshed.")
    if not st.session_state.available_datasets:
        st.info("No datasets yet. Use Batch Generation first.")
        return
    for ds in st.session_state.available_datasets:
        with st.expander(f"📁 {ds.get('name')} ({ds.get('size',0):,} ODEs)"):
            st.write(f"Created: {ds.get('created_at','?')}")
            st.write(f"Path: {ds.get('path','?')}")

def page_ml_training():
    st.title("🤖 ML Training")
    d = api.datasets(); datasets = [ds.get("name") for ds in d.get("datasets", [])]
    if not datasets: st.warning("No datasets. Create one first."); return
    col1, col2 = st.columns(2)
    with col1:
        ds_name = st.selectbox("Dataset", datasets, index=0)
        epochs  = st.slider("Epochs", 10, 200, 50, 10)
        model_type = st.selectbox("Model Type", ["pattern_net","transformer","vae"], index=0)
    with col2:
        batch_size = st.selectbox("Batch size", [16,32,64,128], index=1)
        lr         = st.select_slider("Learning rate", [1e-5,1e-4,1e-3,1e-2], value=1e-3, format_func=lambda x: f"{x:.0e}")
        early      = st.checkbox("Early stopping", True)
    cfg = {"batch_size": batch_size, "learning_rate": lr, "early_stopping": early}
    if model_type == "transformer":
        c1, c2 = st.columns(2)
        with c1: cfg["d_model"] = st.selectbox("d_model", [256,512,768], index=1); cfg["n_heads"] = st.selectbox("Heads", [4,8,12], index=1)
        with c2: cfg["n_layers"] = st.slider("Layers", 2, 12, 6); cfg["dim_feedforward"] = st.selectbox("FF dim", [1024,2048,4096], index=1)
    elif model_type == "vae":
        c1, c2 = st.columns(2)
        with c1: cfg["latent_dim"] = st.slider("Latent dim", 16, 256, 64, 16); cfg["hidden_dim"] = st.slider("Hidden dim", 128, 512, 256, 32)
        with c2: cfg["beta"] = st.slider("KL β", 0.1, 10.0, 1.0, 0.1)
    if st.button("🚀 Start Training", type="primary"):
        body = {"dataset": ds_name, "model_type": model_type, "epochs": epochs, "batch_size": cfg.get("batch_size", 32), "learning_rate": cfg.get("learning_rate", 0.001), "early_stopping": cfg.get("early_stopping", True), "config": cfg}
        data, err = api._request("POST", "/ml/train", json_body=body)
        if err or not data or "job_id" not in data:
            st.error(f"Could not start training: {err or 'no response'}"); return
        js = wait_for_job(data["job_id"], max_secs=epochs*15, poll=2)
        if js and js.get("status") == "completed": st.success("Training complete"); st.json(js.get("results", {}))
        else: st.error("Training failed or timed out.")

def page_ml_generate():
    st.title("🧪 ML Generation")
    models = api.models().get("models", [])
    if not models: st.warning("No models available. Train one first."); return
    idx = st.selectbox("Model", list(range(len(models))), format_func=lambda i: models[i].get("name","model"))
    model = models[idx]
    st.info(f"{model.get('name')} — {model.get('metadata',{}).get('model_type','?')}")
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("# of ODEs", 5, 100, 20, 5)
        temp      = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    with col2:
        target_gen = st.text_input("Target generator (optional)")
        target_fun = st.text_input("Target function (optional)")
    if st.button("🎨 Generate", type="primary"):
        payload: Dict[str, Any] = {"model_path": model.get("path"), "n_samples": n_samples, "temperature": temp}
        if target_gen: payload["generator"] = target_gen
        if target_fun: payload["function"]  = target_fun
        data, err = api._request("POST", "/ml/generate", json_body=payload)
        if err or not data or "job_id" not in data:
            st.error(f"Failed to start ML generation: {err or 'no response'}"); return
        js = wait_for_job(data["job_id"], max_secs=900, poll=2)
        if not js or js.get("status") != "completed":
            st.error("ML generation did not complete."); return
        results = js.get("results", {}); odes = results.get("odes", [])
        st.success(f"Generated {len(odes)} ODEs"); st.write(pd.DataFrame(odes).head())

def page_analysis():
    st.title("📈 Analysis")
    data = st.session_state.generated_odes or st.session_state.batch_cache
    if not data: st.warning("No data to analyze yet."); return
    df = pd.DataFrame(data); st.write(df.head(3))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", len(df))
    if "verified" in df: c2.metric("Verified", f"{int(df['verified'].sum())} ({df['verified'].mean()*100:.1f}%)")
    if "generator" in df: c3.metric("Generators", df["generator"].nunique())
    if "generator" in df:
        st.subheader("By generator")
        counts = df["generator"].value_counts()
        st.plotly_chart(px.bar(x=counts.index, y=counts.values, labels={"x":"Generator","y":"Count"}), use_container_width=True)

def page_tools():
    st.title("🛠️ Tools & Status")
    probe_and_load()
    h = st.session_state.api_status or {}
    resolved = st.session_state.get("resolved_prefix")
    st.markdown(f"<div class='chip'>Resolved prefix: <b>{resolved if resolved is not None else '(undetermined)'}</b></div>", unsafe_allow_html=True)
    st.subheader("Raw stats")
    st.json(cached_stats())
    st.subheader("Probe endpoints (first 800 chars)")
    d1, e1 = api._request("GET", "/generators"); st.code((json.dumps(d1, indent=2) if isinstance(d1, dict) else str(d1))[:800])
    d2, e2 = api._request("GET", "/functions");  st.code((json.dumps(d2, indent=2) if isinstance(d2, dict) else str(d2))[:800])
    st.subheader("Prometheus metrics (first 2k chars)")
    met = api.metrics()
    st.text(met[:2000] + ("…" if len(met) > 2000 else ""))

def page_docs():
    st.title("📚 Documentation")
    st.markdown("""
**No baked-in prefix**
- The GUI probes `/` then tries both `""` and `"/api/v1"` on first call.
- Optionally set `API_PREFIX` to force a prefix.

**Server**
- Default routes live at `/...`
- Set `ADD_V1_ALIAS=1` on the server to additionally expose `/api/v1/...` aliases.
""")

# ---------- Router ----------
def main():
    header()
    with st.sidebar:
        page = st.radio("Go to", ["🏠 Dashboard","⚡ Quick Generate","📦 Batch Generation","✅ Verification","📊 Datasets","🤖 ML Training","🧪 ML Generation","📈 Analysis","🛠️ Tools","📚 Docs"])
        st.markdown("---"); st.caption(f"API: {API_BASE_URL}")
        st.caption(f"Resolved prefix: {st.session_state.get('resolved_prefix', '(auto)')}")
        st.caption(f"Demo mode: {'ON' if USE_DEMO else 'OFF'}")
    if page == "🏠 Dashboard": page_dashboard()
    elif page == "⚡ Quick Generate": page_quick_generate()
    elif page == "📦 Batch Generation": page_batch_generation()
    elif page == "✅ Verification": page_verify()
    elif page == "📊 Datasets": page_datasets()
    elif page == "🤖 ML Training": page_ml_training()
    elif page == "🧪 ML Generation": page_ml_generate()
    elif page == "📈 Analysis": page_analysis()
    else: page_docs()

if __name__ == "__main__":
    main()
