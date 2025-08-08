# gui/integrated_interface.py
"""
ODE Master Generator — Integrated Streamlit Interface
- Generation (single & batch)
- Verification
- Dataset management
- ML training & ML-based generation
- Analysis & system tools
Designed for Railway deployment.
"""

from __future__ import annotations

import os
import json
import time
import base64
import logging
import asyncio
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import aiohttp
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px

# --------------------------- #
#        Page Settings        #
# --------------------------- #

st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- #
#       Global Settings       #
# --------------------------- #

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ode-ui")

API_BASE_URL = os.getenv("API_BASE_URL", "https://ode-api-production.up.railway.app").rstrip("/")
API_KEY = os.getenv("API_KEY", "test-key")

# --------------------------- #
#           Styles            #
# --------------------------- #

st.markdown(
    """
<style>
.stApp { background-color: #f8fafc; }
.block-container { padding-top: 1.5rem; }
.section { margin: 1.0rem 0 0.75rem 0; font-weight: 700; font-size: 1.05rem; }
.card { background: #fff; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.metric-card { background:#fff;border-radius:12px;padding:1rem;text-align:center;box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.badge { display:inline-block;padding:.2rem .5rem;border-radius:999px;font-size:.80rem; }
.badge-ok { background:#dcfce7;color:#166534;border:1px solid #22c55e; }
.badge-bad { background:#fee2e2;color:#991b1b;border:1px solid #ef4444; }
.small { color:#64748b;font-size:.85rem; }
.latex { font-family: 'Computer Modern', 'Times New Roman', serif; background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:.75rem; overflow-x:auto; }
</style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- #
#         API Client          #
# --------------------------- #

class ODEAPI:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.timeout = timeout

    # ---- async health ---- #
    async def health(self) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    f"{self.base}/health",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=7),
                ) as r:
                    if r.status == 200:
                        return await r.json()
                    return {"status": "error", "message": f"status {r.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ---- basic GETs ---- #
    def _get(self, path: str, err: str, ok_default=None):
        try:
            r = requests.get(f"{self.base}{path}", headers=self.headers, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error("%s: %s", err, e)
            return ok_default

    def get_generators(self) -> Dict[str, Any]:
        return self._get("/api/v1/generators", "get_generators failed", {"all": []})

    def get_functions(self) -> Dict[str, Any]:
        return self._get("/api/v1/functions", "get_functions failed", {"functions": []})

    def get_models(self) -> Dict[str, Any]:
        return self._get("/api/v1/models", "get_models failed", {"models": []})

    def list_datasets(self) -> Dict[str, Any]:
        return self._get("/api/v1/datasets", "list_datasets failed", {"datasets": []})

    def get_statistics(self) -> Dict[str, Any]:
        return self._get("/api/v1/stats", "get_statistics failed", {})

    def get_metrics(self) -> str:
        try:
            r = requests.get(f"{self.base}/metrics", headers={"X-API-Key": self.headers["X-API-Key"]}, timeout=12)
            r.raise_for_status()
            return r.text
        except Exception as e:
            log.error("get_metrics failed: %s", e)
            return ""

    def get_job_status(self, job_id: str) -> Dict[str, Any] | None:
        try:
            r = requests.get(f"{self.base}/api/v1/jobs/{job_id}", headers=self.headers, timeout=12)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error("get_job_status failed: %s", e)
            return {"status": "error", "error": str(e)}

    # ---- POSTs ---- #
    def _post(self, path: str, payload: Dict[str, Any], err: str):
        try:
            r = requests.post(f"{self.base}{path}", json=payload, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error("%s: %s", err, e)
            st.error(f"API Error: {e}")
            return None

    def generate(self, generator: str, function: str, parameters: Dict, count=1, verify=True):
        return self._post(
            "/api/v1/generate",
            {"generator": generator, "function": function, "parameters": parameters, "count": count, "verify": verify},
            "generate failed",
        )

    def batch_generate(
        self,
        generators: List[str],
        functions: List[str],
        samples_per_combination: int,
        parameters: Optional[Dict] = None,
        verify: bool = True,
        dataset_name: Optional[str] = None,
    ):
        payload = {
            "generators": generators,
            "functions": functions,
            "samples_per_combination": samples_per_combination,
            "verify": verify,
        }
        if parameters:
            payload["parameters"] = parameters
        if dataset_name:
            payload["dataset_name"] = dataset_name
        return self._post("/api/v1/batch_generate", payload, "batch_generate failed")

    def verify(self, ode: str, solution: str, method="substitution"):
        return self._post("/api/v1/verify", {"ode": ode, "solution": solution, "method": method}, "verify failed")

    def create_dataset(self, odes: List[Dict], dataset_name: Optional[str] = None):
        return self._post("/api/v1/datasets/create", {"odes": odes, "dataset_name": dataset_name}, "create_dataset failed")

    def train_ml(self, dataset: str, model_type: str, epochs: int, config: Dict[str, Any]):
        payload = {
            "dataset": dataset,
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": config.get("batch_size", 32),
            "learning_rate": config.get("learning_rate", 1e-3),
            "early_stopping": config.get("early_stopping", True),
            "config": config,
        }
        return self._post("/api/v1/ml/train", payload, "train_ml failed")

    def ml_generate(self, model_path: str, n_samples: int, temperature: float = 0.8, **kwargs):
        payload = {"model_path": model_path, "n_samples": n_samples, "temperature": temperature, **kwargs}
        return self._post("/api/v1/ml/generate", payload, "ml_generate failed")


@st.cache_resource(show_spinner=False)
def get_api() -> ODEAPI:
    return ODEAPI(API_BASE_URL, API_KEY)


# --------------------------- #
#       Utility Helpers       #
# --------------------------- #

def run_async(coro):
    """Run an async function safely even if a loop exists."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

def wait_for_job(job_id: str, max_attempts=300, poll_interval=2) -> Optional[Dict[str, Any]]:
    api = get_api()
    bar = st.progress(0)
    info = st.empty()
    for i in range(max_attempts):
        js = api.get_job_status(job_id)
        if not js:
            time.sleep(poll_interval)
            continue
        status = js.get("status", "unknown")
        prog = int(js.get("progress", 0))
        bar.progress(min(prog, 100))
        meta = js.get("metadata") or {}
        msg = meta.get("status") or status
        info.text(f"Status: {msg}")
        if status == "completed":
            bar.progress(100)
            return js
        if status == "failed":
            st.error(js.get("error", "Job failed"))
            return None
        time.sleep(poll_interval)
    st.error("Job timed out.")
    return None

def latex_box(title: str, text: str):
    st.markdown(f"**{title}**")
    st.markdown(f'<div class="latex">{text}</div>', unsafe_allow_html=True)

def format_eq(eq: str) -> str:
    if not eq:
        return ""
    try:
        # quick prettify for common Derivative patterns
        s = eq.replace("Derivative(y(x), (x, 3))", "y'''(x)")
        s = s.replace("Derivative(y(x), x, 2)", "y''(x)")
        s = s.replace("Derivative(y(x), (x, 2))", "y''(x)")
        s = s.replace("Derivative(y(x), x)", "y'(x)")
        if "Eq(" in s:
            inner = s[3:-1] if s.endswith(")") else s[3:]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                s = f"{parts[0].strip()} = {parts[1].strip()}"
        return s
    except Exception:
        return eq

def plot_solution(solution_str: str, x_range: Tuple[float, float] = (-5, 5), params: Dict[str, float] | None = None):
    try:
        import sympy as sp
        x = sp.Symbol("x")
        expr = sp.sympify(solution_str)
        if params:
            for k, v in params.items():
                expr = expr.subs(k, v)
        f = sp.lambdify(x, expr, "numpy")
        xs = np.linspace(x_range[0], x_range[1], 800)
        ys = np.real(f(xs))
        ys = np.where(np.abs(ys) > 1e10, np.nan, ys)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y(x)"))
        fig.update_layout(template="plotly_white", height=380, title="Solution Plot", xaxis_title="x", yaxis_title="y(x)")
        return fig
    except Exception as e:
        log.warning("plot_solution failed: %s", e)
        return None

def jsonl_download(data: List[Dict], filename: str, label: str):
    buf = "\n".join(json.dumps(x) for x in data).encode()
    b64 = base64.b64encode(buf).decode()
    st.markdown(f'<a href="data:application/jsonl;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

def csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    st.markdown(f'<a href="data:text/csv;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

# --------------------------- #
#        Session State        #
# --------------------------- #

for key, default in {
    "generated_odes": [],
    "batch_dataset": [],
    "ml_generated_odes": [],
    "current_dataset": None,
    "available_datasets": [],
    "available_models": [],
    "available_generators": [],
    "available_functions": [],
    "api_status": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------- #
#         Sidebar / UX        #
# --------------------------- #

st.markdown(
    """
<div class="card" style="background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 100%);color:white;margin-bottom:1rem;">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div><h2 style="margin:.2rem 0;">🔬 ODE Master Generator</h2><div class="small" style="opacity:.9;">Integrated Generation • Verification • ML</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Connection")
    api = get_api()
    if st.session_state["api_status"] is None:
        st.session_state["api_status"] = run_async(api.health())

    healthy = (st.session_state["api_status"] or {}).get("status") == "healthy"
    st.markdown(
        f'<span class="badge {"badge-ok" if healthy else "badge-bad"}">'
        f'API {"Online" if healthy else "Offline"}</span>',
        unsafe_allow_html=True,
    )
    if not healthy and st.button("Retry"):
        st.session_state["api_status"] = run_async(api.health())
        st.experimental_rerun()

    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "⚡ Quick Generate",
            "📦 Batch Generation",
            "✅ Verification",
            "📊 Dataset Management",
            "🤖 ML Training",
            "🧪 ML Generation",
            "📈 Analysis",
            "🛠️ System Tools",
            "📚 Docs",
        ],
        label_visibility="collapsed",
    )

# preload lists once
if not st.session_state["available_generators"]:
    st.session_state["available_generators"] = api.get_generators().get("all", [])
if not st.session_state["available_functions"]:
    st.session_state["available_functions"] = api.get_functions().get("functions", [])
if not st.session_state["available_datasets"]:
    st.session_state["available_datasets"] = api.list_datasets().get("datasets", [])
if not st.session_state["available_models"]:
    st.session_state["available_models"] = api.get_models().get("models", [])

# --------------------------- #
#        Page: Dashboard      #
# --------------------------- #

def page_dashboard():
    st.subheader("System Dashboard")
    stats = api.get_statistics() or {}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="small">System</div>'
                    f'<h3>{"🟢 ONLINE" if stats.get("status") == "operational" else "🔴 OFFLINE"}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="small">Generated (24h)</div>'
                    f'<h3>{stats.get("total_generated_24h", 0):,}</h3></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="small">Verification Rate</div>'
                    f'<h3>{stats.get("verification_success_rate", 0):.1%}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="small">Active Jobs</div>'
                    f'<h3>{stats.get("active_jobs", 0)}</h3></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">Recent Activity</div>', unsafe_allow_html=True)
    if st.session_state["generated_odes"]:
        for i, ode in enumerate(st.session_state["generated_odes"][-5:][::-1], 1):
            with st.expander(f"Recent #{i} — {ode.get('generator','?')} / {ode.get('function','?')}", expanded=(i==1)):
                latex_box("ODE", format_eq(ode.get("ode", "")))
                if ode.get("solution"):
                    latex_box("Solution", f"y(x) = {ode['solution']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity", ode.get("complexity", "N/A"))
                c2.metric("Verified", "✅" if ode.get("verified") else "❌")
                c3.metric("Time (ms)", f"{ode.get('properties',{}).get('generation_time_ms', 0):.0f}")
    else:
        st.info("No ODEs yet — try Quick Generate.")

# --------------------------- #
#     Page: Quick Generate    #
# --------------------------- #

def param_controls(key_prefix: str, generator: str) -> Dict[str, float]:
    p: Dict[str, float] = {}
    c1, c2 = st.columns(2)
    with c1:
        p["alpha"] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1, key=f"{key_prefix}_alpha")
        p["beta"] = st.slider("β (beta)", 0.1, 3.0, 1.0, 0.1, key=f"{key_prefix}_beta")
    with c2:
        p["M"] = st.slider("M", -1.0, 1.0, 0.0, 0.1, key=f"{key_prefix}_M")
        if generator.startswith("N"):
            p["q"] = st.slider("q (power)", 2, 5, 2, key=f"{key_prefix}_q")
            if generator in ["N2", "N3", "N6", "N7"]:
                p["v"] = st.slider("v (power)", 2, 5, 3, key=f"{key_prefix}_v")
        if generator in ["L4", "N6"]:
            p["a"] = st.slider("a (pantograph)", 2.0, 5.0, 2.0, 0.5, key=f"{key_prefix}_a")
    return p

def page_quick_generate():
    st.subheader("⚡ Quick ODE Generation")
    gens = st.session_state["available_generators"]
    funcs = st.session_state["available_functions"]
    if not gens or not funcs:
        st.error("No generators/functions available. Check API.")
        return

    c1, c2 = st.columns(2)
    with c1:
        generator = st.selectbox("Generator", gens, format_func=lambda x: f"{x} ({'Linear' if x.startswith('L') else 'Nonlinear'})")
    with c2:
        function = st.selectbox("Function", funcs)

    st.markdown('<div class="section">Parameters</div>', unsafe_allow_html=True)
    params = param_controls("quick", generator)

    with st.expander("Advanced"):
        verify = st.checkbox("Verify solution", True)
        count = st.number_input("Number of ODEs", 1, 10, 1)
        show_plot = st.checkbox("Plot solution", True)
        x_range = st.slider("Plot x-range", -10.0, 10.0, (-5.0, 5.0))

    if st.button("🚀 Generate"):
        with st.spinner("Generating..."):
            res = api.generate(generator, function, params, count=count, verify=verify)
            if res and "job_id" in res:
                done = wait_for_job(res["job_id"])
                if done and done.get("results"):
                    out = done["results"]
                    st.success(f"Generated {len(out)} ODE(s).")
                    for i, ode in enumerate(out, 1):
                        st.session_state["generated_odes"].append(ode)
                        st.markdown("---")
                        st.markdown(f"**Result #{i}**")
                        latex_box("ODE", format_eq(ode.get("ode", "")))
                        if ode.get("solution"):
                            latex_box("Solution", f"y(x) = {ode['solution']}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Complexity", ode.get("complexity", "N/A"))
                        c2.metric("Verified", "✅" if ode.get("verified") else "❌")
                        c3.metric("Confidence", f"{ode.get('properties',{}).get('verification_confidence',0):.2%}")
                        c4.metric("Time (ms)", f"{ode.get('properties',{}).get('generation_time_ms',0):.0f}")
                        if show_plot and ode.get("solution"):
                            fig = plot_solution(ode["solution"], x_range, params)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Plot unavailable for this solution.")

    if st.session_state["generated_odes"]:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Download generated"):
                jsonl_download(st.session_state["generated_odes"], "quick_generated_odes.jsonl", "Download JSONL")
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state["generated_odes"] = []
                st.success("Cleared.")

# --------------------------- #
#     Page: Batch Generate    #
# --------------------------- #

def page_batch_generation():
    st.subheader("📦 Batch ODE Generation")
    gens = st.session_state["available_generators"]
    funcs = st.session_state["available_functions"]
    if not gens or not funcs:
        st.error("No generators/functions available. Check API.")
        return

    c1, c2 = st.columns(2)
    with c1:
        lin = [g for g in gens if g.startswith("L")]
        nonlin = [g for g in gens if g.startswith("N")]
        sel_lin = st.multiselect("Linear Generators", lin, default=lin[: min(2, len(lin))])
        sel_nonlin = st.multiselect("Nonlinear Generators", nonlin, default=nonlin[: min(2, len(nonlin))])
        selected_gens = sel_lin + sel_nonlin
    with c2:
        selected_funcs = st.multiselect("Functions", funcs, default=funcs[: min(6, len(funcs))])

    samples_per = st.slider("Samples per (generator,function)", 1, 20, 5)
    st.info(f"Planned: **{len(selected_gens) * len(selected_funcs) * samples_per:,}** ODEs.")

    st.markdown('<div class="section">Parameter Grid</div>', unsafe_allow_html=True)
    param_ranges: Dict[str, List] = {}
    c1, c2 = st.columns(2)
    with c1:
        param_ranges["alpha"] = st.multiselect("α values", [-2.0,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0], default=[0.0,1.0])
        param_ranges["beta"]  = st.multiselect("β values", [0.5,1.0,1.5,2.0,2.5,3.0], default=[1.0,2.0])
        param_ranges["M"]     = st.multiselect("M values",  [-1.0,-0.5,0.0,0.5,1.0], default=[0.0])
    with c2:
        if any(g.startswith("N") for g in selected_gens):
            param_ranges["q"] = st.multiselect("q values", [2,3,4,5], default=[2,3])
            param_ranges["v"] = st.multiselect("v values", [2,3,4,5], default=[2,3])
        if any(g in ["L4","N6"] for g in selected_gens):
            param_ranges["a"] = st.multiselect("a values", [2.0,2.5,3.0,3.5,4.0], default=[2.0])

    st.markdown('<div class="section">Advanced</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        verify = st.checkbox("Verify all", True)
        save_dataset = st.checkbox("Save as dataset", True)
        dataset_name = None
        if save_dataset:
            dataset_name = st.text_input("Dataset name", f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with c2:
        export_fmt = st.selectbox("Export format (if returning ODEs)", ["JSONL", "CSV"])

    if st.button("🚀 Start Batch"):
        if not selected_gens or not selected_funcs:
            st.error("Pick at least one generator and one function.")
            return
        with st.spinner("Batch job started..."):
            res = api.batch_generate(
                generators=selected_gens,
                functions=selected_funcs,
                samples_per_combination=samples_per,
                parameters=param_ranges if any(param_ranges.values()) else None,
                verify=verify,
                dataset_name=dataset_name if save_dataset else None,
            )
            if res and "job_id" in res:
                done = wait_for_job(res["job_id"], max_attempts=600, poll_interval=3)
                if done and done.get("status") == "completed":
                    results = done.get("results", {})
                    st.success(f"Done. Generated {results.get('total_generated', 0):,} ODEs.")
                    if save_dataset and results.get("dataset_info"):
                        info = results["dataset_info"]
                        st.session_state["current_dataset"] = info["name"]
                        st.session_state["available_datasets"].append(info)
                        with st.container():
                            st.markdown(f"**Dataset:** {info['name']} — {info.get('size', 0):,} ODEs")
                    if "odes" in results:
                        st.session_state["batch_dataset"] = results["odes"]
                        st.markdown("### Sample")
                        for ode in results["odes"][:5]:
                            with st.expander(f"{ode.get('generator_name','?')} / {ode.get('function_name','?')}"):
                                latex_box("ODE", format_eq(ode.get("ode_symbolic","")))
                                if ode.get("solution_symbolic"):
                                    latex_box("Solution", f"y(x) = {ode['solution_symbolic']}")
                        if st.button("📥 Download dataset"):
                            if export_fmt == "JSONL":
                                jsonl_download(results["odes"], f"batch_odes_{len(results['odes'])}.jsonl", "Download JSONL")
                            else:
                                df = pd.DataFrame(results["odes"])
                                csv_download(df, "batch_odes.csv", "Download CSV")

# --------------------------- #
#       Page: Verification    #
# --------------------------- #

def page_verification():
    st.subheader("✅ ODE Verification")
    t1, t2 = st.tabs(["Single", "Batch"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            ode_s = st.text_area("ODE (SymPy Eq(...))", "Eq(Derivative(y(x), x, 2) + y(x), pi*sin(x))", height=120)
        with c2:
            sol_s = st.text_area("Solution expression", "pi*sin(x)", height=120)
        method = st.selectbox("Method", ["substitution","numerical","checkodesol"])
        if st.button("🔍 Verify"):
            with st.spinner("Verifying..."):
                res = api.verify(ode_s, sol_s, method)
                if res:
                    ok = res.get("verified", False)
                    st.success(f"Verified ✅ (confidence {res.get('confidence',0):.2%})") if ok else st.error("Not verified ❌")
                    with st.expander("Details"):
                        st.json(res)
    with t2:
        up = st.file_uploader("Upload JSONL (ode_symbolic, solution_symbolic)", type=["jsonl","json"])
        if up is not None:
            lines = [json.loads(l) for l in up.read().decode().splitlines() if l.strip()]
            st.info(f"Loaded {len(lines)} ODEs.")
            method = st.selectbox("Batch method", ["substitution","numerical","all"])
            if st.button("Run batch verification"):
                out = []
                good = 0
                bar = st.progress(0)
                for i, row in enumerate(lines, 1):
                    ode = row.get("ode_symbolic")
                    sol = row.get("solution_symbolic")
                    res = api.verify(ode, sol, method) if (ode and sol) else None
                    row["verification_result"] = res
                    out.append(row)
                    if res and res.get("verified"):
                        good += 1
                    bar.progress(i/len(lines))
                st.success(f"{good}/{len(lines)} verified.")
                if st.button("📥 Download verified file"):
                    jsonl_download(out, "verified_odes.jsonl", "Download JSONL")

# --------------------------- #
#     Page: Datasets          #
# --------------------------- #

def page_datasets():
    st.subheader("📊 Dataset Management")
    if st.button("🔄 Refresh"):
        st.session_state["available_datasets"] = api.list_datasets().get("datasets", [])
        st.success("Refreshed.")
    if not st.session_state["available_datasets"]:
        st.info("No datasets yet. Use Batch Generation.")
    else:
        for ds in st.session_state["available_datasets"]:
            with st.expander(f"📁 {ds['name']} — {ds.get('size',0):,} ODEs"):
                st.write(f"Path: {ds.get('path','?')}")
                st.write(f"Created: {ds.get('created_at','?')}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Select", key=f"sel_{ds['name']}"):
                        st.session_state["current_dataset"] = ds["name"]
                        st.success(f"Selected {ds['name']}")
                with c2:
                    st.caption("Download link would be provided by API (if available).")

    st.markdown("### Create Dataset from Last Batch")
    if st.session_state["batch_dataset"]:
        st.info(f"{len(st.session_state['batch_dataset'])} ODEs buffered from last run.")
        new_name = st.text_input("Dataset name", f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        if st.button("💾 Save as dataset"):
            with st.spinner("Saving..."):
                res = api.create_dataset(st.session_state["batch_dataset"], new_name)
                if res:
                    st.success(f"Saved as '{new_name}'")
                    st.session_state["current_dataset"] = new_name
                    st.session_state["available_datasets"] = api.list_datasets().get("datasets", [])
                    st.session_state["batch_dataset"] = []
    else:
        st.info("Run a Batch Generation to buffer ODEs for saving.")

# --------------------------- #
#       Page: ML Train        #
# --------------------------- #

def page_ml_training():
    st.subheader("🤖 ML Training")
    datasets = st.session_state["available_datasets"]
    if not datasets:
        st.warning("No datasets available.")
        return
    names = [d["name"] for d in datasets]
    ds_name = st.selectbox("Dataset", names, index=0)
    model_type = st.selectbox("Model Type", ["pattern_net", "transformer", "vae"])
    c1, c2 = st.columns(2)
    with c1:
        epochs = st.slider("Epochs", 10, 200, 50, 10)
        batch_size = st.selectbox("Batch size", [16,32,64,128], index=1)
    with c2:
        lr = st.select_slider("Learning rate", options=[1e-5,1e-4,1e-3,1e-2], value=1e-3, format_func=lambda x: f"{x:.0e}")
        early = st.checkbox("Early stopping", True)

    # model-specific
    cfg: Dict[str, Any] = {"batch_size": batch_size, "learning_rate": lr, "early_stopping": early}
    if model_type == "pattern_net":
        cfg["hidden_dims"] = st.multiselect("Hidden dims", [64,128,256,512], default=[256,128,64])
        cfg["dropout_rate"] = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    elif model_type == "transformer":
        c1, c2 = st.columns(2)
        with c1:
            cfg["d_model"] = st.selectbox("d_model", [256,512,768], index=1)
            cfg["n_heads"] = st.selectbox("Heads", [4,8,12], index=1)
        with c2:
            cfg["n_layers"] = st.slider("Layers", 2, 12, 6)
            cfg["dim_feedforward"] = st.selectbox("FF dim", [1024,2048,4096], index=1)
    else:  # vae
        c1, c2 = st.columns(2)
        with c1:
            cfg["latent_dim"] = st.slider("Latent dim", 16, 256, 64, 16)
            cfg["hidden_dim"] = st.slider("Hidden dim", 128, 512, 256, 32)
        with c2:
            cfg["beta"] = st.slider("KL β", 0.1, 10.0, 1.0, 0.1)

    if st.button("🚀 Start Training"):
        with st.spinner("Submitting training job..."):
            res = api.train_ml(ds_name, model_type, epochs, cfg)
            if res and "job_id" in res:
                done = wait_for_job(res["job_id"], max_attempts=epochs * 10, poll_interval=2)
                if done and done.get("status") == "completed":
                    results = done.get("results", {})
                    st.success("Training completed.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Loss", f"{results.get('final_metrics',{}).get('loss',0):.4f}")
                    c2.metric("Acc", f"{results.get('final_metrics',{}).get('accuracy',0):.2%}")
                    c3.metric("Val Loss", f"{results.get('final_metrics',{}).get('validation_loss',0):.4f}")
                    c4.metric("Time (s)", f"{results.get('training_time',0):.1f}")
                    st.session_state["available_models"] = api.get_models().get("models", [])

# --------------------------- #
#     Page: ML Generation     #
# --------------------------- #

def page_ml_generation():
    st.subheader("🧪 ML-Based Generation")
    models = st.session_state["available_models"]
    if not models:
        st.warning("No trained models yet.")
        return
    options = [(m["path"], f"{m['name']} ({m.get('metadata',{}).get('model_type','?')}) — {m.get('created','')[:10]}", m) for m in models]
    idx = st.selectbox("Model", range(len(options)), format_func=lambda i: options[i][1] if options else "—")
    model_path, _, meta = options[idx]
    st.caption(f"Trained on: {meta.get('metadata',{}).get('dataset','?')}")

    c1, c2 = st.columns(2)
    with c1:
        n_samples = st.slider("# ODEs", 5, 100, 20, 5)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    with c2:
        gen = st.selectbox("Target generator", ["Auto"] + st.session_state["available_generators"])
        fun = st.selectbox("Target function", ["Auto"] + st.session_state["available_functions"][:20])
    with st.expander("Advanced filters"):
        c1, c2 = st.columns(2)
        with c1:
            cmin = st.number_input("Min complexity", 0, 10000, 50, 10)
        with c2:
            cmax = st.number_input("Max complexity", cmin, 20000, 200, 10)
        verify = st.checkbox("Verify outputs", True)
        dedup = st.checkbox("Filter duplicates", True)

    if st.button("🎨 Generate"):
        kwargs = {"complexity_range": [cmin, cmax]}
        if gen != "Auto":
            kwargs["generator"] = gen
        if fun != "Auto":
            kwargs["function"] = fun
        res = api.ml_generate(model_path, n_samples, temperature, **kwargs)
        if res and "job_id" in res:
            done = wait_for_job(res["job_id"], max_attempts=180)
            if done and done.get("status") == "completed":
                results = done.get("results", {})
                odes = results.get("odes", [])
                st.success(f"Generated {len(odes)} ODEs.")
                st.session_state["ml_generated_odes"].extend(odes)
                show = st.slider("Show first N", 1, len(odes), min(10, len(odes)))
                for ode in odes[:show]:
                    with st.expander(f"{ode.get('generator','ML')} / {ode.get('function','?')} — {'✅' if ode.get('verified') else '❓'}"):
                        latex_box("ODE", format_eq(ode.get("ode","")))
                        if ode.get("solution"):
                            latex_box("Solution", f"y(x) = {ode['solution']}")
                if st.button("📥 Download generated"):
                    jsonl_download(odes, f"ml_generated_{len(odes)}.jsonl", "Download JSONL")
                if st.button("💾 Save as dataset"):
                    name = f"ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    with st.spinner("Saving..."):
                        ok = api.create_dataset(odes, name)
                        if ok:
                            st.success(f"Saved dataset '{name}'")
                            st.session_state["available_datasets"] = api.list_datasets().get("datasets", [])

# --------------------------- #
#        Page: Analysis       #
# --------------------------- #

def page_analysis():
    st.subheader("📈 Analysis & Visualization")
    src = st.radio("Data source", ["Generated", "Batch buffer", "ML generated"], horizontal=True)
    if src == "Generated":
        data = st.session_state["generated_odes"]
    elif src == "Batch buffer":
        data = st.session_state["batch_dataset"]
    else:
        data = st.session_state["ml_generated_odes"]

    if not data:
        st.info("No data for analysis.")
        return

    df = pd.DataFrame(data)
    st.info(f"Rows: {len(df)}")

    t1, t2, t3 = st.tabs(["Overview", "Distributions", "Verification"])
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df))
        if "verified" in df:
            c2.metric("Verified", int(df["verified"].sum()))
        if "generator" in df:
            c3.metric("Generators", df["generator"].nunique())
        if "function" in df:
            c4.metric("Functions", df["function"].nunique())

        if "generator" in df:
            cnt = df["generator"].value_counts()
            st.plotly_chart(px.bar(x=cnt.index, y=cnt.values, labels={"x":"Generator","y":"Count"}, title="By Generator"), use_container_width=True)
        if "function" in df:
            cnt = df["function"].value_counts().head(15)
            st.plotly_chart(px.pie(values=cnt.values, names=cnt.index, title="Top Functions"), use_container_width=True)

    with t2:
        if "complexity" in df:
            st.plotly_chart(px.histogram(df, x="complexity", nbins=50, title="Complexity distribution"), use_container_width=True)
            if "generator" in df:
                st.plotly_chart(px.box(df, x="generator", y="complexity", title="Complexity by Generator", color="generator"), use_container_width=True)

    with t3:
        if "verified" in df:
            ver_rate = df["verified"].mean() * 100 if len(df) else 0
            st.metric("Verification rate", f"{ver_rate:.1f}%")
            if "complexity" in df:
                bins = pd.qcut(df["complexity"], q=min(10, df["complexity"].nunique()), duplicates="drop")
                ver_by_bin = df.groupby(bins)["verified"].mean() * 100
                st.plotly_chart(px.line(x=ver_by_bin.index.astype(str), y=ver_by_bin.values, markers=True,
                                        labels={"x":"Complexity bin","y":"Verification %"}, title="Verification vs Complexity"),
                                use_container_width=True)

    st.markdown("### Export")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Download JSONL"):
            jsonl_download(data, "analysis_export.jsonl", "Download JSONL")
    with c2:
        if st.button("📥 Download CSV"):
            csv_download(df, "analysis_export.csv", "Download CSV")

# --------------------------- #
#       Page: System Tools    #
# --------------------------- #

def page_tools():
    st.subheader("🛠️ System Tools")
    t1, t2, t3 = st.tabs(["Status", "Metrics", "Utilities"])
    with t1:
        status = st.session_state["api_status"] or {}
        col1, col2 = st.columns(2)
        with col1:
            st.json(status)
        with col2:
            stats = api.get_statistics() or {}
            st.json(stats)
        if st.button("🔄 Refresh health"):
            st.session_state["api_status"] = run_async(api.health())
            st.success("Refreshed.")

    with t2:
        if st.button("Fetch Prometheus metrics"):
            text = api.get_metrics()
            st.text_area("metrics", text[:2000] + ("..." if len(text) > 2000 else ""), height=300)

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear in-memory ODEs"):
                st.session_state["generated_odes"] = []
                st.session_state["batch_dataset"] = []
                st.session_state["ml_generated_odes"] = []
                st.success("Cleared.")
        with c2:
            if st.button("Reset session"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.experimental_rerun()

# --------------------------- #
#          Page: Docs         #
# --------------------------- #

def page_docs():
    st.subheader("📚 Documentation (Quick)")
    st.markdown(
        """
**Workflow**
1. Generate (Quick or Batch)  
2. Verify (optional)  
3. Create/Select Dataset  
4. Train ML  
5. Generate via ML  
6. Analyze & Export

**API headers**: `X-API-Key` required.

**Generators**: L* (linear), N* (nonlinear).  
**Common params**: α, β, M; nonlinear: q, v; pantograph: a.
        """
    )

# --------------------------- #
#             Main            #
# --------------------------- #

def main():
    pages = {
        "🏠 Dashboard": page_dashboard,
        "⚡ Quick Generate": page_quick_generate,
        "📦 Batch Generation": page_batch_generation,
        "✅ Verification": page_verification,
        "📊 Dataset Management": page_datasets,
        "🤖 ML Training": page_ml_training,
        "🧪 ML Generation": page_ml_generation,
        "📈 Analysis": page_analysis,
        "🛠️ System Tools": page_tools,
        "📚 Docs": page_docs,
    }
    pages[page]()

if __name__ == "__main__":
    main()
