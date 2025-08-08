# streamlit_app.py
"""
ODE Master Generator — Enhanced Streamlit UI
============================================

Improvements for Railway deployment:
- Automatic API endpoint discovery
- Robust connection retry logic  
- Better error messages with troubleshooting hints
- Connection status indicators
- Fallback to demo mode automatically
- Support for various API configurations
"""

from __future__ import annotations

import os
import time
import json
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional for parsing/plotting
try:
    import sympy as sp
except Exception:
    sp = None

# ---------- Config ----------

st.set_page_config(
    page_title="ODE Master Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Environment variables with Railway-friendly defaults
API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
if not API_BASE_URL:
    # Try Railway internal URL first, then localhost
    RAILWAY_PRIVATE_DOMAIN = os.getenv("RAILWAY_PRIVATE_DOMAIN", "")
    if RAILWAY_PRIVATE_DOMAIN:
        API_BASE_URL = f"http://{RAILWAY_PRIVATE_DOMAIN}"
    else:
        API_BASE_URL = "http://localhost:8000"

API_BASE_URL = API_BASE_URL.rstrip("/")
API_KEY = os.getenv("API_KEY", "test-key")
USE_DEMO = str(os.getenv("USE_DEMO", "1")).lower() in {"1", "true", "yes", "on"}
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# ---------- Styling ----------

st.markdown(
    """
    <style>
      .app-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:18px 22px;border-radius:14px;color:#fff;margin-bottom:18px;box-shadow:0 8px 20px rgba(0,0,0,.08)}
      .metric-card{background:#fff;padding:14px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:8px}
      .info-box{background:#eef2ff;border:1px solid #c7d2fe;border-radius:10px;padding:12px}
      .warn-box{background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:12px}
      .error-box{background:#fee;border:1px solid #fcc;border-radius:10px;padding:12px}
      .success-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:12px}
      .ok-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#10b981;margin-right:8px;box-shadow:0 0 8px #10b981}
      .bad-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:8px;box-shadow:0 0 8px #ef4444}
      .warn-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#f59e0b;margin-right:8px;box-shadow:0 0 8px #f59e0b}
      .latex-box{font-size:1.05em;padding:10px;border-radius:8px;background:#f8fafc;border:1px solid #e5e7eb;overflow-x:auto}
      .connection-status{padding:8px 12px;border-radius:8px;margin-bottom:12px;font-size:0.9em}
      .connection-ok{background:#f0fdf4;border:1px solid #10b981;color:#065f46}
      .connection-error{background:#fef2f2;border:1px solid #ef4444;color:#991b1b}
      .connection-demo{background:#fef3c7;border:1px solid #f59e0b;color:#92400e}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session State ----------

def init_session_state():
    """Initialize session state with defaults"""
    ss = st.session_state
    ss.setdefault("api_status", None)
    ss.setdefault("api_url", API_BASE_URL)
    ss.setdefault("api_reachable", False)
    ss.setdefault("using_demo", False)
    ss.setdefault("connection_attempts", 0)
    ss.setdefault("last_health_check", None)
    ss.setdefault("available_generators", [])
    ss.setdefault("available_functions", [])
    ss.setdefault("generated_odes", [])
    ss.setdefault("batch_dataset", [])
    ss.setdefault("current_dataset", None)
    ss.setdefault("available_datasets", [])
    ss.setdefault("available_models", [])
    ss.setdefault("ml_generated_odes", [])
    ss.setdefault("api_endpoints", {})

init_session_state()

# ---------- Demo Data ----------

DEMO_GENERATORS = ["L1", "L2", "L3", "L4", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
DEMO_FUNCTIONS = [
    "sine", "cosine", "tangent_safe", "exponential", "exp_scaled",
    "quadratic", "cubic", "sinh", "cosh", "tanh", "log_safe"
]

def get_demo_ode():
    """Generate a demo ODE response"""
    return {
        "id": str(uuid.uuid4()),
        "ode": "y'' + y = π·sin(x)",
        "solution": "π·sin(x)",
        "verified": True,
        "complexity": 50,
        "generator": "L1",
        "function": "sine",
        "parameters": {"alpha": 1.0, "beta": 1.0, "M": 0.0},
        "timestamp": datetime.now().isoformat(),
        "properties": {
            "generation_time_ms": 150,
            "verification_confidence": 0.99
        }
    }

# ---------- Enhanced API Client ----------

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def _try_endpoints(self, path: str, method: str = "GET", 
                       json_body: Any = None, timeout: int = REQUEST_TIMEOUT) -> Tuple[Optional[Any], Optional[str], str]:
        """Try multiple endpoint variations to find the working one"""
        # Possible endpoint patterns - order matters!
        variations = []
        
        # For health/metrics, try root first
        if path in ["/health", "/metrics", "/"]:
            variations.append(f"{self.base_url}{path}")
        else:
            # Try these patterns in order
            variations.extend([
                f"{self.base_url}{path}",  # Direct path (no prefix)
                f"{self.base_url}/api/v1{path}",  # With /api/v1 prefix
                f"{self.base_url}/api{path}",  # With /api prefix
                f"{self.base_url}/v1{path}",  # With /v1 prefix
            ])
        
        last_error = None
        for url in variations:
            try:
                if method == "GET":
                    r = self.session.get(url, timeout=timeout)
                elif method == "POST":
                    r = self.session.post(url, json=json_body, timeout=timeout)
                else:
                    r = self.session.request(method, url, json=json_body, timeout=timeout)
                
                if r.status_code < 400:
                    # Success! Store this endpoint pattern
                    if path not in st.session_state.api_endpoints:
                        st.session_state.api_endpoints[path] = url
                    
                    if path == "/metrics":
                        return r.text, None, url
                    
                    try:
                        return r.json() if r.text.strip() else {}, None, url
                    except:
                        return r.text, None, url
                        
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                
            except requests.exceptions.ConnectionError:
                last_error = "Connection refused"
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except Exception as e:
                last_error = str(e)
        
        return None, last_error, ""
    
    def _request(self, method: str, path: str, json_body: Any = None, 
                 timeout: int = REQUEST_TIMEOUT) -> Tuple[Optional[Any], Optional[str]]:
        """Make a request, using cached endpoint if available"""
        # Check if we have a cached working endpoint
        if path in st.session_state.api_endpoints:
            url = st.session_state.api_endpoints[path]
            try:
                if method == "GET":
                    r = self.session.get(url, timeout=timeout)
                elif method == "POST":
                    r = self.session.post(url, json=json_body, timeout=timeout)
                else:
                    r = self.session.request(method, url, json=json_body, timeout=timeout)
                
                if r.status_code < 400:
                    if path == "/metrics":
                        return r.text, None
                    return r.json() if r.text.strip() else {}, None
            except:
                # Cached endpoint failed, remove it
                del st.session_state.api_endpoints[path]
        
        # Try to find working endpoint
        data, err, _ = self._try_endpoints(path, method, json_body, timeout)
        return data, err
    
    def health(self) -> Dict[str, Any]:
        """Check API health with automatic endpoint discovery"""
        data, err, url = self._try_endpoints("/health", timeout=5)
        
        if data:
            return {
                "status": data.get("status", "unknown"),
                "ml_enabled": data.get("ml_enabled", False),
                "redis_enabled": data.get("redis_enabled", False),
                "generators": data.get("generators", 0),
                "functions": data.get("functions", 0),
                "working_url": url
            }
        
        return {
            "status": "error",
            "error": err or "Could not connect to API",
            "working_url": None
        }
    
    def generators(self) -> List[str]:
        """Get available generators"""
        data, _ = self._request("GET", "/generators")
        
        if isinstance(data, dict):
            # Handle various response formats
            if "all" in data:
                return data["all"]
            elif "linear" in data and "nonlinear" in data:
                return data["linear"] + data["nonlinear"]
            elif "generators" in data:
                return data["generators"]
        elif isinstance(data, list):
            return data
        
        return []
    
    def functions(self) -> List[str]:
        """Get available functions"""
        data, _ = self._request("GET", "/functions")
        
        if isinstance(data, dict):
            if "functions" in data:
                return data["functions"]
            elif "all" in data:
                return data["all"]
        elif isinstance(data, list):
            return data
        
        return []
    
    def generate(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate ODEs"""
        data, err = self._request("POST", "/generate", json_body=kwargs)
        if err:
            st.error(f"Generation failed: {err}")
            return None
        return data
    
    def batch_generate(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Batch generate ODEs"""
        data, err = self._request("POST", "/batch_generate", json_body=kwargs)
        if err:
            st.error(f"Batch generation failed: {err}")
            return None
        return data
    
    def verify(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Verify ODE solution"""
        data, err = self._request("POST", "/verify", json_body=kwargs)
        if err:
            st.error(f"Verification failed: {err}")
            return None
        return data
    
    def datasets(self) -> Dict[str, Any]:
        """List datasets"""
        data, _ = self._request("GET", "/datasets")
        return data or {"datasets": [], "count": 0}
    
    def create_dataset(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Create dataset"""
        data, err = self._request("POST", "/datasets/create", json_body=kwargs)
        if err:
            st.error(f"Dataset creation failed: {err}")
            return None
        return data
    
    def models(self) -> Dict[str, Any]:
        """List ML models"""
        data, _ = self._request("GET", "/models")
        return data or {"models": [], "count": 0}
    
    def stats(self) -> Dict[str, Any]:
        """Get statistics"""
        data, _ = self._request("GET", "/stats")
        return data or {}
    
    def metrics(self) -> str:
        """Get Prometheus metrics"""
        data, _ = self._request("GET", "/metrics")
        return data or ""
    
    def job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check job status"""
        data, err = self._request("GET", f"/jobs/{job_id}")
        if err:
            return None
        return data
    
    def ml_train(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Start ML training"""
        data, err = self._request("POST", "/ml/train", json_body=kwargs)
        if err:
            st.error(f"ML training failed: {err}")
            return None
        return data
    
    def ml_generate(self, **kwargs) -> Optional[Dict[str, Any]]:
        """ML generation"""
        data, err = self._request("POST", "/ml/generate", json_body=kwargs)
        if err:
            st.error(f"ML generation failed: {err}")
            return None
        return data

# Initialize API client
api = APIClient(API_BASE_URL, API_KEY)

# ---------- Connection Management ----------

def check_api_connection() -> Tuple[bool, str]:
    """Check API connection and return status"""
    try:
        # First, try to reach the root endpoint to see if API is up
        root_response, root_error, root_url = api._try_endpoints("/", timeout=5)
        
        if root_response and isinstance(root_response, dict):
            # API is responding, check for available endpoints
            endpoints = root_response.get("endpoints", {})
            if endpoints:
                st.session_state.api_status = {
                    "status": "connected",
                    "root_response": root_response,
                    "endpoints": endpoints
                }
        
        # Now try health endpoint
        health = api.health()
        
        if health.get("status") in ["healthy", "ok", "operational"]:
            st.session_state.api_reachable = True
            st.session_state.api_status = health
            st.session_state.last_health_check = time.time()
            
            # Try to get generators and functions
            gens = api.generators()
            funcs = api.functions()
            
            if gens and funcs:
                st.session_state.available_generators = gens
                st.session_state.available_functions = funcs
                st.session_state.using_demo = False
                return True, f"API connected successfully ({len(gens)} generators, {len(funcs)} functions)"
            elif USE_DEMO:
                # API is up but no generators/functions, use demo
                st.session_state.available_generators = DEMO_GENERATORS
                st.session_state.available_functions = DEMO_FUNCTIONS
                st.session_state.using_demo = True
                return True, "API connected but using demo data (no generators/functions from API)"
            else:
                return False, "API connected but no generators/functions available"
        else:
            # Check if we at least got a response from root
            if root_response:
                error_msg = f"API responding but health check failed: {health.get('error', 'unknown')}"
            else:
                error_msg = health.get("error", "Cannot reach API")
            
            if USE_DEMO:
                st.session_state.available_generators = DEMO_GENERATORS
                st.session_state.available_functions = DEMO_FUNCTIONS
                st.session_state.using_demo = True
                st.session_state.api_reachable = False
                return True, f"Using demo mode ({error_msg})"
            else:
                st.session_state.api_reachable = False
                return False, error_msg
                
    except Exception as e:
        if USE_DEMO:
            st.session_state.available_generators = DEMO_GENERATORS
            st.session_state.available_functions = DEMO_FUNCTIONS
            st.session_state.using_demo = True
            st.session_state.api_reachable = False
            return True, f"Using demo mode (Connection error: {str(e)[:100]})"
        else:
            st.session_state.api_reachable = False
            return False, f"Cannot connect to API: {str(e)}"

def display_connection_status():
    """Display connection status banner"""
    if st.session_state.using_demo:
        st.markdown(
            f"""<div class='connection-status connection-demo'>
            <span class='warn-dot'></span>
            <b>Demo Mode</b> - API unavailable, using sample data
            </div>""",
            unsafe_allow_html=True
        )
    elif st.session_state.api_reachable:
        st.markdown(
            f"""<div class='connection-status connection-ok'>
            <span class='ok-dot'></span>
            <b>Connected</b> - {st.session_state.api_url}
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class='connection-status connection-error'>
            <span class='bad-dot'></span>
            <b>Disconnected</b> - Cannot reach API
            </div>""",
            unsafe_allow_html=True
        )

def ensure_connection():
    """Ensure we have a working connection or demo mode"""
    # Check if we need to refresh connection status
    if st.session_state.last_health_check is None or \
       time.time() - st.session_state.last_health_check > 30:
        
        success, message = check_api_connection()
        
        if not success and not USE_DEMO:
            st.error(f"""
            ### 🔴 API Connection Failed
            
            {message}
            
            **Troubleshooting:**
            1. Check if your API server is running
            2. Verify the API_BASE_URL environment variable
            3. Current URL: `{st.session_state.api_url}`
            4. Enable demo mode with USE_DEMO=1
            
            **For Railway deployment:**
            - Check your service is deployed and running
            - Verify environment variables are set correctly
            - Check service logs for errors
            """)
            st.stop()

# ---------- Helper Functions ----------

import uuid  # Add this import at the top

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

def render_equation_block(title: str, expr: str):
    st.markdown(f"**{title}**")
    st.markdown(f"<div class='latex-box'>{expr}</div>", unsafe_allow_html=True)

def params_controls(generator: str, key_prefix: str = "") -> Dict[str, Any]:
    col1, col2 = st.columns(2)
    params: Dict[str, Any] = {}
    with col1:
        params["alpha"] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1, key=f"{key_prefix}_alpha")
        params["beta"] = st.slider("β (beta)", 0.1, 3.0, 1.0, 0.1, key=f"{key_prefix}_beta")
    with col2:
        params["M"] = st.slider("M", -1.0, 1.0, 0.0, 0.1, key=f"{key_prefix}_M")
        if generator.startswith("N"):
            params["q"] = st.slider("q (power)", 2, 5, 2, key=f"{key_prefix}_q")
            if generator in {"N2", "N3", "N6", "N7"}:
                params["v"] = st.slider("v (power)", 2, 5, 3, key=f"{key_prefix}_v")
        if generator in {"L4", "N6"}:
            params["a"] = st.slider("a (pantograph)", 2.0, 5.0, 2.0, 0.5, key=f"{key_prefix}_a")
    return params

def wait_for_job(job_id: str, max_secs: int = 900, poll: float = 1.5) -> Optional[Dict[str, Any]]:
    """Wait for job completion with progress display"""
    if st.session_state.using_demo:
        # Simulate job completion in demo mode
        time.sleep(1)
        return {
            "status": "completed",
            "results": [get_demo_ode() for _ in range(3)]
        }
    
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
                info.text(f"{status} — epoch {meta.get('current_epoch')}/{meta.get('total_epochs', '?')}")
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
    st.markdown(
        f"<a download='{filename}' href='data:application/json;base64,{b64}'>📥 Download {filename}</a>",
        unsafe_allow_html=True
    )

# ---------- Page Functions ----------

def page_dashboard():
    st.title("Dashboard")
    ensure_connection()
    display_connection_status()
    
    # Metrics
    colA, colB, colC, colD = st.columns(4)
    
    with colA:
        if st.session_state.api_reachable:
            st.metric("API Status", "🟢 Online")
        elif st.session_state.using_demo:
            st.metric("API Status", "🟡 Demo")
        else:
            st.metric("API Status", "🔴 Offline")
    
    stats = api.stats() if st.session_state.api_reachable else {}
    
    with colB:
        st.metric("Generated (24h)", f"{stats.get('total_generated_24h', 0):,}")
    with colC:
        st.metric("Verification Rate", f"{stats.get('verification_success_rate', 0):.1%}")
    with colD:
        st.metric("Active Jobs", stats.get('active_jobs', 0))
    
    # Recent ODEs
    st.subheader("Recent ODEs")
    if not st.session_state.generated_odes:
        st.info("No recent ODEs yet. Try Quick Generate.")
    else:
        for i, ode in enumerate(st.session_state.generated_odes[-5:][::-1], 1):
            with st.expander(f"Recent {i}: {ode.get('generator', '?')} × {ode.get('function', '?')}"):
                render_equation_block("ODE", ode.get("ode", ""))
                if ode.get("solution"):
                    render_equation_block("Solution", f"y(x) = {ode['solution']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity", ode.get("complexity", "N/A"))
                c2.metric("Verified", "✅" if ode.get("verified") else "❌")
                c3.metric("Gen Time", f"{ode.get('properties', {}).get('generation_time_ms', 0):.0f} ms")
    
    # Resources
    st.subheader("Available Resources")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generators", len(st.session_state.available_generators))
    c2.metric("Functions", len(st.session_state.available_functions))
    c3.metric("Datasets", len(st.session_state.available_datasets))
    c4.metric("Models", len(st.session_state.available_models))
    
    if st.session_state.using_demo:
        st.info("📌 **Demo Mode Active** - Limited functionality available. Connect to an API server for full features.")

def page_quick_generate():
    st.title("⚡ Quick Generate")
    ensure_connection()
    display_connection_status()
    
    gens = st.session_state.available_generators
    funcs = st.session_state.available_functions
    
    if not gens or not funcs:
        st.error("""
        ### No generators or functions available
        
        Please check:
        1. API server is running and accessible
        2. Core modules are properly installed
        3. Or enable demo mode with USE_DEMO=1
        """)
        return
    
    c1, c2 = st.columns(2)
    with c1:
        generator = st.selectbox("Generator", gens, index=0)
    with c2:
        function = st.selectbox("Function", funcs, index=0)
    
    st.markdown("### Parameters")
    params = params_controls(generator, key_prefix="quick")
    
    with st.expander("Advanced Options"):
        colA, colB = st.columns(2)
        with colA:
            verify = st.checkbox("Verify solution", True)
            count = st.number_input("Number of ODEs", 1, 10, 1)
        with colB:
            show_plot = st.checkbox("Plot solution (if available)", False)
            x_range = st.slider("Plot range", -10.0, 10.0, (-5.0, 5.0))
    
    if st.button("🚀 Generate", type="primary"):
        if st.session_state.using_demo:
            # Demo mode generation
            st.success("Generated (Demo Mode)")
            demo_odes = [get_demo_ode() for _ in range(count)]
            for i, ode in enumerate(demo_odes, 1):
                st.session_state.generated_odes.append(ode)
                st.markdown("---")
                st.subheader(f"ODE {i}")
                render_equation_block("Generated ODE", ode["ode"])
                render_equation_block("Solution", f"y(x) = {ode['solution']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Complexity", ode["complexity"])
                c2.metric("Verified", "✅" if ode["verified"] else "❌")
                c3.metric("Time", f"{ode['properties']['generation_time_ms']:.0f} ms")
        else:
            # Real API generation
            with st.status("Generating...", expanded=True) as s:
                resp = api.generate(
                    generator=generator,
                    function=function,
                    parameters=params,
                    count=count,
                    verify=verify
                )
                
                if not resp or "job_id" not in resp:
                    s.update(label="Failed to start job", state="error")
                    return
                
                s.update(label=f"Job {resp['job_id']} started")
            
            js = wait_for_job(resp["job_id"])
            
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
                    c3.metric("Confidence", f"{ode.get('properties', {}).get('verification_confidence', 0):.2%}")
                    c4.metric("Time", f"{ode.get('properties', {}).get('generation_time_ms', 0):.0f} ms")
            else:
                st.error("No results returned")
    
    # Export section
    if st.session_state.generated_odes:
        st.markdown("### Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download all"):
                download_jsonl(st.session_state.generated_odes, "generated_odes.jsonl")
        with col2:
            if st.button("🗑️ Clear cache"):
                st.session_state.generated_odes = []
                st.success("Cleared")

def page_tools():
    st.title("🛠️ Tools & Diagnostics")
    
    st.subheader("🔌 Connection Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Configuration")
        st.code(f"""
API URL: {st.session_state.api_url}
API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if len(API_KEY) > 4 else '****'}
Demo Mode: {USE_DEMO}
Timeout: {REQUEST_TIMEOUT}s
        """)
        
        if st.button("🔄 Test Connection"):
            with st.spinner("Testing connection..."):
                success, message = check_api_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    with col2:
        st.markdown("### API Endpoints Found")
        if st.session_state.api_endpoints:
            for path, url in st.session_state.api_endpoints.items():
                st.code(f"{path} → {url}")
        else:
            st.info("No endpoints discovered yet")
    
    st.subheader("📊 API Status")
    
    if st.session_state.api_status:
        st.json(st.session_state.api_status)
    else:
        st.info("No status data available")
    
    # Stats
    if st.session_state.api_reachable:
        st.subheader("📈 Statistics")
        stats = api.stats()
        if stats:
            st.json(stats)
    
    # Session management
    st.subheader("🧹 Session Management")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("Clear ODEs Cache"):
            st.session_state.generated_odes = []
            st.session_state.batch_dataset = []
            st.session_state.ml_generated_odes = []
            st.success("Cleared ODE caches")
    
    with c2:
        if st.button("Reset Connection"):
            st.session_state.api_endpoints = {}
            st.session_state.last_health_check = None
            st.session_state.connection_attempts = 0
            check_api_connection()
            st.success("Connection reset")
    
    with c3:
        if st.button("Full Reset"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session fully reset")
            st.rerun()

# Add placeholder functions for other pages
def page_batch_generation():
    st.title("📦 Batch Generation")
    ensure_connection()
    display_connection_status()
    
    if st.session_state.using_demo:
        st.warning("Batch generation is not available in demo mode")
        return
    
    st.info("Batch generation functionality - implement based on API availability")

def page_verify():
    st.title("✅ Verification")
    ensure_connection()
    display_connection_status()
    
    if st.session_state.using_demo:
        st.warning("Verification is limited in demo mode")
    
    st.info("Verification functionality - implement based on API availability")

def page_datasets():
    st.title("📊 Datasets")
    ensure_connection()
    display_connection_status()
    
    if st.session_state.using_demo:
        st.warning("Dataset management is not available in demo mode")
        return
    
    st.info("Dataset management - implement based on API availability")

def page_ml_training():
    st.title("🤖 ML Training")
    ensure_connection()
    display_connection_status()
    
    if st.session_state.using_demo:
        st.warning("ML features are not available in demo mode")
        return
    
    st.info("ML training - implement based on API availability")

def page_ml_generate():
    st.title("🧪 ML Generation")
    ensure_connection()
    display_connection_status()
    
    if st.session_state.using_demo:
        st.warning("ML generation is not available in demo mode")
        return
    
    st.info("ML generation - implement based on API availability")

def page_analysis():
    st.title("📈 Analysis")
    ensure_connection()
    display_connection_status()
    
    st.info("Analysis functionality - implement based on data availability")

def page_docs():
    st.title("📚 Documentation")
    
    st.markdown("""
    ## 🚀 Quick Start
    
    ### For Railway Deployment:
    
    1. **Environment Variables** (set in Railway):
       ```
       API_BASE_URL=https://your-api.railway.app
       API_KEY=your-api-key
       USE_DEMO=1  # Enable demo mode fallback
       ```
    
    2. **Connection Issues?**
       - Check the Tools page for diagnostics
       - Verify your API service is running
       - Check Railway logs for errors
       - Demo mode will activate automatically if API is unreachable
    
    ### Features:
    
    - **Automatic API Discovery**: Tries multiple endpoint patterns
    - **Demo Mode**: Works without API connection
    - **Connection Retry**: Automatic reconnection attempts
    - **Endpoint Caching**: Remembers working endpoints
    
    ### Workflow:
    
    1. **Quick Generate**: Generate individual ODEs
    2. **Batch Generation**: Generate multiple ODEs
    3. **Verification**: Verify ODE solutions
    4. **Datasets**: Manage ODE datasets
    5. **ML Training**: Train ML models
    6. **ML Generation**: Generate using ML
    7. **Analysis**: Analyze generated data
    
    ### Troubleshooting:
    
    - **"No generators/functions available"**: 
      - API is reachable but core modules missing
      - Enable USE_DEMO=1 for demo mode
    
    - **"Cannot connect to API"**:
      - Check API_BASE_URL is correct
      - Verify API service is running
      - Check network/firewall settings
    
    - **Railway specific**:
      - Use internal domain for faster connections
      - Check service is deployed and healthy
      - Verify environment variables are set
    """)

# ---------- Main App ----------

def main():
    header()
    
    with st.sidebar:
        st.markdown("### Navigation")
        
        page = st.radio(
            "Go to",
            [
                "🏠 Dashboard",
                "⚡ Quick Generate",
                "📦 Batch Generation",
                "✅ Verification",
                "📊 Datasets",
                "🤖 ML Training",
                "🧪 ML Generation",
                "📈 Analysis",
                "🛠️ Tools",
                "📚 Docs"
            ],
        )
        
        st.markdown("---")
        
        # Connection status in sidebar
        if st.session_state.api_reachable:
            st.success("✅ API Connected")
        elif st.session_state.using_demo:
            st.warning("🔶 Demo Mode")
        else:
            st.error("❌ API Offline")
        
        st.caption(f"**URL:** {st.session_state.api_url}")
        
        if st.button("🔄 Refresh Connection"):
            check_api_connection()
            st.rerun()
    
    # Route to pages
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
    elif page == "📚 Docs":
        page_docs()

if __name__ == "__main__":
    main()
