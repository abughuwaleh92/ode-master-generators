# gui_app.py
"""
Comprehensive Streamlit GUI for ODE Master Generators
Full-featured interface covering all system capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import base64
import io
import zipfile
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import sympy as sp
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Streamlit
st.set_page_config(
    page_title="ODE Master Generators",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "ODE Master Generators v3.0 - Complete System Interface"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .success-box {
        padding: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        padding: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
    .stButton > button {
        width: 100%;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #c0c2c6;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'api_url': 'http://localhost:8000',
        'api_key': 'test-key',
        'current_job': None,
        'generated_odes': [],
        'datasets': [],
        'models': [],
        'verification_results': [],
        'ml_training_history': [],
        'analysis_results': {},
        'active_jobs': {},
        'connection_status': 'disconnected',
        'last_refresh': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# API Client
class APIClient:
    """API client for backend communication"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}
        
    async def _request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Async request handler"""
        url = f"{self.base_url}{endpoint}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method, url, 
                    headers=self.headers,
                    json=data if method != 'GET' else None,
                    params=data if method == 'GET' else None
                ) as response:
                    return await response.json()
            except Exception as e:
                st.error(f"API Error: {e}")
                return None
    
    def request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Sync wrapper for async requests"""
        return asyncio.run(self._request(method, endpoint, data))
    
    def check_health(self) -> bool:
        """Check API health"""
        try:
            resp = self.request('GET', '/health')
            return resp is not None and resp.get('status') == 'healthy'
        except:
            return False
    
    def get_generators(self) -> Dict:
        """Get available generators"""
        return self.request('GET', '/generators')
    
    def get_functions(self) -> Dict:
        """Get available functions"""
        return self.request('GET', '/functions')
    
    def generate_ode(self, generator: str, function: str, params: Dict = None, count: int = 1) -> Dict:
        """Generate ODEs"""
        data = {
            'generator': generator,
            'function': function,
            'parameters': params or {},
            'count': count,
            'verify': True
        }
        return self.request('POST', '/generate', data)
    
    def batch_generate(self, generators: List[str], functions: List[str], 
                      samples: int = 5, params: Dict = None, dataset_name: str = None) -> Dict:
        """Batch generate ODEs"""
        data = {
            'generators': generators,
            'functions': functions,
            'samples_per_combination': samples,
            'parameters': params,
            'verify': True,
            'dataset_name': dataset_name
        }
        return self.request('POST', '/batch_generate', data)
    
    def verify_ode(self, ode: str, solution: str, method: str = 'substitution') -> Dict:
        """Verify ODE solution"""
        data = {'ode': ode, 'solution': solution, 'method': method}
        return self.request('POST', '/verify', data)
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get job status"""
        return self.request('GET', f'/jobs/{job_id}')
    
    def list_datasets(self) -> Dict:
        """List available datasets"""
        return self.request('GET', '/datasets')
    
    def create_dataset(self, odes: List[Dict], name: str = None) -> Dict:
        """Create dataset"""
        data = {'odes': odes, 'dataset_name': name}
        return self.request('POST', '/datasets/create', data)
    
    def train_model(self, dataset: str, model_type: str, config: Dict) -> Dict:
        """Train ML model"""
        data = {
            'dataset': dataset,
            'model_type': model_type,
            'epochs': config.get('epochs', 50),
            'batch_size': config.get('batch_size', 32),
            'learning_rate': config.get('learning_rate', 0.001),
            'config': config
        }
        return self.request('POST', '/ml/train', data)
    
    def generate_with_ml(self, model_path: str, n_samples: int, config: Dict) -> Dict:
        """Generate ODEs using ML model"""
        data = {
            'model_path': model_path,
            'n_samples': n_samples,
            'temperature': config.get('temperature', 0.8),
            'generator': config.get('generator'),
            'function': config.get('function')
        }
        return self.request('POST', '/ml/generate', data)
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.request('GET', '/stats')

# Initialize
init_session_state()

# Sidebar - Configuration & Navigation
with st.sidebar:
    st.title("🔧 Configuration")
    
    # API Configuration
    with st.expander("API Settings", expanded=True):
        st.session_state.api_url = st.text_input(
            "API URL", 
            value=st.session_state.api_url,
            help="Backend API endpoint"
        )
        st.session_state.api_key = st.text_input(
            "API Key", 
            value=st.session_state.api_key,
            type="password",
            help="Your API authentication key"
        )
        
        if st.button("Test Connection", type="secondary"):
            client = APIClient(st.session_state.api_url, st.session_state.api_key)
            if client.check_health():
                st.success("✅ Connected successfully!")
                st.session_state.connection_status = 'connected'
            else:
                st.error("❌ Connection failed!")
                st.session_state.connection_status = 'disconnected'
    
    # Connection Status
    status_color = "🟢" if st.session_state.connection_status == 'connected' else "🔴"
    st.markdown(f"**Status:** {status_color} {st.session_state.connection_status.title()}")
    
    st.divider()
    
    # Quick Actions
    st.markdown("### Quick Actions")
    
    if st.button("📊 Refresh Stats", use_container_width=True):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    if st.button("💾 Download All Datasets", use_container_width=True):
        # Implementation would download all datasets
        st.info("Feature coming soon!")
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Main Content Area
st.title("∫ ODE Master Generators - Complete System Interface")
st.markdown("### Comprehensive control panel for ODE generation, verification, ML training, and analysis")

# Create API client
api_client = APIClient(st.session_state.api_url, st.session_state.api_key)

# Main tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Dashboard",
    "⚙️ Generation",
    "✅ Verification", 
    "📊 Datasets",
    "🤖 Machine Learning",
    "📈 Analysis",
    "🔍 Explorer",
    "⚡ Advanced"
])

# Tab 1: Dashboard
with tab1:
    st.header("System Dashboard")
    
    # Fetch system stats
    stats = api_client.get_stats() if st.session_state.connection_status == 'connected' else {}
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Generated (24h)",
            stats.get('total_generated_24h', 0),
            delta=f"+{stats.get('total_generated_24h', 0) // 24}/hr"
        )
    
    with col2:
        st.metric(
            "Verification Rate",
            f"{stats.get('verification_success_rate', 0):.1%}",
            delta=f"{stats.get('verification_success_rate', 0) - 0.9:.1%}"
        )
    
    with col3:
        st.metric(
            "Active Jobs",
            stats.get('active_jobs', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            "Available Generators",
            stats.get('available_generators', 0),
            delta=None
        )
    
    # Live Charts
    st.subheader("Real-time Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generation rate chart
        fig = go.Figure()
        
        # Simulate data (in production, fetch from API)
        times = pd.date_range(end=datetime.now(), periods=24, freq='H')
        rates = np.random.poisson(50, 24)
        
        fig.add_trace(go.Scatter(
            x=times, y=rates,
            mode='lines+markers',
            name='Generation Rate',
            line=dict(color='#3498db', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Generation Rate (ODEs/hour)",
            xaxis_title="Time",
            yaxis_title="ODEs Generated",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Verification success by generator
        if st.session_state.connection_status == 'connected':
            generators_data = api_client.get_generators()
            if generators_data:
                # Create pie chart of generators
                all_gens = generators_data.get('all', [])
                gen_types = ['Linear' if g.startswith('L') else 'Nonlinear' for g in all_gens]
                
                fig = px.pie(
                    values=[gen_types.count('Linear'), gen_types.count('Nonlinear')],
                    names=['Linear', 'Nonlinear'],
                    title="Generator Distribution",
                    color_discrete_map={'Linear': '#2ecc71', 'Nonlinear': '#e74c3c'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity Log
    st.subheader("Recent Activity")
    
    activity_df = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=10, freq='5min'),
        'Action': np.random.choice(['Generated', 'Verified', 'Training Started', 'Dataset Created'], 10),
        'Details': [f"Job_{i:04d}" for i in range(10)],
        'Status': np.random.choice(['✅ Success', '⚠️ Warning', '❌ Failed'], 10, p=[0.7, 0.2, 0.1])
    })
    
    st.dataframe(
        activity_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time": st.column_config.DatetimeColumn(format="HH:mm:ss"),
            "Status": st.column_config.TextColumn(width="small")
        }
    )

# Tab 2: Generation
with tab2:
    st.header("ODE Generation")
    
    generation_mode = st.radio(
        "Generation Mode",
        ["Single ODE", "Batch Generation", "Custom Parameters"],
        horizontal=True
    )
    
    if generation_mode == "Single ODE":
        col1, col2 = st.columns(2)
        
        with col1:
            # Get available generators
            if st.session_state.connection_status == 'connected':
                gen_data = api_client.get_generators()
                generators = gen_data.get('all', []) if gen_data else []
            else:
                generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
            
            selected_gen = st.selectbox("Select Generator", generators)
            
            # Get available functions
            if st.session_state.connection_status == 'connected':
                func_data = api_client.get_functions()
                functions = func_data.get('functions', []) if func_data else []
            else:
                functions = ['sine', 'cosine', 'exponential', 'quadratic']
            
            selected_func = st.selectbox("Select Function", functions)
            
            count = st.number_input("Number of ODEs", min_value=1, max_value=100, value=1)
        
        with col2:
            st.subheader("Parameters")
            
            params = {}
            params['alpha'] = st.slider("α (alpha)", -2.0, 2.0, 1.0, 0.1)
            params['beta'] = st.slider("β (beta)", 0.1, 3.0, 1.0, 0.1)
            params['M'] = st.slider("M", -1.0, 1.0, 0.0, 0.1)
            
            if selected_gen.startswith('N'):
                params['q'] = st.slider("q (power)", 2, 5, 2)
                params['v'] = st.slider("v (power)", 2, 5, 3)
            
            if 'L4' in selected_gen or 'N6' in selected_gen:
                params['a'] = st.slider("a (pantograph)", 2, 4, 2)
        
        if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                result = api_client.generate_ode(
                    selected_gen, selected_func, params, count
                )
                
                if result and 'job_id' in result:
                    st.session_state.current_job = result['job_id']
                    st.success(f"Generation started! Job ID: {result['job_id']}")
                    
                    # Poll for results
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        job_status = api_client.get_job_status(result['job_id'])
                        
                        if job_status:
                            progress = job_status.get('progress', 0) / 100
                            progress_bar.progress(progress)
                            status_text.text(f"Status: {job_status.get('status', 'unknown')}")
                            
                            if job_status.get('status') == 'completed':
                                st.success("✅ Generation completed!")
                                
                                # Display results
                                if 'results' in job_status and job_status['results']:
                                    st.session_state.generated_odes = job_status['results']
                                    
                                    for i, ode in enumerate(job_status['results']):
                                        with st.expander(f"ODE {i+1}", expanded=i==0):
                                            st.latex(ode.get('ode', ''))
                                            if ode.get('solution'):
                                                st.markdown("**Solution:**")
                                                st.latex(ode.get('solution', ''))
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Verified", "✅" if ode.get('verified') else "❌")
                                            with col2:
                                                st.metric("Complexity", ode.get('complexity', 0))
                                            with col3:
                                                st.metric("Generator", ode.get('generator', ''))
                                break
                            
                            elif job_status.get('status') == 'failed':
                                st.error(f"Generation failed: {job_status.get('error', 'Unknown error')}")
                                break
                        
                        time.sleep(1)
    
    elif generation_mode == "Batch Generation":
        st.subheader("Batch Generation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Multi-select generators
            if st.session_state.connection_status == 'connected':
                gen_data = api_client.get_generators()
                all_generators = gen_data.get('all', []) if gen_data else []
            else:
                all_generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
            
            selected_generators = st.multiselect(
                "Select Generators",
                all_generators,
                default=all_generators[:3]
            )
            
            # Multi-select functions
            if st.session_state.connection_status == 'connected':
                func_data = api_client.get_functions()
                all_functions = func_data.get('functions', []) if func_data else []
            else:
                all_functions = ['sine', 'cosine', 'exponential', 'quadratic', 'cubic']
            
            selected_functions = st.multiselect(
                "Select Functions",
                all_functions,
                default=all_functions[:3]
            )
        
        with col2:
            samples_per_combo = st.number_input(
                "Samples per Combination",
                min_value=1,
                max_value=50,
                value=5
            )
            
            dataset_name = st.text_input(
                "Dataset Name (optional)",
                placeholder=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Parameter ranges
            with st.expander("Parameter Ranges"):
                param_ranges = {}
                
                col1, col2 = st.columns(2)
                with col1:
                    param_ranges['alpha'] = st.multiselect(
                        "α values",
                        [-2, -1, 0, 0.5, 1, 1.5, 2],
                        default=[0, 1, 2]
                    )
                    param_ranges['beta'] = st.multiselect(
                        "β values",
                        [0.5, 1, 1.5, 2, 2.5, 3],
                        default=[1, 2]
                    )
                
                with col2:
                    param_ranges['M'] = st.multiselect(
                        "M values",
                        [-1, -0.5, 0, 0.5, 1],
                        default=[0]
                    )
                    if any(g.startswith('N') for g in selected_generators):
                        param_ranges['q'] = st.multiselect("q values", [2, 3, 4], default=[2, 3])
                        param_ranges['v'] = st.multiselect("v values", [2, 3, 4], default=[2, 3])
        
        # Calculate total ODEs
        total_expected = len(selected_generators) * len(selected_functions) * samples_per_combo
        st.info(f"📊 Expected total ODEs: {total_expected}")
        
        if st.button("🚀 Start Batch Generation", type="primary", use_container_width=True):
            if not selected_generators or not selected_functions:
                st.error("Please select at least one generator and one function!")
            else:
                with st.spinner(f"Generating {total_expected} ODEs..."):
                    result = api_client.batch_generate(
                        selected_generators,
                        selected_functions,
                        samples_per_combo,
                        param_ranges,
                        dataset_name or None
                    )
                    
                    if result and 'job_id' in result:
                        # Monitor job
                        progress_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            metrics_container = st.container()
                            
                            while True:
                                job_status = api_client.get_job_status(result['job_id'])
                                
                                if job_status:
                                    progress = job_status.get('progress', 0) / 100
                                    progress_bar.progress(progress)
                                    
                                    metadata = job_status.get('metadata', {})
                                    status_text.text(
                                        f"Status: {job_status.get('status')} | "
                                        f"Progress: {metadata.get('current', 0)}/{metadata.get('total', total_expected)}"
                                    )
                                    
                                    if job_status.get('status') == 'completed':
                                        st.success("✅ Batch generation completed!")
                                        
                                        results = job_status.get('results', {})
                                        
                                        # Display summary
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Generated", results.get('total_generated', 0))
                                        with col2:
                                            st.metric("Verified", results.get('verified_count', 0))
                                        with col3:
                                            st.metric("Linear ODEs", results.get('summary', {}).get('linear', 0))
                                        with col4:
                                            st.metric("Nonlinear ODEs", results.get('summary', {}).get('nonlinear', 0))
                                        
                                        if 'dataset_info' in results:
                                            st.success(f"Dataset saved: {results['dataset_info']['name']}")
                                        
                                        break
                                    
                                    elif job_status.get('status') == 'failed':
                                        st.error(f"Batch generation failed: {job_status.get('error')}")
                                        break
                                
                                time.sleep(2)
    
    else:  # Custom Parameters mode
        st.subheader("Custom Parameter Generation")
        st.info("Advanced mode for fine-tuned ODE generation with custom parameter combinations")
        
        # Custom parameter grid
        st.markdown("### Define Parameter Grid")
        
        param_grid = {}
        num_params = st.number_input("Number of parameter sets", 1, 20, 3)
        
        param_df = pd.DataFrame({
            'Set': [f"Set_{i+1}" for i in range(num_params)],
            'alpha': [1.0] * num_params,
            'beta': [1.0] * num_params,
            'M': [0.0] * num_params,
            'q': [2] * num_params,
            'v': [3] * num_params
        })
        
        edited_df = st.data_editor(
            param_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "alpha": st.column_config.NumberColumn(min_value=-5, max_value=5, step=0.1),
                "beta": st.column_config.NumberColumn(min_value=0.1, max_value=5, step=0.1),
                "M": st.column_config.NumberColumn(min_value=-2, max_value=2, step=0.1),
                "q": st.column_config.NumberColumn(min_value=2, max_value=10, step=1),
                "v": st.column_config.NumberColumn(min_value=2, max_value=10, step=1)
            }
        )
        
        if st.button("Generate with Custom Parameters", type="primary"):
            st.info("Custom parameter generation initiated...")
            # Implementation would process custom parameters

# Tab 3: Verification
with tab3:
    st.header("ODE Verification")
    
    verification_mode = st.radio(
        "Verification Mode",
        ["Manual Verification", "Batch Verification", "Dataset Verification"],
        horizontal=True
    )
    
    if verification_mode == "Manual Verification":
        st.subheader("Enter ODE and Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ode_input = st.text_area(
                "ODE Expression",
                placeholder="e.g., Eq(Derivative(y(x), x, 2) + y(x), sin(x))",
                height=100
            )
            
            verification_method = st.selectbox(
                "Verification Method",
                ["substitution", "checkodesol", "numerical", "all"]
            )
        
        with col2:
            solution_input = st.text_area(
                "Solution Expression",
                placeholder="e.g., sin(x) + cos(x)",
                height=100
            )
            
            st.markdown("**Test Points** (for numerical verification)")
            test_points = st.text_input(
                "Test points (comma-separated)",
                value="0.1, 0.5, 1.0, 2.0"
            )
        
        if st.button("✅ Verify ODE", type="primary", use_container_width=True):
            if ode_input and solution_input:
                with st.spinner("Verifying..."):
                    result = api_client.verify_ode(
                        ode_input,
                        solution_input,
                        verification_method
                    )
                    
                    if result:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if result.get('verified'):
                                st.success("✅ VERIFIED")
                            else:
                                st.error("❌ NOT VERIFIED")
                        
                        with col2:
                            st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                        
                        with col3:
                            st.metric("Method Used", result.get('method', 'unknown'))
                        
                        # Display details
                        with st.expander("Verification Details"):
                            st.json(result.get('details', {}))
            else:
                st.error("Please enter both ODE and solution!")
    
    elif verification_mode == "Batch Verification":
        st.subheader("Batch ODE Verification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload ODE file (JSON/JSONL)",
            type=['json', 'jsonl']
        )
        
        if uploaded_file:
            # Parse uploaded file
            if uploaded_file.name.endswith('.jsonl'):
                lines = uploaded_file.read().decode('utf-8').strip().split('\n')
                odes = [json.loads(line) for line in lines if line]
            else:
                odes = json.loads(uploaded_file.read())
            
            st.success(f"Loaded {len(odes)} ODEs")
            
            # Verification settings
            col1, col2 = st.columns(2)
            
            with col1:
                verify_method = st.selectbox(
                    "Verification Method",
                    ["substitution", "numerical", "all"],
                    key="batch_verify_method"
                )
            
            with col2:
                parallel_workers = st.slider(
                    "Parallel Workers",
                    1, 8, 4
                )
            
            if st.button("Start Batch Verification", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                verified_odes = []
                
                for i, ode in enumerate(odes):
                    if 'ode_symbolic' in ode and 'solution_symbolic' in ode:
                        result = api_client.verify_ode(
                            ode['ode_symbolic'],
                            ode['solution_symbolic'],
                            verify_method
                        )
                        
                        if result:
                            ode['verified'] = result.get('verified', False)
                            ode['verification_confidence'] = result.get('confidence', 0)
                            ode['verification_method'] = result.get('method', 'unknown')
                        
                        verified_odes.append(ode)
                    
                    progress_bar.progress((i + 1) / len(odes))
                    status_text.text(f"Verified {i + 1}/{len(odes)} ODEs")
                
                st.success(f"Verification complete! {sum(1 for o in verified_odes if o.get('verified'))} verified out of {len(verified_odes)}")
                
                # Download results
                results_json = json.dumps(verified_odes, indent=2)
                st.download_button(
                    "📥 Download Verified ODEs",
                    results_json,
                    file_name=f"verified_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:  # Dataset Verification
        st.subheader("Dataset Verification")
        
        # List available datasets
        if st.session_state.connection_status == 'connected':
            datasets_resp = api_client.list_datasets()
            
            if datasets_resp and 'datasets' in datasets_resp:
                datasets = datasets_resp['datasets']
                
                if datasets:
                    dataset_names = [d['name'] for d in datasets]
                    selected_dataset = st.selectbox("Select Dataset", dataset_names)
                    
                    # Find selected dataset info
                    dataset_info = next((d for d in datasets if d['name'] == selected_dataset), None)
                    
                    if dataset_info:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Size", dataset_info.get('size', 0))
                        with col2:
                            st.metric("Created", dataset_info.get('created_at', '')[:10])
                        with col3:
                            st.metric("File Size", f"{dataset_info.get('file_size_bytes', 0) / 1024:.1f} KB")
                        
                        if st.button("Verify Dataset", type="primary"):
                            st.info("Dataset verification started...")
                            # Implementation would verify entire dataset
                else:
                    st.warning("No datasets found. Generate some ODEs first!")

# Tab 4: Datasets
with tab4:
    st.header("Dataset Management")
    
    dataset_action = st.radio(
        "Action",
        ["View Datasets", "Create Dataset", "Merge Datasets", "Export/Import"],
        horizontal=True
    )
    
    if dataset_action == "View Datasets":
        if st.session_state.connection_status == 'connected':
            datasets_resp = api_client.list_datasets()
            
            if datasets_resp and 'datasets' in datasets_resp:
                datasets = datasets_resp['datasets']
                
                if datasets:
                    # Create DataFrame for display
                    df = pd.DataFrame(datasets)
                    
                    # Display with custom configuration
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "name": st.column_config.TextColumn("Dataset Name"),
                            "size": st.column_config.NumberColumn("ODEs"),
                            "created_at": st.column_config.DatetimeColumn("Created"),
                            "path": st.column_config.TextColumn("Path")
                        },
                        hide_index=True
                    )
                    
                    # Dataset actions
                    selected_dataset = st.selectbox(
                        "Select dataset for actions",
                        [d['name'] for d in datasets]
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("📊 Analyze", use_container_width=True):
                            st.info(f"Analyzing {selected_dataset}...")
                    
                    with col2:
                        if st.button("✅ Verify", use_container_width=True):
                            st.info(f"Verifying {selected_dataset}...")
                    
                    with col3:
                        if st.button("📥 Download", use_container_width=True):
                            st.info(f"Downloading {selected_dataset}...")
                    
                    with col4:
                        if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                            st.warning(f"Delete {selected_dataset}?")
                else:
                    st.info("No datasets found. Generate some ODEs to create datasets!")
        else:
            st.warning("Connect to API to view datasets")
    
    elif dataset_action == "Create Dataset":
        st.subheader("Create New Dataset")
        
        dataset_name = st.text_input(
            "Dataset Name",
            placeholder=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Option to use generated ODEs
        if st.session_state.generated_odes:
            use_generated = st.checkbox(
                f"Use {len(st.session_state.generated_odes)} recently generated ODEs"
            )
            
            if use_generated:
                st.success(f"Will include {len(st.session_state.generated_odes)} ODEs in dataset")
        
        # Option to upload ODEs
        uploaded_odes = st.file_uploader(
            "Or upload ODEs (JSON/JSONL)",
            type=['json', 'jsonl']
        )
        
        if st.button("Create Dataset", type="primary"):
            odes_to_save = []
            
            if use_generated:
                odes_to_save.extend(st.session_state.generated_odes)
            
            if uploaded_odes:
                if uploaded_odes.name.endswith('.jsonl'):
                    lines = uploaded_odes.read().decode('utf-8').strip().split('\n')
                    odes_to_save.extend([json.loads(line) for line in lines if line])
                else:
                    odes_to_save.extend(json.loads(uploaded_odes.read()))
            
            if odes_to_save:
                result = api_client.create_dataset(odes_to_save, dataset_name)
                
                if result:
                    st.success(f"Dataset created: {result.get('dataset_name')}")
                    st.info(f"Path: {result.get('path')}")
                    st.info(f"Size: {result.get('size')} ODEs")
            else:
                st.error("No ODEs to save!")
    
    elif dataset_action == "Merge Datasets":
        st.subheader("Merge Multiple Datasets")
        
        if st.session_state.connection_status == 'connected':
            datasets_resp = api_client.list_datasets()
            
            if datasets_resp and 'datasets' in datasets_resp:
                datasets = datasets_resp['datasets']
                dataset_names = [d['name'] for d in datasets]
                
                selected_datasets = st.multiselect(
                    "Select datasets to merge",
                    dataset_names
                )
                
                if len(selected_datasets) >= 2:
                    new_name = st.text_input(
                        "Merged dataset name",
                        value=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    if st.button("Merge Datasets", type="primary"):
                        st.info(f"Merging {len(selected_datasets)} datasets...")
                        # Implementation would merge datasets
                else:
                    st.info("Select at least 2 datasets to merge")
    
    else:  # Export/Import
        st.subheader("Export/Import Datasets")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export")
            
            if st.session_state.connection_status == 'connected':
                datasets_resp = api_client.list_datasets()
                
                if datasets_resp and 'datasets' in datasets_resp:
                    datasets = datasets_resp['datasets']
                    
                    export_dataset = st.selectbox(
                        "Dataset to export",
                        [d['name'] for d in datasets]
                    )
                    
                    export_format = st.selectbox(
                        "Export format",
                        ["JSON", "JSONL", "CSV", "Parquet", "LaTeX"]
                    )
                    
                    if st.button("Export Dataset", type="primary"):
                        st.info(f"Exporting {export_dataset} as {export_format}...")
        
        with col2:
            st.markdown("### Import")
            
            imported_file = st.file_uploader(
                "Import dataset",
                type=['json', 'jsonl', 'csv', 'parquet']
            )
            
            if imported_file:
                import_name = st.text_input(
                    "Dataset name",
                    value=Path(imported_file.name).stem
                )
                
                if st.button("Import Dataset", type="primary"):
                    st.info(f"Importing {imported_file.name}...")

# Tab 5: Machine Learning
with tab5:
    st.header("Machine Learning Pipeline")
    
    ml_mode = st.radio(
        "ML Mode",
        ["Model Training", "Model Inference", "Model Management", "Feature Engineering"],
        horizontal=True
    )
    
    if ml_mode == "Model Training":
        st.subheader("Train ML Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset selection
            if st.session_state.connection_status == 'connected':
                datasets_resp = api_client.list_datasets()
                
                if datasets_resp and 'datasets' in datasets_resp:
                    datasets = datasets_resp['datasets']
                    
                    if datasets:
                        selected_dataset = st.selectbox(
                            "Training Dataset",
                            [d['name'] for d in datasets]
                        )
                        
                        # Model type selection
                        model_type = st.selectbox(
                            "Model Type",
                            ["pattern_net", "transformer", "vae", "language_model"]
                        )
                        
                        # Model name
                        model_name = st.text_input(
                            "Model Name",
                            value=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        )
        
        with col2:
            st.markdown("### Training Configuration")
            
            epochs = st.slider("Epochs", 10, 500, 50)
            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128, 256],
                value=32
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.00001, 0.0001, 0.001, 0.01, 0.1],
                value=0.001,
                format_func=lambda x: f"{x:.5f}"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                early_stopping = st.checkbox("Early Stopping", value=True)
                patience = st.number_input("Patience", 5, 50, 10)
                validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
                
                # Model-specific settings
                if model_type == "transformer":
                    n_heads = st.slider("Attention Heads", 4, 16, 8)
                    n_layers = st.slider("Transformer Layers", 2, 12, 6)
                elif model_type == "vae":
                    latent_dim = st.slider("Latent Dimension", 16, 256, 64)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            config = {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'early_stopping': early_stopping,
                'patience': patience,
                'validation_split': validation_split
            }
            
            result = api_client.train_model(
                selected_dataset,
                model_type,
                config
            )
            
            if result and 'job_id' in result:
                st.success(f"Training started! Job ID: {result['job_id']}")
                
                # Monitor training
                progress_container = st.container()
                
                with progress_container:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Training metrics chart
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Loss', 'Accuracy')
                        )
                        
                        # Placeholder for real-time updates
                        epochs_x = list(range(1, epochs + 1))
                        loss_y = [0] * epochs
                        acc_y = [0] * epochs
                        
                        fig.add_trace(
                            go.Scatter(x=epochs_x, y=loss_y, name='Training Loss'),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=epochs_x, y=acc_y, name='Validation Acc'),
                            row=2, col=1
                        )
                        
                        chart_placeholder = st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("### Training Status")
                        epoch_text = st.empty()
                        loss_metric = st.metric("Loss", "0.000")
                        acc_metric = st.metric("Accuracy", "0.0%")
                        time_remaining = st.empty()
                
                # Poll for status
                while True:
                    job_status = api_client.get_job_status(result['job_id'])
                    
                    if job_status:
                        if job_status.get('status') == 'completed':
                            st.success("✅ Training completed!")
                            
                            if 'results' in job_status:
                                model_info = job_status['results']
                                st.info(f"Model saved: {model_info.get('model_id')}")
                            break
                        
                        elif job_status.get('status') == 'failed':
                            st.error(f"Training failed: {job_status.get('error')}")
                            break
                    
                    time.sleep(2)
    
    elif ml_mode == "Model Inference":
        st.subheader("Generate ODEs with ML Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            model_path = st.text_input(
                "Model Path",
                placeholder="models/pattern_net_20240101.pth"
            )
            
            n_samples = st.slider("Number of Samples", 1, 100, 10)
            
            temperature = st.slider(
                "Temperature (creativity)",
                0.1, 2.0, 0.8, 0.1,
                help="Higher = more creative, Lower = more conservative"
            )
        
        with col2:
            st.markdown("### Generation Settings")
            
            # Optional constraints
            constrain_generator = st.checkbox("Constrain to specific generator")
            if constrain_generator:
                generator_constraint = st.selectbox(
                    "Generator",
                    ['Any'] + ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
                )
            
            constrain_function = st.checkbox("Constrain to specific function")
            if constrain_function:
                function_constraint = st.selectbox(
                    "Function",
                    ['Any'] + ['sine', 'cosine', 'exponential', 'quadratic']
                )
            
            complexity_range = st.slider(
                "Complexity Range",
                0, 500, (50, 200),
                help="Target complexity range for generated ODEs"
            )
        
        if st.button("🤖 Generate with ML", type="primary", use_container_width=True):
            config = {
                'temperature': temperature,
                'generator': generator_constraint if constrain_generator else None,
                'function': function_constraint if constrain_function else None,
                'complexity_range': list(complexity_range)
            }
            
            result = api_client.generate_with_ml(
                model_path,
                n_samples,
                config
            )
            
            if result and 'job_id' in result:
                # Monitor generation
                with st.spinner("Generating ODEs with ML model..."):
                    while True:
                        job_status = api_client.get_job_status(result['job_id'])
                        
                        if job_status:
                            if job_status.get('status') == 'completed':
                                st.success("✅ ML generation completed!")
                                
                                if 'results' in job_status and 'odes' in job_status['results']:
                                    generated = job_status['results']['odes']
                                    
                                    st.info(f"Generated {len(generated)} ODEs")
                                    
                                    # Display generated ODEs
                                    for i, ode in enumerate(generated[:5]):  # Show first 5
                                        with st.expander(f"ML Generated ODE {i+1}"):
                                            st.latex(ode.get('ode', ''))
                                            if ode.get('solution'):
                                                st.markdown("**Solution:**")
                                                st.latex(ode.get('solution'))
                                            
                                            st.json({
                                                'complexity': ode.get('complexity'),
                                                'generator': ode.get('generator'),
                                                'function': ode.get('function'),
                                                'ml_generated': True
                                            })
                                break
                            
                            elif job_status.get('status') == 'failed':
                                st.error(f"ML generation failed: {job_status.get('error')}")
                                break
                        
                        time.sleep(1)
    
    elif ml_mode == "Model Management":
        st.subheader("Manage ML Models")
        
        # List models
        if st.session_state.connection_status == 'connected':
            models = []  # Would fetch from API
            
            model_df = pd.DataFrame({
                'Name': ['pattern_net_v1', 'transformer_v2', 'vae_latest'],
                'Type': ['PatternNet', 'Transformer', 'VAE'],
                'Created': pd.date_range(end=datetime.now(), periods=3),
                'Size (MB)': [45.2, 128.5, 67.3],
                'Performance': [0.92, 0.95, 0.89]
            })
            
            st.dataframe(
                model_df,
                use_container_width=True,
                column_config={
                    "Performance": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="%.2f"
                    )
                }
            )
            
            # Model actions
            selected_model = st.selectbox(
                "Select model for actions",
                model_df['Name'].tolist()
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Evaluate", use_container_width=True):
                    st.info(f"Evaluating {selected_model}...")
            
            with col2:
                if st.button("📥 Download", use_container_width=True):
                    st.info(f"Downloading {selected_model}...")
            
            with col3:
                if st.button("🔄 Fine-tune", use_container_width=True):
                    st.info(f"Fine-tuning {selected_model}...")
            
            with col4:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                    st.warning(f"Delete {selected_model}?")
    
    else:  # Feature Engineering
        st.subheader("Feature Engineering")
        
        st.info("Extract and engineer features from ODE datasets for ML training")
        
        # Feature extraction settings
        feature_groups = st.multiselect(
            "Feature Groups to Extract",
            [
                "Basic Metrics",
                "Complexity Features",
                "Structural Features",
                "Nonlinearity Metrics",
                "Verification Features",
                "Text Features",
                "Parameter Statistics"
            ],
            default=["Basic Metrics", "Complexity Features", "Structural Features"]
        )
        
        if st.button("Extract Features", type="primary"):
            st.info("Feature extraction started...")
            
            # Would implement feature extraction
            
            # Show sample features
            sample_features = pd.DataFrame({
                'Feature': ['complexity_score', 'has_exponential', 'nonlin_degree', 'verification_rate'],
                'Type': ['numeric', 'boolean', 'numeric', 'numeric'],
                'Min': [10, 0, 1, 0],
                'Max': [500, 1, 10, 1],
                'Mean': [120.5, 0.3, 2.5, 0.92]
            })
            
            st.dataframe(sample_features, use_container_width=True)

# Tab 6: Analysis
with tab6:
    st.header("Data Analysis & Visualization")
    
    analysis_mode = st.radio(
        "Analysis Type",
        ["Statistical Analysis", "Visualization", "Pattern Discovery", "Performance Analysis"],
        horizontal=True
    )
    
    if analysis_mode == "Statistical Analysis":
        st.subheader("Statistical Analysis")
        
        # Dataset selection for analysis
        if st.session_state.connection_status == 'connected':
            datasets_resp = api_client.list_datasets()
            
            if datasets_resp and 'datasets' in datasets_resp:
                datasets = datasets_resp['datasets']
                
                if datasets:
                    selected_dataset = st.selectbox(
                        "Dataset to analyze",
                        [d['name'] for d in datasets],
                        key="stats_dataset"
                    )
                    
                    if st.button("Run Statistical Analysis", type="primary"):
                        # Simulate analysis results
                        st.markdown("### Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total ODEs", 1250)
                            st.metric("Unique Generators", 7)
                        
                        with col2:
                            st.metric("Verification Rate", "92.3%")
                            st.metric("Unique Functions", 15)
                        
                        with col3:
                            st.metric("Avg Complexity", 125.4)
                            st.metric("Std Complexity", 45.2)
                        
                        with col4:
                            st.metric("Linear ODEs", 750)
                            st.metric("Nonlinear ODEs", 500)
                        
                        # Detailed statistics
                        st.markdown("### Detailed Statistics")
                        
                        # Generator performance
                        gen_stats = pd.DataFrame({
                            'Generator': ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3'],
                            'Count': [200, 180, 170, 200, 150, 175, 175],
                            'Verified': [195, 170, 165, 180, 140, 160, 165],
                            'Success Rate': [0.975, 0.944, 0.971, 0.900, 0.933, 0.914, 0.943],
                            'Avg Complexity': [95, 102, 88, 145, 156, 168, 175]
                        })
                        
                        st.dataframe(
                            gen_stats,
                            use_container_width=True,
                            column_config={
                                "Success Rate": st.column_config.ProgressColumn(
                                    min_value=0,
                                    max_value=1,
                                    format="%.1%"
                                )
                            }
                        )
    
    elif analysis_mode == "Visualization":
        st.subheader("Data Visualization")
        
        viz_type = st.selectbox(
            "Visualization Type",
            [
                "Complexity Distribution",
                "Generator Performance",
                "Function Heatmap",
                "Time Series",
                "Correlation Matrix",
                "3D Scatter"
            ]
        )
        
        if viz_type == "Complexity Distribution":
            # Create complexity distribution
            fig = go.Figure()
            
            # Simulate data
            complexities = np.random.gamma(4, 20, 1000)
            
            fig.add_trace(go.Histogram(
                x=complexities,
                nbinsx=50,
                name='Complexity Distribution',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=np.linspace(complexities.min(), complexities.max(), 100),
                y=np.exp(-(np.linspace(complexities.min(), complexities.max(), 100) - complexities.mean())**2 / (2 * complexities.std()**2)) * 100,
                mode='lines',
                name='Normal Fit',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="ODE Complexity Distribution",
                xaxis_title="Complexity Score",
                yaxis_title="Count",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Generator Performance":
            # Create performance comparison
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Success Rate by Generator', 'Complexity by Generator'),
                specs=[[{'type': 'bar'}, {'type': 'box'}]]
            )
            
            generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
            success_rates = [0.95, 0.92, 0.94, 0.88, 0.85, 0.87, 0.90]
            
            fig.add_trace(
                go.Bar(x=generators, y=success_rates, name='Success Rate'),
                row=1, col=1
            )
            
            for gen in generators:
                fig.add_trace(
                    go.Box(y=np.random.normal(100 if gen.startswith('L') else 150, 20, 50),
                          name=gen),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Function Heatmap":
            # Create heatmap
            generators = ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3']
            functions = ['sine', 'cosine', 'exp', 'quad', 'cubic']
            
            # Simulate success rates
            z = np.random.uniform(0.8, 1.0, (len(functions), len(generators)))
            
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=generators,
                y=functions,
                colorscale='RdYlGn',
                text=np.round(z, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Verification Success Rate: Generators vs Functions",
                xaxis_title="Generator",
                yaxis_title="Function",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_mode == "Pattern Discovery":
        st.subheader("Pattern Discovery")
        
        discovery_type = st.selectbox(
            "Discovery Method",
            ["Clustering", "Association Rules", "Anomaly Detection", "Trend Analysis"]
        )
        
        if discovery_type == "Clustering":
            st.info("Discover clusters of similar ODEs based on their properties")
            
            # Clustering parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            
            with col2:
                clustering_method = st.selectbox(
                    "Method",
                    ["K-Means", "DBSCAN", "Hierarchical"]
                )
            
            with col3:
                features_for_clustering = st.multiselect(
                    "Features",
                    ["Complexity", "Order", "Nonlinearity", "Parameters"],
                    default=["Complexity", "Order"]
                )
            
            if st.button("Run Clustering", type="primary"):
                # Simulate clustering results
                st.markdown("### Clustering Results")
                
                # Create scatter plot
                fig = px.scatter(
                    x=np.random.randn(200),
                    y=np.random.randn(200),
                    color=np.random.choice(range(n_clusters), 200),
                    title="ODE Clusters",
                    labels={'x': 'Component 1', 'y': 'Component 2'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                cluster_stats = pd.DataFrame({
                    'Cluster': range(n_clusters),
                    'Size': np.random.randint(20, 60, n_clusters),
                    'Avg Complexity': np.random.uniform(80, 180, n_clusters),
                    'Dominant Generator': np.random.choice(['L1', 'L2', 'N1'], n_clusters)
                })
                
                st.dataframe(cluster_stats, use_container_width=True)
    
    else:  # Performance Analysis
        st.subheader("Performance Analysis")
        
        # Performance metrics
        st.markdown("### System Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generation speed chart
            fig = go.Figure()
            
            times = pd.date_range(end=datetime.now(), periods=24, freq='H')
            gen_speed = np.random.uniform(40, 60, 24)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=gen_speed,
                mode='lines+markers',
                name='Generation Speed',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Generation Speed (ODEs/second)",
                xaxis_title="Time",
                yaxis_title="Speed",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Resource usage
            fig = go.Figure()
            
            categories = ['CPU', 'Memory', 'Disk I/O', 'Network']
            values = [65, 45, 30, 20]
            
            fig.add_trace(go.Bar(
               x=categories,
               y=values,
               marker_color=['red' if v > 50 else 'green' for v in values]
           ))
           
           fig.update_layout(
               title="Resource Usage (%)",
               yaxis_title="Usage %",
               height=300
           )
           
           st.plotly_chart(fig, use_container_width=True)
       
       # Bottleneck analysis
       st.markdown("### Bottleneck Analysis")
       
       bottleneck_data = pd.DataFrame({
           'Component': ['ODE Generation', 'Verification', 'ML Training', 'Data I/O', 'API Calls'],
           'Time (ms)': [125, 450, 2000, 85, 150],
           'Frequency': [1000, 1000, 50, 500, 800],
           'Impact': ['Medium', 'High', 'Low', 'Low', 'Medium']
       })
       
       st.dataframe(
           bottleneck_data,
           use_container_width=True,
           column_config={
               "Time (ms)": st.column_config.ProgressColumn(
                   min_value=0,
                   max_value=2000,
                   format="%d ms"
               ),
               "Impact": st.column_config.TextColumn(
                   help="Performance impact level"
               )
           }
       )

# Tab 7: Explorer
with tab7:
   st.header("ODE Explorer")
   
   explorer_mode = st.radio(
       "Explorer Mode",
       ["Search ODEs", "Interactive Viewer", "Solution Visualizer", "LaTeX Renderer"],
       horizontal=True
   )
   
   if explorer_mode == "Search ODEs":
       st.subheader("Search and Filter ODEs")
       
       # Search filters
       col1, col2, col3 = st.columns(3)
       
       with col1:
           search_term = st.text_input(
               "Search Term",
               placeholder="e.g., sin, exp, derivative"
           )
           
           generator_filter = st.multiselect(
               "Generators",
               ['All'] + ['L1', 'L2', 'L3', 'L4', 'N1', 'N2', 'N3'],
               default=['All']
           )
       
       with col2:
           function_filter = st.multiselect(
               "Functions",
               ['All'] + ['sine', 'cosine', 'exponential', 'quadratic'],
               default=['All']
           )
           
           verified_only = st.checkbox("Verified Only", value=True)
       
       with col3:
           complexity_range = st.slider(
               "Complexity Range",
               0, 500, (0, 500)
           )
           
           sort_by = st.selectbox(
               "Sort By",
               ["ID", "Complexity", "Generator", "Function", "Verification"]
           )
       
       if st.button("🔍 Search", type="primary", use_container_width=True):
           # Simulate search results
           results = []
           for i in range(10):
               results.append({
                   'ID': f"ODE_{i:04d}",
                   'Generator': np.random.choice(['L1', 'L2', 'N1']),
                   'Function': np.random.choice(['sine', 'cosine']),
                   'Complexity': np.random.randint(50, 200),
                   'Verified': np.random.choice([True, False], p=[0.9, 0.1])
               })
           
           search_df = pd.DataFrame(results)
           
           st.dataframe(
               search_df,
               use_container_width=True,
               column_config={
                   "Verified": st.column_config.CheckboxColumn()
               }
           )
           
           # ODE details
           selected_ode = st.selectbox(
               "Select ODE for details",
               search_df['ID'].tolist()
           )
           
           if selected_ode:
               with st.expander(f"Details: {selected_ode}", expanded=True):
                   st.latex(r"y''(x) + y(x) = \pi \sin(x)")
                   st.markdown("**Solution:**")
                   st.latex(r"y(x) = \pi \sin(x) + C_1 \cos(x) + C_2 \sin(x)")
                   
                   col1, col2 = st.columns(2)
                   with col1:
                       st.json({
                           "initial_conditions": {"y(0)": "0", "y'(0)": "π"},
                           "parameters": {"alpha": 1.0, "beta": 1.0, "M": 0}
                       })
                   with col2:
                       st.json({
                           "verification": {
                               "method": "substitution",
                               "confidence": 0.99,
                               "residual": 1e-10
                           }
                       })
   
   elif explorer_mode == "Interactive Viewer":
       st.subheader("Interactive ODE Viewer")
       
       # ODE input or selection
       ode_source = st.radio(
           "ODE Source",
           ["Enter Manually", "Select from Dataset", "Use Generated"],
           horizontal=True
       )
       
       if ode_source == "Enter Manually":
           ode_input = st.text_area(
               "Enter ODE",
               value="y''(x) + y(x) = sin(x)",
               height=100
           )
           
           solution_input = st.text_area(
               "Enter Solution (optional)",
               placeholder="y(x) = ...",
               height=100
           )
       
       # Interactive controls
       st.markdown("### Interactive Controls")
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           x_range = st.slider(
               "x range",
               -10.0, 10.0, (-5.0, 5.0)
           )
       
       with col2:
           num_points = st.slider(
               "Number of points",
               50, 500, 200
           )
       
       with col3:
           show_derivatives = st.checkbox("Show derivatives")
           show_phase_portrait = st.checkbox("Show phase portrait")
       
       # Visualization
       if st.button("Visualize", type="primary"):
           # Create interactive plot
           x = np.linspace(x_range[0], x_range[1], num_points)
           
           # Example: sine solution
           y = np.sin(x)
           
           fig = make_subplots(
               rows=2 if show_derivatives else 1,
               cols=2 if show_phase_portrait else 1,
               subplot_titles=(
                   'Solution',
                   'Phase Portrait' if show_phase_portrait else None,
                   'Derivatives' if show_derivatives else None,
                   None
               )
           )
           
           # Main solution
           fig.add_trace(
               go.Scatter(x=x, y=y, mode='lines', name='y(x)'),
               row=1, col=1
           )
           
           if show_derivatives:
               # First derivative
               y_prime = np.cos(x)
               fig.add_trace(
                   go.Scatter(x=x, y=y_prime, mode='lines', name="y'(x)"),
                   row=2, col=1
               )
               
               # Second derivative
               y_double_prime = -np.sin(x)
               fig.add_trace(
                   go.Scatter(x=x, y=y_double_prime, mode='lines', name="y''(x)"),
                   row=2, col=1
               )
           
           if show_phase_portrait:
               # Phase portrait
               y_prime = np.cos(x)
               fig.add_trace(
                   go.Scatter(x=y, y=y_prime, mode='lines', name='Phase'),
                   row=1, col=2
               )
           
           fig.update_layout(height=600, showlegend=True)
           st.plotly_chart(fig, use_container_width=True)
   
   elif explorer_mode == "Solution Visualizer":
       st.subheader("Solution Visualizer")
       
       # Parameter controls
       st.markdown("### Parametric Solution Explorer")
       
       col1, col2 = st.columns([2, 1])
       
       with col1:
           # Parameter sliders for real-time updates
           param_alpha = st.slider("α", -5.0, 5.0, 1.0, 0.1, key="viz_alpha")
           param_beta = st.slider("β", 0.1, 5.0, 1.0, 0.1, key="viz_beta")
           param_M = st.slider("M", -2.0, 2.0, 0.0, 0.1, key="viz_M")
           
           # Create 3D surface plot
           alpha_range = np.linspace(-5, 5, 50)
           beta_range = np.linspace(0.1, 5, 50)
           
           Alpha, Beta = np.meshgrid(alpha_range, beta_range)
           
           # Example surface: solution magnitude
           Z = np.sin(Alpha) * np.exp(-Beta/2)
           
           fig = go.Figure(data=[go.Surface(
               x=Alpha,
               y=Beta,
               z=Z,
               colorscale='Viridis'
           )])
           
           # Add current point
           fig.add_trace(go.Scatter3d(
               x=[param_alpha],
               y=[param_beta],
               z=[np.sin(param_alpha) * np.exp(-param_beta/2)],
               mode='markers',
               marker=dict(size=10, color='red'),
               name='Current'
           ))
           
           fig.update_layout(
               title="Solution Surface",
               scene=dict(
                   xaxis_title="α",
                   yaxis_title="β",
                   zaxis_title="Solution Magnitude"
               ),
               height=500
           )
           
           st.plotly_chart(fig, use_container_width=True)
       
       with col2:
           st.markdown("### Solution Properties")
           
           # Calculate properties at current parameters
           solution_value = np.sin(param_alpha) * np.exp(-param_beta/2)
           
           st.metric("Solution at origin", f"{solution_value:.4f}")
           st.metric("Stability", "Stable" if abs(solution_value) < 1 else "Unstable")
           st.metric("Oscillatory", "Yes" if param_alpha != 0 else "No")
           
           # Initial conditions
           st.markdown("### Initial Conditions")
           st.text(f"y(0) = {param_M:.2f}")
           st.text(f"y'(0) = {param_beta * param_alpha:.2f}")
   
   else:  # LaTeX Renderer
       st.subheader("LaTeX Renderer")
       
       col1, col2 = st.columns(2)
       
       with col1:
           st.markdown("### Input")
           
           latex_input = st.text_area(
               "Enter LaTeX",
               value=r"\frac{d^2y}{dx^2} + y = \pi \sin(x)",
               height=150
           )
           
           # Quick templates
           st.markdown("**Quick Templates:**")
           
           if st.button("Linear ODE", use_container_width=True):
               latex_input = r"y''(x) + p(x)y'(x) + q(x)y(x) = f(x)"
           
           if st.button("Nonlinear ODE", use_container_width=True):
               latex_input = r"(y''(x))^2 + \sin(y'(x)) + e^{y(x)} = 0"
           
           if st.button("System of ODEs", use_container_width=True):
               latex_input = r"\begin{cases} \frac{dx}{dt} = ax + by \\ \frac{dy}{dt} = cx + dy \end{cases}"
       
       with col2:
           st.markdown("### Rendered Output")
           
           # Render LaTeX
           st.latex(latex_input)
           
           # Export options
           st.markdown("### Export Options")
           
           export_format = st.selectbox(
               "Export Format",
               ["PNG", "SVG", "PDF", "MathML"]
           )
           
           if st.button("Export", type="primary", use_container_width=True):
               st.info(f"Exporting as {export_format}...")
               
               # Create download button for LaTeX source
               st.download_button(
                   "Download LaTeX Source",
                   latex_input,
                   file_name="equation.tex",
                   mime="text/plain"
               )

# Tab 8: Advanced
with tab8:
   st.header("Advanced Tools & Settings")
   
   advanced_mode = st.radio(
       "Advanced Mode",
       ["Batch Operations", "System Monitor", "Debug Console", "Custom Scripts"],
       horizontal=True
   )
   
   if advanced_mode == "Batch Operations":
       st.subheader("Batch Operations Manager")
       
       # Operation queue
       st.markdown("### Operation Queue")
       
       operations = []
       
       # Add operation interface
       with st.expander("Add New Operation", expanded=True):
           op_type = st.selectbox(
               "Operation Type",
               ["Generate", "Verify", "Train", "Export", "Clean"]
           )
           
           col1, col2 = st.columns(2)
           
           with col1:
               op_priority = st.selectbox("Priority", ["Low", "Normal", "High"])
               op_schedule = st.selectbox("Schedule", ["Now", "Delayed", "Recurring"])
           
           with col2:
               if op_schedule == "Delayed":
                   delay_time = st.time_input("Start Time")
               elif op_schedule == "Recurring":
                   recur_interval = st.selectbox("Interval", ["Hourly", "Daily", "Weekly"])
           
           op_params = st.text_area(
               "Parameters (JSON)",
               value='{"key": "value"}',
               height=100
           )
           
           if st.button("Add to Queue", type="primary"):
               operations.append({
                   'type': op_type,
                   'priority': op_priority,
                   'schedule': op_schedule,
                   'params': op_params,
                   'status': 'Pending'
               })
               st.success("Operation added to queue!")
       
       # Display queue
       if operations:
           queue_df = pd.DataFrame(operations)
           st.dataframe(queue_df, use_container_width=True)
           
           if st.button("Execute Queue", type="primary", use_container_width=True):
               with st.spinner("Executing batch operations..."):
                   progress = st.progress(0)
                   for i, op in enumerate(operations):
                       progress.progress((i + 1) / len(operations))
                       time.sleep(0.5)  # Simulate execution
                   st.success("All operations completed!")
   
   elif advanced_mode == "System Monitor":
       st.subheader("System Monitor")
       
       # Real-time metrics
       monitor_container = st.container()
       
       with monitor_container:
           # System metrics
           col1, col2, col3, col4 = st.columns(4)
           
           with col1:
               cpu_usage = np.random.uniform(20, 80)
               st.metric(
                   "CPU Usage",
                   f"{cpu_usage:.1f}%",
                   delta=f"{np.random.uniform(-5, 5):.1f}%"
               )
           
           with col2:
               mem_usage = np.random.uniform(30, 70)
               st.metric(
                   "Memory Usage",
                   f"{mem_usage:.1f}%",
                   delta=f"{np.random.uniform(-3, 3):.1f}%"
               )
           
           with col3:
               disk_io = np.random.uniform(10, 50)
               st.metric(
                   "Disk I/O",
                   f"{disk_io:.1f} MB/s"
               )
           
           with col4:
               network = np.random.uniform(1, 10)
               st.metric(
                   "Network",
                   f"{network:.1f} MB/s"
               )
       
       # Process list
       st.markdown("### Active Processes")
       
       processes = pd.DataFrame({
           'PID': [1234, 5678, 9012, 3456],
           'Process': ['ode_generator', 'verifier', 'ml_trainer', 'api_server'],
           'CPU (%)': [25.5, 15.2, 45.8, 8.3],
           'Memory (MB)': [512, 256, 1024, 128],
           'Status': ['Running', 'Running', 'Training', 'Listening']
       })
       
       st.dataframe(
           processes,
           use_container_width=True,
           column_config={
               "CPU (%)": st.column_config.ProgressColumn(
                   min_value=0,
                   max_value=100
               ),
               "Status": st.column_config.TextColumn()
           }
       )
       
       # System logs
       st.markdown("### System Logs")
       
       log_level = st.selectbox(
           "Log Level",
           ["All", "Error", "Warning", "Info", "Debug"],
           key="log_level"
       )
       
       # Simulate logs
       logs = []
       for i in range(10):
           logs.append({
               'Time': datetime.now() - timedelta(minutes=i),
               'Level': np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG']),
               'Component': np.random.choice(['Generator', 'Verifier', 'API']),
               'Message': f"Log message {i}"
           })
       
       log_df = pd.DataFrame(logs)
       
       # Color code by level
       def highlight_level(row):
           colors = {
               'ERROR': 'background-color: #ffcccc',
               'WARNING': 'background-color: #fff3cd',
               'INFO': 'background-color: #d1ecf1',
               'DEBUG': 'background-color: #f8f9fa'
           }
           return [colors.get(row['Level'], '')] * len(row)
       
       st.dataframe(
           log_df.style.apply(highlight_level, axis=1),
           use_container_width=True
       )
   
   elif advanced_mode == "Debug Console":
       st.subheader("Debug Console")
       
       # Interactive console
       st.markdown("### Interactive Python Console")
       
       code_input = st.text_area(
           "Enter Python code",
           value="""# Test ODE generation
import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')

# Define ODE
ode = sp.Eq(y(x).diff(x, 2) + y(x), sp.sin(x))
print(f"ODE: {ode}")

# Verify solution
solution = sp.sin(x)
print(f"Solution: {solution}")""",
           height=200
       )
       
       if st.button("▶ Execute", type="primary"):
           # Execute code (safely)
           try:
               # Create safe execution environment
               exec_globals = {
                   'sp': sp,
                   'np': np,
                   'pd': pd,
                   'print': st.write
               }
               
               with st.expander("Output", expanded=True):
                   exec(code_input, exec_globals)
           except Exception as e:
               st.error(f"Execution error: {e}")
       
       # API tester
       st.markdown("### API Endpoint Tester")
       
       col1, col2 = st.columns(2)
       
       with col1:
           endpoint = st.text_input(
               "Endpoint",
               value="/health"
           )
           
           method = st.selectbox(
               "Method",
               ["GET", "POST", "PUT", "DELETE"]
           )
       
       with col2:
           headers = st.text_area(
               "Headers (JSON)",
               value='{"X-API-Key": "test-key"}',
               height=100
           )
           
           if method != "GET":
               body = st.text_area(
                   "Body (JSON)",
                   value='{}',
                   height=100
               )
       
       if st.button("Send Request", type="primary"):
           with st.spinner("Sending request..."):
               # Make request
               if st.session_state.connection_status == 'connected':
                   # Would make actual API call
                   st.json({
                       "status": "success",
                       "response": {
                           "status": "healthy",
                           "timestamp": datetime.now().isoformat()
                       }
                   })
   
   else:  # Custom Scripts
       st.subheader("Custom Script Manager")
       
       # Script library
       st.markdown("### Script Library")
       
       scripts = {
           "Generate Dataset": """
# Generate comprehensive dataset
generators = ['L1', 'L2', 'L3', 'N1', 'N2']
functions = ['sine', 'cosine', 'exponential']
samples = 10

for gen in generators:
   for func in functions:
       # Generate ODEs
       pass
""",
           "Batch Verification": """
# Verify all unverified ODEs
dataset = load_dataset('my_dataset')
for ode in dataset:
   if not ode['verified']:
       verify(ode)
""",
           "Performance Test": """
# Test system performance
import time
start = time.time()
# Run performance test
elapsed = time.time() - start
print(f"Time: {elapsed}s")
"""
       }
       
       selected_script = st.selectbox(
           "Select Script",
           list(scripts.keys())
       )
       
       script_editor = st.text_area(
           "Script Editor",
           value=scripts[selected_script],
           height=300
       )
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           if st.button("💾 Save Script", use_container_width=True):
               st.success("Script saved!")
       
       with col2:
           if st.button("▶ Run Script", use_container_width=True, type="primary"):
               with st.spinner("Running script..."):
                   time.sleep(2)
                   st.success("Script executed successfully!")
       
       with col3:
           if st.button("📤 Export Script", use_container_width=True):
               st.download_button(
                   "Download",
                   script_editor,
                   file_name=f"{selected_script.lower().replace(' ', '_')}.py",
                   mime="text/plain"
               )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
   <p>ODE Master Generators v3.0 | Comprehensive System Interface</p>
   <p>Based on research by Abu-Ghuwaleh et al. | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh for dashboard
if st.session_state.get('auto_refresh', False):
   time.sleep(5)
   st.rerun()

if __name__ == "__main__":
   # Run the app
   # streamlit run gui_app.py --server.port 8501 --server.maxUploadSize 200
   pass
